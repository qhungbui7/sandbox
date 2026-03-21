#!/usr/bin/env python3
"""Evaluate recovery time after phase transitions using a trained checkpoint."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.amt import DriftMonitor, encode_mem, maybe_reset_traces, trace_update
from src.action_utils import actions_to_env_numpy, init_prev_action, sample_policy_actions
from src.envs import (
    CarRacingPreprocessWrapper,
    DiscreteCarRacingWrapper,
    FrameStackLastAxisWrapper,
    PartialObsWrapper,
    PiecewiseDriftWrapper,
)
from src.models import ActorCritic, FeatureEncoder, Predictor, RecurrentActorCritic
from src.utils import obs_to_tensor, resolve_device, set_seed


def _parse_floats(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    return [float(value)]


def _parse_ints(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(value)]


def _load_checkpoint(run_dir: str | None, checkpoint: str | None, device: str) -> dict:
    if checkpoint is not None:
        path = Path(checkpoint)
    elif run_dir is not None:
        path = Path(run_dir) / "checkpoint.pt"
    else:
        raise ValueError("Provide either --run-dir or --checkpoint")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def _make_eval_env(
    run_args: dict,
    seed: int,
    phase_len: int,
    action_mode: str,
) -> gym.Env:
    env_id = run_args.get("env_id", "CartPole-v1")
    env = gym.make(env_id)
    if str(env_id).startswith("CarRacing"):
        if str(action_mode) == "discrete":
            env = DiscreteCarRacingWrapper(env)
        env = CarRacingPreprocessWrapper(
            env,
            downsample=int(run_args.get("carracing_downsample", 1)),
            grayscale=bool(run_args.get("carracing_grayscale", False)),
        )
    mask_indices = _parse_ints(run_args.get("mask_indices", []))
    if mask_indices:
        env = PartialObsWrapper(env, mask_indices)
    obs_shift_scale = float(run_args.get("obs_shift_scale", 0.1))
    reward_scale_low = float(run_args.get("reward_scale_low", 0.8))
    reward_scale_high = float(run_args.get("reward_scale_high", 1.2))
    env = PiecewiseDriftWrapper(
        env,
        seed=seed,
        phase_len=phase_len,
        obs_shift_scale=obs_shift_scale,
        reward_scale_low=reward_scale_low,
        reward_scale_high=reward_scale_high,
        carry_phase=True,
    )
    frame_stack = int(run_args.get("frame_stack", 1))
    if frame_stack > 1:
        env = FrameStackLastAxisWrapper(env, num_stack=frame_stack)
    return env


def _infer_action_mode(run_args: dict, policy_state: dict) -> str:
    for key in ("resolved_action_mode", "action_space"):
        raw = run_args.get(key)
        if isinstance(raw, str) and raw.strip().lower() in {"discrete", "continuous"}:
            return raw.strip().lower()
    if ("pi_log_std" in policy_state) or any(str(k).startswith("pi_mean.") for k in policy_state):
        return "continuous"
    return "discrete"


def _resolve_action_spec(env: gym.Env, action_mode: str):
    if action_mode == "discrete":
        return int(env.action_space.n), (), None, None
    action_shape = tuple(int(x) for x in env.action_space.shape)
    act_dim = int(math.prod(action_shape))
    return act_dim, action_shape, env.action_space.low.reshape(-1).astype("float32"), env.action_space.high.reshape(-1).astype("float32")


@torch.no_grad()
def evaluate_recovery_feedforward(
    payload: dict,
    device: str,
    phase_len: int,
    eval_steps: int,
    window_size: int,
    seed: int,
) -> dict:
    run_args = payload["args"]
    policy_state = payload["policy_state"]
    action_mode = _infer_action_mode(run_args, policy_state)

    env = _make_eval_env(run_args, seed=seed, phase_len=phase_len, action_mode=action_mode)
    obs_dim = int(math.prod(env.observation_space.shape))
    act_dim, action_shape, action_low, action_high = _resolve_action_spec(env, action_mode)
    feat_dim = int(run_args.get("feat_dim", 64))

    use_prev_action = any(
        str(k).startswith("f_pol.act_emb.") or str(k).startswith("f_pol.fuse.")
        for k in policy_state
    )
    core_weight = policy_state.get("core.net.0.weight")
    inferred_mem_dim = max(int(core_weight.shape[1]) - feat_dim, 0) if isinstance(core_weight, torch.Tensor) else 0
    use_traces = inferred_mem_dim > 0
    mem_dim = inferred_mem_dim if use_traces else 0
    M = (mem_dim // feat_dim) if use_traces else 0

    ac = ActorCritic(
        obs_dim=obs_dim, act_dim=act_dim,
        act_embed_dim=int(run_args.get("act_embed_dim", 16)),
        hidden_dim=int(run_args.get("hidden_dim", 128)),
        feat_dim=feat_dim, mem_dim=mem_dim,
        encoder_type=str(run_args.get("encoder", "mlp")),
        obs_shape=tuple(env.observation_space.shape),
        action_type=action_mode,
        use_prev_action=use_prev_action, use_traces=use_traces,
    ).to(device)
    ac.load_state_dict(policy_state)
    ac.eval()

    f_mem = None
    if use_traces:
        f_mem = FeatureEncoder(
            obs_dim=obs_dim, act_dim=act_dim,
            act_embed_dim=int(run_args.get("act_embed_dim", 16)),
            hidden_dim=int(run_args.get("hidden_dim", 128)),
            feat_dim=feat_dim,
            encoder_type=str(run_args.get("encoder", "mlp")),
            obs_shape=tuple(env.observation_space.shape),
            action_type=action_mode,
            use_prev_action=use_prev_action,
        ).to(device)
        if payload.get("f_mem_state") is not None:
            f_mem.load_state_dict(payload["f_mem_state"])
        else:
            f_mem.load_state_dict(ac.f_pol.state_dict())
        f_mem.eval()

    predictor = None
    if use_traces and payload.get("predictor_state") is not None:
        predictor = Predictor(
            feat_dim=feat_dim, act_dim=act_dim,
            hidden_dim=int(run_args.get("hidden_dim", 128)),
            action_type=action_mode,
        ).to(device)
        predictor.load_state_dict(payload["predictor_state"])
        predictor.eval()

    gamma = float(run_args.get("gamma", 0.99))
    lambda_pred = float(run_args.get("lambda_pred", 0.0))
    reset_strategy = str(run_args.get("reset_strategy", "none"))
    reset_long_fraction = float(run_args.get("reset_long_fraction", 0.5))

    alpha_base = alpha_max = long_mask = drift = None
    skip_drift = True
    if use_traces:
        ab = _parse_floats(run_args.get("alpha_base", "1.0"))
        am = _parse_floats(run_args.get("alpha_max", "1.0"))
        if len(ab) == 1 and M > 1:
            ab = ab * M
        if len(am) == 1 and M > 1:
            am = am * M
        alpha_base = torch.tensor(ab, device=device, dtype=torch.float32).unsqueeze(0)
        alpha_max = torch.tensor(am, device=device, dtype=torch.float32).unsqueeze(0)
        skip_drift = (reset_strategy == "none") and torch.allclose(alpha_base, alpha_max)
        long_start = int(math.floor((1.0 - reset_long_fraction) * M))
        long_mask = torch.zeros(M, device=device, dtype=torch.bool)
        if reset_strategy == "partial":
            long_mask[long_start:] = True
        if not skip_drift:
            drift = DriftMonitor(
                num_envs=1,
                rho_s=float(run_args.get("rho_s", 0.1)),
                rho_l=float(run_args.get("rho_l", 0.01)),
                beta=float(run_args.get("beta", 0.01)),
                tau_soft=float(run_args.get("tau_soft", 1.0)),
                kappa=float(run_args.get("kappa", 0.5)),
                tau_on=float(run_args.get("tau_on", 2.5)),
                tau_off=float(run_args.get("tau_off", 1.5)),
                K=int(run_args.get("K", 5)),
                cooldown_steps=int(run_args.get("cooldown_steps", 200)),
                warmup_steps=int(run_args.get("warmup_steps", 1000)),
                device=device,
            )

    obs_norm = str(run_args.get("obs_normalization", "auto"))
    drift_signal = str(run_args.get("drift_signal", "combined"))
    step_rewards = []
    step_phases = []
    step_resets = []

    obs_np, info = env.reset(seed=seed)
    prev_action = init_prev_action(1, action_mode, act_dim, device) if use_prev_action else None
    traces = None
    if use_traces:
        traces = torch.zeros((1, M, feat_dim), device=device)
        obs_t = obs_to_tensor(obs_np, device=device, obs_normalization=obs_norm).unsqueeze(0)
        x_mem0 = encode_mem(f_mem, obs_t, prev_action)
        traces = trace_update(traces, x_mem0, alpha_base.expand(1, -1))

    for step_idx in range(eval_steps):
        obs_t = obs_to_tensor(obs_np, device=device, obs_normalization=obs_norm).unsqueeze(0)
        traces_flat = traces.reshape(1, -1) if traces is not None else None
        policy_out, value = ac(obs_t, prev_action, traces_flat)
        action, logp, entropy, latent = sample_policy_actions(
            policy_out, action_mode, deterministic=True,
            action_shape=action_shape, action_low=action_low, action_high=action_high,
        )
        action_np = actions_to_env_numpy(action, action_mode, action_shape=action_shape)
        if action_np.ndim > 1:
            action_np = action_np[0]

        next_obs_np, reward, terminated, truncated, info = env.step(action_np)
        step_rewards.append(float(reward))
        step_phases.append(int(info.get("phase", 0)))

        reset_event = False
        if use_traces:
            next_obs_t = obs_to_tensor(next_obs_np, device=device, obs_normalization=obs_norm).unsqueeze(0)
            x_mem_t = encode_mem(f_mem, obs_t, prev_action)
            next_pa = action.long() if action_mode == "discrete" else action
            x_mem_next = encode_mem(f_mem, next_obs_t, next_pa)

            if skip_drift:
                traces = trace_update(traces, x_mem_next, alpha_base.expand(1, -1))
            else:
                _, v_prov = ac(
                    next_obs_t, next_pa,
                    trace_update(traces, x_mem_next, alpha_base.expand(1, -1)).reshape(1, -1),
                )
                delta_prov = reward + gamma * (not (terminated or truncated)) * v_prov.squeeze() - value.squeeze()
                pred_err = torch.zeros(1, device=device)
                if predictor is not None and lambda_pred > 0.0:
                    x_hat = predictor(x_mem_t, action)
                    pred_err = (x_mem_next - x_hat).pow(2).mean(dim=-1)
                if drift_signal == "prediction_only":
                    e = pred_err
                elif drift_signal == "td_only":
                    e = delta_prov.abs()
                else:
                    e = delta_prov.abs() + lambda_pred * pred_err
                gate, re = drift.update(e)
                reset_event = bool(re.item())
                if reset_event and not (terminated or truncated):
                    traces = maybe_reset_traces(traces, re, x_mem_next, reset_strategy, long_mask)
                alpha = alpha_base + gate.unsqueeze(-1) * (alpha_max - alpha_base)
                alpha = alpha.clamp(0.0, 1.0)
                traces = trace_update(traces, x_mem_next, alpha)

        step_resets.append(reset_event)
        if use_prev_action:
            prev_action = action.long() if action_mode == "discrete" else action

        if terminated or truncated:
            obs_np, info = env.reset()
            if use_prev_action:
                prev_action = init_prev_action(1, action_mode, act_dim, device)
            if use_traces:
                traces = torch.zeros((1, M, feat_dim), device=device)
                obs_t_new = obs_to_tensor(obs_np, device=device, obs_normalization=obs_norm).unsqueeze(0)
                x_mem0 = encode_mem(f_mem, obs_t_new, prev_action)
                traces = trace_update(traces, x_mem0, alpha_base.expand(1, -1))
                if drift is not None:
                    drift.reset_where(torch.ones(1, device=device, dtype=torch.bool))
        else:
            obs_np = next_obs_np

    env.close()

    transitions = []
    prev_phase = step_phases[0] if step_phases else 0
    for i, ph in enumerate(step_phases):
        if ph != prev_phase:
            transitions.append(i)
            prev_phase = ph

    recovery_times = []
    for t_idx in transitions:
        pre_start = max(0, t_idx - window_size)
        pre_rewards = step_rewards[pre_start:t_idx]
        if not pre_rewards:
            continue
        baseline = sum(pre_rewards) / len(pre_rewards)
        recovered = False
        for j in range(t_idx, min(t_idx + 10 * window_size, len(step_rewards) - window_size + 1)):
            window = step_rewards[j:j + window_size]
            if sum(window) / len(window) >= baseline:
                recovery_times.append(j - t_idx)
                recovered = True
                break
        if not recovered:
            recovery_times.append(-1)

    valid_times = [t for t in recovery_times if t >= 0]
    return {
        "phase_len": phase_len,
        "eval_steps": eval_steps,
        "window_size": window_size,
        "seed": seed,
        "num_transitions": len(transitions),
        "transition_steps": transitions,
        "recovery_times": recovery_times,
        "mean_recovery_time": float(np.mean(valid_times)) if valid_times else -1.0,
        "median_recovery_time": float(np.median(valid_times)) if valid_times else -1.0,
        "total_resets": sum(step_resets),
        "reset_rate": sum(step_resets) / max(len(step_resets), 1),
    }


@torch.no_grad()
def evaluate_recovery_recurrent(
    payload: dict,
    device: str,
    phase_len: int,
    eval_steps: int,
    window_size: int,
    seed: int,
) -> dict:
    run_args = payload["args"]
    policy_state = payload["policy_state"]
    action_mode = _infer_action_mode(run_args, policy_state)

    env = _make_eval_env(run_args, seed=seed, phase_len=phase_len, action_mode=action_mode)
    obs_dim = int(math.prod(env.observation_space.shape))
    act_dim, action_shape, action_low, action_high = _resolve_action_spec(env, action_mode)
    feat_dim = int(run_args.get("feat_dim", 64))
    hidden_dim = int(run_args.get("hidden_dim", 128))

    use_prev_action = any(
        str(k).startswith("f_pol.act_emb.") or str(k).startswith("f_pol.fuse.")
        for k in policy_state
    )

    ac = RecurrentActorCritic(
        obs_dim=obs_dim, act_dim=act_dim,
        act_embed_dim=int(run_args.get("act_embed_dim", 16)),
        hidden_dim=hidden_dim, feat_dim=feat_dim,
        encoder_type=str(run_args.get("encoder", "mlp")),
        obs_shape=tuple(env.observation_space.shape),
        action_type=action_mode,
        use_prev_action=use_prev_action,
    ).to(device)
    ac.load_state_dict(policy_state)
    ac.eval()

    obs_norm = str(run_args.get("obs_normalization", "auto"))
    step_rewards = []
    step_phases = []

    obs_np, info = env.reset(seed=seed)
    prev_action = init_prev_action(1, action_mode, act_dim, device) if use_prev_action else None
    h = torch.zeros(1, 1, feat_dim, device=device)
    c = torch.zeros(1, 1, feat_dim, device=device)

    for step_idx in range(eval_steps):
        obs_t = obs_to_tensor(obs_np, device=device, obs_normalization=obs_norm).unsqueeze(0)
        policy_out, value, (h, c) = ac(obs_t.unsqueeze(0), prev_action.unsqueeze(0) if prev_action is not None else None, (h, c))
        policy_out = policy_out.squeeze(0)
        action, logp, entropy, latent = sample_policy_actions(
            policy_out, action_mode, deterministic=True,
            action_shape=action_shape, action_low=action_low, action_high=action_high,
        )
        action_np = actions_to_env_numpy(action, action_mode, action_shape=action_shape)
        if action_np.ndim > 1:
            action_np = action_np[0]

        next_obs_np, reward, terminated, truncated, info = env.step(action_np)
        step_rewards.append(float(reward))
        step_phases.append(int(info.get("phase", 0)))

        if use_prev_action:
            prev_action = action.long() if action_mode == "discrete" else action

        if terminated or truncated:
            obs_np, info = env.reset()
            if use_prev_action:
                prev_action = init_prev_action(1, action_mode, act_dim, device)
            h = torch.zeros(1, 1, feat_dim, device=device)
            c = torch.zeros(1, 1, feat_dim, device=device)
        else:
            obs_np = next_obs_np

    env.close()

    transitions = []
    prev_phase = step_phases[0] if step_phases else 0
    for i, ph in enumerate(step_phases):
        if ph != prev_phase:
            transitions.append(i)
            prev_phase = ph

    recovery_times = []
    for t_idx in transitions:
        pre_start = max(0, t_idx - window_size)
        pre_rewards = step_rewards[pre_start:t_idx]
        if not pre_rewards:
            continue
        baseline = sum(pre_rewards) / len(pre_rewards)
        recovered = False
        for j in range(t_idx, min(t_idx + 10 * window_size, len(step_rewards) - window_size + 1)):
            window = step_rewards[j:j + window_size]
            if sum(window) / len(window) >= baseline:
                recovery_times.append(j - t_idx)
                recovered = True
                break
        if not recovered:
            recovery_times.append(-1)

    valid_times = [t for t in recovery_times if t >= 0]
    return {
        "phase_len": phase_len,
        "eval_steps": eval_steps,
        "window_size": window_size,
        "seed": seed,
        "num_transitions": len(transitions),
        "transition_steps": transitions,
        "recovery_times": recovery_times,
        "mean_recovery_time": float(np.mean(valid_times)) if valid_times else -1.0,
        "median_recovery_time": float(np.median(valid_times)) if valid_times else -1.0,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate recovery time after phase transitions")
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--phase-len", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=5000)
    p.add_argument("--window-size", type=int, default=20)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--cuda-id", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device, args.cuda_id)
    payload = _load_checkpoint(args.run_dir, args.checkpoint, device)

    run_args = payload["args"]
    policy = str(run_args.get("policy", "amt"))

    if policy == "recurrent":
        result = evaluate_recovery_recurrent(
            payload=payload, device=device, phase_len=args.phase_len,
            eval_steps=args.eval_steps, window_size=args.window_size, seed=args.seed,
        )
    else:
        result = evaluate_recovery_feedforward(
            payload=payload, device=device, phase_len=args.phase_len,
            eval_steps=args.eval_steps, window_size=args.window_size, seed=args.seed,
        )

    result["policy"] = policy
    result["env_id"] = run_args.get("env_id", "unknown")

    output_path = args.output
    if output_path is None and args.run_dir is not None:
        output_path = str(Path(args.run_dir) / "recovery_eval.json")
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Recovery evaluation saved to: {output_path}")

    print(f"Transitions: {result['num_transitions']}")
    print(f"Recovery times: {result['recovery_times']}")
    print(f"Mean recovery time: {result['mean_recovery_time']:.1f}")
    print(f"Median recovery time: {result['median_recovery_time']:.1f}")


if __name__ == "__main__":
    main()
