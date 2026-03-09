#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium import error as gym_error
from gymnasium.wrappers import RecordVideo

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
from src.utils import obs_to_tensor, resolve_device


def _resolve_step_limit(env: gym.Env, max_steps: int | None, fallback: int = 1000) -> int:
    if max_steps is not None:
        return max(1, int(max_steps))

    candidates = [
        getattr(getattr(env, "spec", None), "max_episode_steps", None),
        getattr(getattr(env.unwrapped, "spec", None), "max_episode_steps", None),
        getattr(env, "_max_episode_steps", None),
    ]
    for value in candidates:
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            return int(value)
    return int(fallback)


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


def _parse_optional_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _infer_action_mode(run_args: dict, policy_state: dict) -> str:
    for key in ("resolved_action_mode", "action_space"):
        raw = run_args.get(key)
        if isinstance(raw, str):
            mode = raw.strip().lower()
            if mode in {"discrete", "continuous"}:
                return mode

    if ("pi_log_std" in policy_state) or any(str(k).startswith("pi_mean.") for k in policy_state):
        return "continuous"
    if any(str(k).startswith("pi.") for k in policy_state):
        return "discrete"
    raise ValueError("Unable to infer action mode from checkpoint args/state.")


def _infer_use_prev_action(run_args: dict, policy_state: dict) -> bool:
    explicit = _parse_optional_bool(run_args.get("use_prev_action"))
    if explicit is not None:
        return explicit
    return any(
        str(key).startswith("f_pol.act_emb.")
        or str(key).startswith("f_pol.act_proj.")
        or str(key).startswith("f_pol.fuse.")
        for key in policy_state
    )


def _infer_use_traces(run_args: dict, checkpoint: dict, policy_state: dict, feat_dim: int) -> bool:
    explicit = _parse_optional_bool(run_args.get("use_traces"))
    if explicit is not None:
        return explicit
    if str(run_args.get("policy", "")).lower() == "recurrent":
        return False
    if checkpoint.get("f_mem_state") is not None:
        return True
    core_weight = policy_state.get("core.net.0.weight")
    if isinstance(core_weight, torch.Tensor):
        return int(core_weight.shape[1]) > int(feat_dim)
    return False


def _resolve_action_spec(
    env: gym.Env, action_mode: str
) -> tuple[int, tuple[int, ...], object | None, object | None]:
    mode = str(action_mode)
    if mode == "discrete":
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError(
                f"Checkpoint expects a discrete action space, got {type(env.action_space)} from eval env."
            )
        return int(env.action_space.n), (), None, None
    if mode == "continuous":
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError(
                f"Checkpoint expects a continuous action space, got {type(env.action_space)} from eval env."
            )
        action_shape = tuple(int(x) for x in env.action_space.shape)
        act_dim = int(math.prod(action_shape))
        action_low = env.action_space.low.reshape(-1).astype("float32", copy=False)
        action_high = env.action_space.high.reshape(-1).astype("float32", copy=False)
        return act_dim, action_shape, action_low, action_high
    raise ValueError(f"Unsupported action mode: {action_mode}")


def _load_checkpoint(checkpoint: str | None, run_dir: str | None, device: str) -> tuple[dict, Path]:
    if checkpoint is None:
        if run_dir is None:
            raise ValueError("Provide either --checkpoint or --run-dir")
        checkpoint_path = Path(run_dir) / "checkpoint.pt"
    else:
        checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
    return payload, checkpoint_path


def _resolve_checkpoint_for_kind(run_dir: Path, checkpoint_kind: str) -> Path:
    if checkpoint_kind == "best":
        candidates = [run_dir / "checkpoint_best.pt", run_dir / "checkpoint.pt"]
    elif checkpoint_kind == "last":
        candidates = [run_dir / "checkpoint_last.pt", run_dir / "checkpoint.pt"]
    else:
        raise ValueError(f"Unsupported checkpoint kind: {checkpoint_kind}")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint found for `{checkpoint_kind}` under: {run_dir}")


def _resolve_output_video_dir(base_dir: Path, run_name: str, overwrite: bool) -> Path:
    video_dir = base_dir / run_name
    if video_dir.exists():
        if overwrite:
            shutil.rmtree(video_dir)
        else:
            suffix = time.strftime("%Y%m%d_%H%M%S")
            candidate = video_dir.parent / f"{run_name}_{suffix}"
            idx = 1
            while candidate.exists():
                candidate = video_dir.parent / f"{run_name}_{suffix}_{idx}"
                idx += 1
            video_dir = candidate
    return video_dir


def _run_policy_recording(
    *,
    payload: dict,
    device: str,
    episodes: int,
    max_steps: int | None,
    deterministic: bool,
    video_dir: Path,
    seed: int,
    target_steps: int | None,
    eval_env_id: str | None,
    eval_stationary: bool,
    eval_fullobs: bool,
    record_all_episodes: bool,
    video_length: int,
) -> tuple[list[float], list[int]]:
    run_args = payload.get("args")
    if not isinstance(run_args, dict):
        raise ValueError("Checkpoint is missing `args` dictionary.")
    policy = str(run_args.get("policy", "amt"))
    if policy == "recurrent":
        return _record_recurrent(
            run_args=run_args,
            checkpoint=payload,
            device=device,
            episodes=episodes,
            max_steps=max_steps,
            deterministic=deterministic,
            video_dir=video_dir,
            seed=seed,
            target_steps=target_steps,
            eval_env_id=eval_env_id,
            eval_stationary=eval_stationary,
            eval_fullobs=eval_fullobs,
            record_all_episodes=record_all_episodes,
            video_length=video_length,
        )
    return _record_feedforward(
        run_args=run_args,
        checkpoint=payload,
        device=device,
        episodes=episodes,
        max_steps=max_steps,
        deterministic=deterministic,
        video_dir=video_dir,
        seed=seed,
        target_steps=target_steps,
        eval_env_id=eval_env_id,
        eval_stationary=eval_stationary,
        eval_fullobs=eval_fullobs,
        record_all_episodes=record_all_episodes,
        video_length=video_length,
    )


def record_benchmark_run_gameplays(
    *,
    run_dir: Path,
    device: str,
    checkpoint_kind: str,
    episodes: int,
    max_steps: int | None,
    deterministic: bool,
    output_dir: Path,
    name: str | None,
    seed: int,
    target_steps: int | None,
    eval_env_id: str | None,
    eval_stationary: bool,
    eval_fullobs: bool,
    record_all_episodes: bool,
    video_length: int,
    overwrite: bool,
) -> None:
    kinds = ["best", "last"] if checkpoint_kind == "both" else [checkpoint_kind]
    base_name = name or run_dir.name
    generated = 0
    seen_checkpoint_paths: set[Path] = set()
    for kind in kinds:
        try:
            checkpoint_path = _resolve_checkpoint_for_kind(run_dir=run_dir, checkpoint_kind=kind)
        except FileNotFoundError as exc:
            if len(kinds) > 1:
                print(f"[warn] skipping `{kind}`: {exc}")
                continue
            raise

        if checkpoint_path in seen_checkpoint_paths:
            print(f"[warn] skipping `{kind}`: duplicate checkpoint path {checkpoint_path}")
            continue
        seen_checkpoint_paths.add(checkpoint_path)

        try:
            payload, _ = _load_checkpoint(checkpoint=str(checkpoint_path), run_dir=None, device=device)
            suffix = f"_{kind}" if len(kinds) > 1 else ""
            run_name = f"{base_name}{suffix}"
            video_dir = _resolve_output_video_dir(base_dir=output_dir, run_name=run_name, overwrite=overwrite)
            returns, lengths = _run_policy_recording(
                payload=payload,
                device=device,
                episodes=episodes,
                max_steps=max_steps,
                deterministic=deterministic,
                video_dir=video_dir,
                seed=seed,
                target_steps=target_steps,
                eval_env_id=eval_env_id,
                eval_stationary=eval_stationary,
                eval_fullobs=eval_fullobs,
                record_all_episodes=record_all_episodes,
                video_length=video_length,
            )
        except Exception as exc:
            if len(kinds) > 1:
                print(f"[warn] skipping `{kind}` due to recording error: {exc}")
                continue
            if isinstance(exc, RuntimeError):
                raise SystemExit(str(exc)) from exc
            raise
        videos = sorted(video_dir.glob("*.mp4"))
        if not videos:
            if len(kinds) > 1:
                print(f"[warn] skipping `{kind}`: no videos generated in {video_dir}")
                continue
            raise SystemExit(f"No videos generated in {video_dir}")
        mean_ret = sum(returns) / max(len(returns), 1)
        mean_len = sum(lengths) / max(len(lengths), 1)
        print(f"[{kind}] checkpoint: {checkpoint_path}")
        print(f"[{kind}] saved {len(videos)} video(s) to: {video_dir}")
        print(f"[{kind}] episode returns: {returns}")
        print(f"[{kind}] episode lengths: {lengths}")
        print(f"[{kind}] mean return={mean_ret:.3f}, mean length={mean_len:.2f}")
        for video in videos:
            print(video)
        generated += 1

    if generated == 0:
        raise SystemExit(
            f"No gameplay generated from run dir: {run_dir}. "
            "Expected at least one of: checkpoint_best.pt, checkpoint_last.pt, checkpoint.pt"
        )


def _make_eval_env(
    run_args: dict,
    seed: int,
    video_dir: Path,
    action_mode: str,
    eval_env_id: str | None,
    eval_stationary: bool,
    eval_fullobs: bool,
    record_all_episodes: bool,
    video_length: int,
) -> gym.Env:
    env_id = eval_env_id or run_args.get("env_id", "CartPole-v1")
    env = gym.make(env_id, render_mode="rgb_array")
    if str(env_id).startswith("CarRacing"):
        if str(action_mode) == "discrete":
            env = DiscreteCarRacingWrapper(env)
        elif str(action_mode) != "continuous":
            raise ValueError(f"Unsupported CarRacing action mode: {action_mode}")
        env = CarRacingPreprocessWrapper(
            env,
            downsample=int(run_args.get("carracing_downsample", 1)),
            grayscale=bool(run_args.get("carracing_grayscale", False)),
        )
    mask_indices = [] if eval_fullobs else _parse_ints(run_args.get("mask_indices", []))
    if mask_indices:
        env = PartialObsWrapper(env, mask_indices)
    phase_len = int(run_args.get("phase_len", 0))
    obs_shift_scale = float(run_args.get("obs_shift_scale", 0.0))
    reward_scale_low = float(run_args.get("reward_scale_low", 1.0))
    reward_scale_high = float(run_args.get("reward_scale_high", 1.0))
    if eval_stationary:
        phase_len = 0
        obs_shift_scale = 0.0
        reward_scale_low = 1.0
        reward_scale_high = 1.0
    if (phase_len > 0) and ((obs_shift_scale > 0.0) or (reward_scale_low != 1.0) or (reward_scale_high != 1.0)):
        env = PiecewiseDriftWrapper(
            env,
            seed=seed,
            phase_len=phase_len,
            obs_shift_scale=obs_shift_scale,
            reward_scale_low=reward_scale_low,
            reward_scale_high=reward_scale_high,
        )
    frame_stack = int(run_args.get("frame_stack", 1))
    if frame_stack > 1:
        env = FrameStackLastAxisWrapper(env, num_stack=frame_stack)
    video_dir.parent.mkdir(parents=True, exist_ok=True)
    episode_trigger = (lambda _: True) if record_all_episodes else (lambda episode_id: episode_id == 0)
    try:
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=episode_trigger,
            video_length=max(0, int(video_length)),
            name_prefix="episode",
        )
    except gym_error.DependencyNotInstalled as exc:
        raise RuntimeError(
            "RecordVideo requires MoviePy. Install with `pip install moviepy` or `pip install \"gymnasium[other]\"`."
        ) from exc
    return env


@torch.no_grad()
def _record_feedforward(
    run_args: dict,
    checkpoint: dict,
    device: str,
    episodes: int,
    max_steps: int | None,
    deterministic: bool,
    video_dir: Path,
    seed: int,
    target_steps: int | None,
    eval_env_id: str | None,
    eval_stationary: bool,
    eval_fullobs: bool,
    record_all_episodes: bool,
    video_length: int,
) -> tuple[list[float], list[int]]:
    policy_state = checkpoint.get("policy_state")
    if not isinstance(policy_state, dict):
        raise ValueError("Checkpoint is missing `policy_state` dictionary.")
    action_mode = _infer_action_mode(run_args=run_args, policy_state=policy_state)

    env = _make_eval_env(
        run_args=run_args,
        seed=seed,
        video_dir=video_dir,
        action_mode=action_mode,
        eval_env_id=eval_env_id,
        eval_stationary=eval_stationary,
        eval_fullobs=eval_fullobs,
        record_all_episodes=record_all_episodes,
        video_length=video_length,
    )
    try:
        assert isinstance(env.observation_space, gym.spaces.Box)
        obs_dim = int(math.prod(env.observation_space.shape))
        act_dim, action_shape, action_low, action_high = _resolve_action_spec(env=env, action_mode=action_mode)
        feat_dim = int(run_args.get("feat_dim", 64))
        use_prev_action = _infer_use_prev_action(run_args=run_args, policy_state=policy_state)
        use_traces = _infer_use_traces(
            run_args=run_args,
            checkpoint=checkpoint,
            policy_state=policy_state,
            feat_dim=feat_dim,
        )
        core_weight = policy_state.get("core.net.0.weight")
        inferred_mem_dim = 0
        if isinstance(core_weight, torch.Tensor):
            inferred_mem_dim = max(int(core_weight.shape[1]) - feat_dim, 0)
        if inferred_mem_dim > 0:
            use_traces = True
        mem_dim = inferred_mem_dim if use_traces else 0
        if use_traces and (mem_dim <= 0):
            raise ValueError("Trace policy checkpoint inferred, but trace memory dimension is zero.")
        if use_traces and ((mem_dim % feat_dim) != 0):
            raise ValueError(f"Incompatible trace memory: mem_dim={mem_dim}, feat_dim={feat_dim}")
        M = (mem_dim // feat_dim) if use_traces else 0

        ac = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_embed_dim=int(run_args.get("act_embed_dim", 16)),
            hidden_dim=int(run_args.get("hidden_dim", 128)),
            feat_dim=feat_dim,
            mem_dim=mem_dim,
            encoder_type=str(run_args.get("encoder", "mlp")),
            obs_shape=tuple(env.observation_space.shape),
            action_type=action_mode,
            use_prev_action=use_prev_action,
            use_traces=use_traces,
        ).to(device)
        ac.load_state_dict(policy_state)
        ac.eval()

        f_mem = None
        if use_traces:
            f_mem = FeatureEncoder(
                obs_dim=obs_dim,
                act_dim=act_dim,
                act_embed_dim=int(run_args.get("act_embed_dim", 16)),
                hidden_dim=int(run_args.get("hidden_dim", 128)),
                feat_dim=feat_dim,
                encoder_type=str(run_args.get("encoder", "mlp")),
                obs_shape=tuple(env.observation_space.shape),
                action_type=action_mode,
                use_prev_action=use_prev_action,
            ).to(device)
        if f_mem is not None:
            if checkpoint.get("f_mem_state") is not None:
                f_mem.load_state_dict(checkpoint["f_mem_state"])
            else:
                f_mem.load_state_dict(ac.f_pol.state_dict())
            f_mem.eval()

        predictor = None
        if use_traces and (checkpoint.get("predictor_state") is not None):
            predictor = Predictor(
                feat_dim=feat_dim,
                act_dim=act_dim,
                hidden_dim=int(run_args.get("hidden_dim", 128)),
                action_type=action_mode,
            ).to(device)
            predictor.load_state_dict(checkpoint["predictor_state"])
            predictor.eval()

        gamma = float(run_args.get("gamma", 0.99))
        lambda_pred = float(run_args.get("lambda_pred", 0.0))
        reset_strategy = str(run_args.get("reset_strategy", "none"))
        reset_long_fraction = float(run_args.get("reset_long_fraction", 0.5))
        alpha_base = None
        alpha_max = None
        long_mask = None
        skip_drift = True
        if use_traces:
            alpha_base_list = _parse_floats(run_args.get("alpha_base", "1.0"))
            alpha_max_list = _parse_floats(run_args.get("alpha_max", "1.0"))
            if not alpha_base_list:
                alpha_base_list = [1.0]
            if not alpha_max_list:
                alpha_max_list = [1.0]
            if len(alpha_base_list) == 1 and M > 1:
                alpha_base_list = alpha_base_list * M
            if len(alpha_max_list) == 1 and M > 1:
                alpha_max_list = alpha_max_list * M
            if len(alpha_base_list) != len(alpha_max_list):
                raise ValueError("alpha_base and alpha_max lengths do not match.")
            if len(alpha_base_list) != M:
                raise ValueError(
                    f"Trace count mismatch: checkpoint implies M={M}, but alpha schedule has {len(alpha_base_list)} values."
                )
            alpha_base = torch.tensor(alpha_base_list, device=device, dtype=torch.float32).unsqueeze(0)
            alpha_max = torch.tensor(alpha_max_list, device=device, dtype=torch.float32).unsqueeze(0)
            skip_drift = (reset_strategy == "none") and torch.allclose(alpha_base, alpha_max)

            long_start = int(math.floor((1.0 - reset_long_fraction) * M))
            long_mask = torch.zeros(M, device=device, dtype=torch.bool)
            if reset_strategy == "partial":
                long_mask[long_start:] = True

        returns: list[float] = []
        lengths: list[int] = []
        step_limit = _resolve_step_limit(env, max_steps=max_steps)
        total_steps = 0
        episode_idx = 0
        while (episode_idx < episodes) and ((target_steps is None) or (total_steps < target_steps)):
            obs_np, _ = env.reset(seed=seed + episode_idx)
            prev_action = (
                init_prev_action(num_envs=1, action_mode=action_mode, act_dim=act_dim, device=device)
                if use_prev_action
                else None
            )
            obs_t = obs_to_tensor(
                obs_np,
                device=device,
                obs_normalization=str(run_args.get("obs_normalization", "auto")),
            ).unsqueeze(0)
            traces = None
            drift = None
            if use_traces:
                traces = torch.zeros((1, M, feat_dim), device=device)
                x_mem0 = encode_mem(f_mem, obs_t, prev_action)
                traces = trace_update(traces, x_mem0, alpha_base.expand(1, -1))
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

            ep_ret = 0.0
            ep_len = 0
            step_budget = step_limit
            if target_steps is not None:
                step_budget = min(step_budget, max(target_steps - total_steps, 0))
            for _ in range(step_budget):
                traces_flat = traces.reshape(1, -1) if use_traces else None
                policy_out, value = ac(obs_t, prev_action, traces_flat)
                action, _logp, _entropy, _max_action_stat = sample_policy_actions(
                    policy_out=policy_out,
                    action_mode=action_mode,
                    deterministic=deterministic,
                )
                env_action = actions_to_env_numpy(
                    actions=action,
                    action_mode=action_mode,
                    action_shape=action_shape,
                    action_low=action_low,
                    action_high=action_high,
                )[0]
                step_action = int(env_action) if action_mode == "discrete" else env_action
                next_obs, reward, terminated, truncated, _ = env.step(step_action)
                done = bool(terminated or truncated)
                rew = torch.tensor([float(reward)], device=device)
                done_t = torch.tensor([done], device=device, dtype=torch.bool)

                next_obs_t = obs_to_tensor(
                    next_obs,
                    device=device,
                    obs_normalization=str(run_args.get("obs_normalization", "auto")),
                ).unsqueeze(0)
                if use_prev_action:
                    next_prev_action = action.clone()
                    if done:
                        if action_mode == "continuous":
                            next_prev_action[0] = 0.0
                        else:
                            next_prev_action[0] = 0
                else:
                    next_prev_action = None

                if use_traces:
                    x_mem_t = encode_mem(f_mem, obs_t, prev_action)
                    x_mem_next = encode_mem(f_mem, next_obs_t, next_prev_action)
                    if skip_drift:
                        traces_next = trace_update(traces, x_mem_next, alpha_base.expand(1, -1))
                    else:
                        _, v_next_prov = ac(
                            next_obs_t,
                            next_prev_action,
                            trace_update(traces, x_mem_next, alpha_base.expand(1, -1)).reshape(1, -1),
                        )

                        delta = rew + gamma * (~done_t).float() * v_next_prov - value
                        pred_err = torch.zeros(1, device=device)
                        if (predictor is not None) and (lambda_pred > 0.0):
                            pred_err = (x_mem_next - predictor(x_mem_t, action)).pow(2).mean(dim=-1)
                        e = delta.abs() + lambda_pred * pred_err

                        gate, reset_event = drift.update(e)
                        reset_event = reset_event & (~done_t)
                        alpha = alpha_base + gate.unsqueeze(-1) * (alpha_max - alpha_base)
                        alpha = alpha.clamp(0.0, 1.0)
                        traces_reset = maybe_reset_traces(traces, reset_event, x_mem_next, reset_strategy, long_mask)
                        traces_next = trace_update(traces_reset, x_mem_next, alpha)

                    if done:
                        if drift is not None:
                            drift.reset_where(done_t)
                        traces_next[done_t] = 0.0
                        traces_next[done_t] = trace_update(
                            traces_next[done_t],
                            x_mem_next[done_t],
                            alpha_base.expand(done_t.sum(), -1),
                        )

                    traces = traces_next
                prev_action = next_prev_action
                obs_t = next_obs_t

                ep_ret += float(reward)
                ep_len += 1
                total_steps += 1
                if done:
                    break
            returns.append(ep_ret)
            lengths.append(ep_len)
            episode_idx += 1
        return returns, lengths
    finally:
        env.close()


@torch.no_grad()
def _record_recurrent(
    run_args: dict,
    checkpoint: dict,
    device: str,
    episodes: int,
    max_steps: int | None,
    deterministic: bool,
    video_dir: Path,
    seed: int,
    target_steps: int | None,
    eval_env_id: str | None,
    eval_stationary: bool,
    eval_fullobs: bool,
    record_all_episodes: bool,
    video_length: int,
) -> tuple[list[float], list[int]]:
    policy_state = checkpoint.get("policy_state")
    if not isinstance(policy_state, dict):
        raise ValueError("Checkpoint is missing `policy_state` dictionary.")
    action_mode = _infer_action_mode(run_args=run_args, policy_state=policy_state)

    env = _make_eval_env(
        run_args=run_args,
        seed=seed,
        video_dir=video_dir,
        action_mode=action_mode,
        eval_env_id=eval_env_id,
        eval_stationary=eval_stationary,
        eval_fullobs=eval_fullobs,
        record_all_episodes=record_all_episodes,
        video_length=video_length,
    )
    try:
        assert isinstance(env.observation_space, gym.spaces.Box)
        obs_dim = int(math.prod(env.observation_space.shape))
        act_dim, action_shape, action_low, action_high = _resolve_action_spec(env=env, action_mode=action_mode)
        use_prev_action = _infer_use_prev_action(run_args=run_args, policy_state=policy_state)

        ac = RecurrentActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_embed_dim=int(run_args.get("act_embed_dim", 16)),
            hidden_dim=int(run_args.get("hidden_dim", 128)),
            feat_dim=int(run_args.get("feat_dim", 64)),
            encoder_type=str(run_args.get("encoder", "mlp")),
            obs_shape=tuple(env.observation_space.shape),
            action_type=action_mode,
            use_prev_action=use_prev_action,
        ).to(device)
        ac.load_state_dict(policy_state)
        ac.eval()

        returns: list[float] = []
        lengths: list[int] = []
        step_limit = _resolve_step_limit(env, max_steps=max_steps)
        total_steps = 0
        episode_idx = 0
        while (episode_idx < episodes) and ((target_steps is None) or (total_steps < target_steps)):
            obs_np, _ = env.reset(seed=seed + episode_idx)
            obs_t = obs_to_tensor(
                obs_np,
                device=device,
                obs_normalization=str(run_args.get("obs_normalization", "auto")),
            ).unsqueeze(0)
            prev_action = (
                init_prev_action(num_envs=1, action_mode=action_mode, act_dim=act_dim, device=device)
                if use_prev_action
                else None
            )
            hidden = ac.init_hidden(batch_size=1, device=device)

            ep_ret = 0.0
            ep_len = 0
            step_budget = step_limit
            if target_steps is not None:
                step_budget = min(step_budget, max(target_steps - total_steps, 0))
            for _ in range(step_budget):
                policy_out, _, hidden = ac(obs_t, prev_action, hidden)
                action, _logp, _entropy, _max_action_stat = sample_policy_actions(
                    policy_out=policy_out,
                    action_mode=action_mode,
                    deterministic=deterministic,
                )
                env_action = actions_to_env_numpy(
                    actions=action,
                    action_mode=action_mode,
                    action_shape=action_shape,
                    action_low=action_low,
                    action_high=action_high,
                )[0]
                step_action = int(env_action) if action_mode == "discrete" else env_action
                next_obs, reward, terminated, truncated, _ = env.step(step_action)
                done = bool(terminated or truncated)
                obs_t = obs_to_tensor(
                    next_obs,
                    device=device,
                    obs_normalization=str(run_args.get("obs_normalization", "auto")),
                ).unsqueeze(0)
                if use_prev_action:
                    prev_action = action.clone()
                    if done:
                        if action_mode == "continuous":
                            prev_action[0] = 0.0
                        else:
                            prev_action[0] = 0
                else:
                    prev_action = None
                ep_ret += float(reward)
                ep_len += 1
                total_steps += 1
                if done:
                    break
            returns.append(ep_ret)
            lengths.append(ep_len)
            episode_idx += 1
        return returns, lengths
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Record gameplay videos from a training checkpoint.")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.pt")
    p.add_argument("--run-dir", type=str, default=None, help="Run directory containing checkpoint.pt")
    p.add_argument(
        "--checkpoint-kind",
        type=str,
        default="last",
        choices=["last", "best", "both"],
        help="Checkpoint selection when --run-dir is provided. `both` records best + last.",
    )
    default_device = os.environ.get("AMT_DEVICE") or os.environ.get("DEVICE") or "cuda"
    p.add_argument("--device", type=str, default=default_device)
    p.add_argument("--cuda-id", type=int, default=None, help="CUDA device index override (e.g., 0-7).")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--target-steps",
        type=int,
        default=None,
        help="Total inference steps across episodes before stopping (useful for longer videos).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory. Defaults to <run-dir>/gameplay when --run-dir is set, else reports/gameplay.",
    )
    p.add_argument("--name", type=str, default=None, help="Optional output subdir name.")
    p.add_argument("--eval-env-id", type=str, default=None, help="Optional environment override for testing/inference.")
    p.add_argument(
        "--eval-stationary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force stationary eval (disable drift wrapper regardless of training config).",
    )
    p.add_argument(
        "--eval-fullobs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force full-observation eval (disable observation masking).",
    )
    p.add_argument(
        "--record-all-episodes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record every episode. Disable to record only the first episode trigger.",
    )
    p.add_argument(
        "--video-length",
        type=int,
        default=0,
        help="RecordVideo fixed length in frames (0 = full episodes). Combine with --target-steps for long single clips.",
    )
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the same output folder name. If disabled, append a timestamp suffix.",
    )
    args = p.parse_args()
    if args.target_steps is not None and args.target_steps <= 0:
        raise ValueError("--target-steps must be > 0 when provided.")
    if str(args.device).startswith("cuda") and args.cuda_id is None:
        raise ValueError(
            "GPU selection is required. Pass `--cuda-id <index>` in the command, e.g. `--device cuda --cuda-id 0`."
        )
    device = resolve_device(args.device, args.cuda_id)
    if args.checkpoint is not None and args.checkpoint_kind == "both":
        raise ValueError("`--checkpoint-kind both` requires `--run-dir` so best/last checkpoints can be resolved.")

    # Headless servers may not have an audio device; keep rendering silent.
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.run_dir is not None:
        output_dir = Path(args.run_dir) / "gameplay"
    else:
        output_dir = Path("reports/gameplay")

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        record_benchmark_run_gameplays(
            run_dir=run_dir,
            device=device,
            checkpoint_kind=args.checkpoint_kind,
            episodes=args.episodes,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            output_dir=output_dir,
            name=args.name,
            seed=args.seed,
            target_steps=args.target_steps,
            eval_env_id=args.eval_env_id,
            eval_stationary=args.eval_stationary,
            eval_fullobs=args.eval_fullobs,
            record_all_episodes=args.record_all_episodes,
            video_length=args.video_length,
            overwrite=args.overwrite,
        )
        return

    payload, checkpoint_path = _load_checkpoint(checkpoint=args.checkpoint, run_dir=None, device=device)
    run_name = args.name or checkpoint_path.stem
    video_dir = _resolve_output_video_dir(base_dir=output_dir, run_name=run_name, overwrite=args.overwrite)
    try:
        returns, lengths = _run_policy_recording(
            payload=payload,
            device=device,
            episodes=args.episodes,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            video_dir=video_dir,
            seed=args.seed,
            target_steps=args.target_steps,
            eval_env_id=args.eval_env_id,
            eval_stationary=args.eval_stationary,
            eval_fullobs=args.eval_fullobs,
            record_all_episodes=args.record_all_episodes,
            video_length=args.video_length,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No videos generated in {video_dir}")

    mean_ret = sum(returns) / max(len(returns), 1)
    mean_len = sum(lengths) / max(len(lengths), 1)
    print(f"checkpoint: {checkpoint_path}")
    print(f"Saved {len(videos)} video(s) to: {video_dir}")
    print(f"Episode returns: {returns}")
    print(f"Episode lengths: {lengths}")
    print(f"Mean return={mean_ret:.3f}, mean length={mean_len:.2f}")
    for video in videos:
        print(video)


if __name__ == "__main__":
    main()
