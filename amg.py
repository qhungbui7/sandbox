#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

import gymnasium as gym
import wandb

from src.amt import DriftMonitor, ema_update_, encode_mem, rollout, rollout_recurrent, trace_update
from src.envs import EnvPool, PartialObsWrapper, PiecewiseDriftWrapper
from src.models import ActorCritic, FeatureEncoder, Predictor, RecurrentActorCritic
from src.ppo import ppo_update, ppo_update_recurrent
from src.utils import load_env_file, resolve_device, set_seed


def parse_floats(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    return [float(value)]


def parse_ints(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(value)]


def parse_strs(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(value).strip()] if str(value).strip() else []


def init_wandb(args):
    if not args.wandb:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed. Run `pip install wandb` or disable --wandb.")
    tags = parse_strs(args.wandb_tags)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=tags or None,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config=vars(args),
    )


def recent_return_stats(episode_returns: list[tuple[float, int]], window: int = 50) -> tuple[float | None, float | None]:
    if not episode_returns:
        return None, None
    recent = episode_returns[-window:]
    mean_ret = sum(r for r, _ in recent) / len(recent)
    mean_len = sum(l for _, l in recent) / len(recent)
    return mean_ret, mean_len


def train_recurrent(
    args,
    envs: EnvPool,
    obs0: np.ndarray,
    obs_dim: int,
    act_dim: int,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_scaler: torch.amp.GradScaler | None,
):
    ac = RecurrentActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=args.act_embed_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=args.feat_dim,
    ).to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)

    updates = args.total_steps // (args.num_envs * args.horizon)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    prev_action = torch.zeros(args.num_envs, device=device, dtype=torch.int64)
    hidden = ac.init_hidden(args.num_envs, device)
    obs = obs0

    wandb_run = init_wandb(args)

    try:
        for upd in range(updates):
            batch, obs, prev_action, hidden = rollout_recurrent(
                envs=envs,
                ac=ac,
                device=device,
                horizon=args.horizon,
                gamma=args.gamma,
                obs=obs,
                prev_action=prev_action,
                hidden=hidden,
            )

            ppo_stats = ppo_update_recurrent(
                ac=ac,
                opt=opt,
                batch=batch,
                clip_coef=args.clip_coef,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                epochs=args.epochs,
                lam=args.gae_lam,
                gamma=args.gamma,
                generator=rng,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                grad_scaler=grad_scaler,
            )

            metrics = {
                "loop/update": upd + 1,
                "loop/frames": (upd + 1) * args.num_envs * args.horizon,
            }
            metrics.update({f"loss/{k}": v for k, v in ppo_stats.items()})

            mean_ret, mean_len = recent_return_stats(envs.episode_returns, window=50)
            if mean_ret is not None:
                metrics["train/ret50"] = mean_ret
                metrics["train/len50"] = mean_len

            if (upd + 1) % args.log_interval == 0 and mean_ret is not None:
                print(
                    f"update={upd+1:04d}  episodes={len(envs.episode_returns):06d}  "
                    f"ret50={mean_ret:8.2f}  len50={mean_len:6.1f}  "
                    f"kl={ppo_stats['approx_kl']:7.4f}  clipfrac={ppo_stats['clipfrac']:6.3f}"
                )

            if wandb_run is not None:
                wandb_run.log(metrics, step=metrics["loop/frames"])
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def make_env_fn(
    env_id: str,
    seed: int,
    mask_indices: list[int],
    phase_len: int,
    obs_shift_scale: float,
    reward_scale_low: float,
    reward_scale_high: float,
):
    def _thunk():
        env = gym.make(env_id)
        if len(mask_indices) > 0:
            env = PartialObsWrapper(env, mask_indices)
        if (phase_len > 0) and ((obs_shift_scale > 0.0) or (reward_scale_low != 1.0) or (reward_scale_high != 1.0)):
            env = PiecewiseDriftWrapper(
                env,
                seed=seed,
                phase_len=phase_len,
                obs_shift_scale=obs_shift_scale,
                reward_scale_low=reward_scale_low,
                reward_scale_high=reward_scale_high,
            )
        return env

    return _thunk


def main():
    load_env_file()
    default_device = os.environ.get("AMT_DEVICE") or os.environ.get("DEVICE") or "cpu"

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML config file (overrides positional config).")
    p.add_argument("config_path", nargs="?", default=None, help="YAML config file (positional).")
    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=default_device)
    p.add_argument("--cuda-id", type=int, default=None, help="CUDA device index override (e.g., 0-7).")
    p.add_argument("--policy", type=str, default="amt", choices=["amt", "recurrent", "ff"], help="Policy architecture.")

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--feat-dim", type=int, default=64)
    p.add_argument("--act-embed-dim", type=int, default=16)

    p.add_argument("--alpha-base", type=str, default="0.5,0.1,0.01")
    p.add_argument("--alpha-max", type=str, default="1.0,0.5,0.2")

    p.add_argument("--rho-s", type=float, default=0.1)
    p.add_argument("--rho-l", type=float, default=0.01)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--tau-soft", type=float, default=1.0)
    p.add_argument("--kappa", type=float, default=0.5)
    p.add_argument("--tau-on", type=float, default=2.5)
    p.add_argument("--tau-off", type=float, default=1.5)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--cooldown-steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=1000)

    p.add_argument("--reset-strategy", type=str, default="partial", choices=["none", "zero", "obs", "partial"])
    p.add_argument("--reset-long-fraction", type=float, default=0.34)

    p.add_argument("--lambda-pred", type=float, default=0.0)
    p.add_argument("--pred-coef", type=float, default=0.0)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lam", type=float, default=0.95)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--ema-tau", type=float, default=0.995)
    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--mask-indices", type=str, default="1,3")
    p.add_argument("--phase-len", type=int, default=2000)
    p.add_argument("--obs-shift-scale", type=float, default=0.1)
    p.add_argument("--reward-scale-low", type=float, default=0.7)
    p.add_argument("--reward-scale-high", type=float, default=1.3)

    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", type=str, default="amt")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-dir", type=str, default=None)

    p.add_argument("--amp", action="store_true", help="Enable mixed precision training (CUDA only).")
    p.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])

    # two-pass parse to apply YAML defaults then allow CLI overrides
    partial_args, _ = p.parse_known_args()
    if partial_args.config and partial_args.config_path and (partial_args.config != partial_args.config_path):
        raise ValueError("Provide a single config path (either positional or --config).")
    config_input = partial_args.config or partial_args.config_path
    if config_input is not None:
        config_path = Path(config_input)
        if config_path.exists():
            cfg = yaml.safe_load(config_path.read_text()) or {}
            if not isinstance(cfg, dict):
                raise ValueError("Config file must map keys to values.")
            p.set_defaults(**{k: v for k, v in cfg.items() if hasattr(partial_args, k)})
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    args = p.parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device, args.cuda_id)
    device_type = torch.device(device).type
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    use_amp = args.amp and device_type == "cuda"
    if args.amp and not use_amp:
        print("Warning: --amp requested but device is not CUDA; disabling mixed precision.")
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp and (amp_dtype == torch.float16))

    if args.policy == "ff":
        args.alpha_base = "1.0"
        args.alpha_max = "1.0"
        args.reset_strategy = "none"
        args.lambda_pred = 0.0
        args.pred_coef = 0.0
    alpha_base_list = parse_floats(args.alpha_base)
    alpha_max_list = parse_floats(args.alpha_max)
    assert len(alpha_base_list) == len(alpha_max_list)
    M = len(alpha_base_list)

    env_fns = [
        make_env_fn(
            env_id=args.env_id,
            seed=args.seed + 10_000 * i,
            mask_indices=parse_ints(args.mask_indices),
            phase_len=args.phase_len,
            obs_shift_scale=args.obs_shift_scale,
            reward_scale_low=args.reward_scale_low,
            reward_scale_high=args.reward_scale_high,
        )
        for i in range(args.num_envs)
    ]
    envs = EnvPool(env_fns)

    obs0, _ = envs.reset(seed=args.seed)
    assert isinstance(envs.single_observation_space, gym.spaces.Box)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(envs.single_action_space.n)
    feat_dim = args.feat_dim
    mem_dim = M * feat_dim

    if args.policy == "recurrent":
        train_recurrent(
            args=args,
            envs=envs,
            obs0=obs0,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
        )
        return

    ac = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=args.act_embed_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=feat_dim,
        mem_dim=mem_dim,
    ).to(device)

    f_mem = FeatureEncoder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=args.act_embed_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=feat_dim,
    ).to(device)
    f_mem.load_state_dict(ac.f_pol.state_dict())

    predictor = None
    if (args.lambda_pred > 0.0) or (args.pred_coef > 0.0):
        predictor = Predictor(feat_dim=feat_dim, act_dim=act_dim, hidden_dim=args.hidden_dim).to(device)

    alpha_base = torch.tensor(alpha_base_list, device=device, dtype=torch.float32).unsqueeze(0)  # (1, M)
    alpha_max = torch.tensor(alpha_max_list, device=device, dtype=torch.float32).unsqueeze(0)

    traces = torch.zeros((args.num_envs, M, feat_dim), device=device)
    prev_action = torch.zeros(args.num_envs, device=device, dtype=torch.int64)

    with torch.no_grad():
        obs0_t = torch.as_tensor(obs0, device=device, dtype=torch.float32)
        x_mem0 = encode_mem(f_mem, obs0_t, prev_action)
        traces = trace_update(traces, x_mem0, alpha_base.expand(args.num_envs, -1))

    drift = DriftMonitor(
        num_envs=args.num_envs,
        rho_s=args.rho_s,
        rho_l=args.rho_l,
        beta=args.beta,
        tau_soft=args.tau_soft,
        kappa=args.kappa,
        tau_on=args.tau_on,
        tau_off=args.tau_off,
        K=args.K,
        cooldown_steps=args.cooldown_steps,
        warmup_steps=args.warmup_steps,
        device=device,
    )

    params = list(ac.parameters())
    if predictor is not None:
        params += list(predictor.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    updates = args.total_steps // (args.num_envs * args.horizon)

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    obs = obs0
    wandb_run = init_wandb(args)

    try:
        for upd in range(updates):
            batch, obs, prev_action, traces = rollout(
                envs=envs,
                ac=ac,
                f_mem=f_mem,
                drift=drift,
                predictor=predictor,
                device=device,
                horizon=args.horizon,
                gamma=args.gamma,
                lambda_pred=args.lambda_pred,
                alpha_base=alpha_base,
                alpha_max=alpha_max,
                reset_strategy=args.reset_strategy,
                reset_long_fraction=args.reset_long_fraction,
                obs=obs,
                prev_action=prev_action,
                traces=traces,
            )

            ppo_stats = ppo_update(
                ac=ac,
                opt=opt,
                batch=batch,
                clip_coef=args.clip_coef,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                epochs=args.epochs,
                minibatch_size=args.minibatch_size,
                lam=args.gae_lam,
                gamma=args.gamma,
                pred=predictor,
                pred_coef=args.pred_coef,
                generator=rng,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                grad_scaler=grad_scaler,
            )

            ema_update_(f_mem, ac.f_pol, tau=args.ema_tau)

            metrics = {
                "loop/update": upd + 1,
                "loop/frames": (upd + 1) * args.num_envs * args.horizon,
                "diagnostics/gate_mean": float(drift.state.gate.mean().item()),
            }
            metrics.update({f"loss/{k}": v for k, v in ppo_stats.items()})

            mean_ret, mean_len = recent_return_stats(envs.episode_returns, window=50)
            if mean_ret is not None:
                metrics["train/ret50"] = mean_ret
                metrics["train/len50"] = mean_len

            if (upd + 1) % args.log_interval == 0 and mean_ret is not None:
                print(
                    f"update={upd+1:04d}  episodes={len(envs.episode_returns):06d}  "
                    f"ret50={mean_ret:8.2f}  len50={mean_len:6.1f}  gate_mean={metrics['diagnostics/gate_mean']:6.3f}  "
                    f"kl={ppo_stats['approx_kl']:7.4f}  clipfrac={ppo_stats['clipfrac']:6.3f}"
                )

            if wandb_run is not None:
                wandb_run.log(metrics, step=metrics["loop/frames"])
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
