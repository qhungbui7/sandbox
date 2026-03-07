#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import LambdaLR

import gymnasium as gym
import wandb

from src.algorithms import (
    ALL_ALGOS,
    ON_POLICY_ALGOS,
    DQNReplayBuffer,
    dqn_collect_rollout,
    dqn_update,
    hard_update_,
    linear_schedule,
    normalize_algo_name,
    update_on_policy,
)
from src.amt import DriftMonitor, ema_update_, encode_mem, maybe_reset_traces, rollout, rollout_recurrent, trace_update
from src.envs import (
    CarRacingPreprocessWrapper,
    DiscreteCarRacingWrapper,
    EnvPool,
    FrameStackLastAxisWrapper,
    PartialObsWrapper,
    PiecewiseDriftWrapper,
)
from src.models import ActorCritic, FeatureEncoder, Predictor, RecurrentActorCritic
from src.ppo import ppo_update, ppo_update_recurrent
from src.reporting import start_run_report
from src.utils import load_env_file, obs_to_tensor, resolve_device, set_seed

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


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


def sanitize_postfix(value: str | None) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    out = []
    prev_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
            continue
        if not prev_sep:
            out.append("-")
            prev_sep = True
    return "".join(out).strip("-")


def with_postfix(base: str | None, postfix: str) -> str | None:
    if not postfix:
        return base
    if base is None or not str(base).strip():
        return postfix
    base_text = str(base).strip()
    if base_text.endswith(f"_{postfix}"):
        return base_text
    return f"{base_text}_{postfix}"


CONFIG_SECTION_KEYS = ("env", "model", "training", "other")
CONFIG_META_KEYS = {"include", "includes", "inherit", "inherits", "config_paths", "configs", "overrides"}


def _resolve_config_ref(raw_ref, *, base_dir: Path) -> Path:
    if not isinstance(raw_ref, str) or not raw_ref.strip():
        raise ValueError(f"Config reference must be a non-empty string, got: {raw_ref!r}")
    ref = Path(raw_ref.strip())
    if not ref.is_absolute():
        ref = base_dir / ref
    return ref.resolve()


def _flatten_config_sections(mapping: dict, *, source: Path) -> dict:
    flat: dict = {}
    for key, value in mapping.items():
        if key in CONFIG_META_KEYS:
            continue
        if key in CONFIG_SECTION_KEYS:
            if not isinstance(value, dict):
                raise ValueError(f"Config section `{key}` in {source} must be a mapping.")
            flat.update(value)
        else:
            flat[key] = value
    return flat


def load_config_file(config_path: Path, _stack: tuple[Path, ...] = ()) -> tuple[dict, dict]:
    resolved_path = config_path.resolve()
    if resolved_path in _stack:
        chain = " -> ".join(str(p) for p in (*_stack, resolved_path))
        raise ValueError(f"Cyclic config reference detected: {chain}")
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")
    raw_cfg = yaml.safe_load(resolved_path.read_text()) or {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Config file must map keys to values: {resolved_path}")

    merged_cfg: dict = {}
    ancestry = (*_stack, resolved_path)

    include_items: list = []
    for key in ("includes", "include", "inherits", "inherit"):
        if key not in raw_cfg:
            continue
        raw_items = raw_cfg[key]
        if raw_items is None:
            continue
        if isinstance(raw_items, str):
            include_items.append(raw_items)
            continue
        if not isinstance(raw_items, (list, tuple)):
            raise ValueError(f"`{key}` in {resolved_path} must be a string or list of strings.")
        include_items.extend(raw_items)
    for item in include_items:
        include_path = _resolve_config_ref(item, base_dir=resolved_path.parent)
        include_cfg, _ = load_config_file(include_path, ancestry)
        merged_cfg.update(include_cfg)

    config_paths = raw_cfg.get("config_paths", None)
    if config_paths is None:
        config_paths = raw_cfg.get("configs", None)
    if config_paths is not None:
        if not isinstance(config_paths, dict):
            raise ValueError(f"`config_paths` in {resolved_path} must be a mapping.")
        ordered_keys = [k for k in CONFIG_SECTION_KEYS if k in config_paths]
        ordered_keys.extend([k for k in config_paths.keys() if k not in ordered_keys])
        for key in ordered_keys:
            ref = config_paths[key]
            if ref is None:
                continue
            include_path = _resolve_config_ref(ref, base_dir=resolved_path.parent)
            include_cfg, _ = load_config_file(include_path, ancestry)
            merged_cfg.update(include_cfg)

    merged_cfg.update(_flatten_config_sections(raw_cfg, source=resolved_path))

    overrides = raw_cfg.get("overrides", None)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError(f"`overrides` in {resolved_path} must be a mapping.")
        merged_cfg.update(_flatten_config_sections(overrides, source=resolved_path))

    return merged_cfg, raw_cfg


def collect_cli_provided_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    provided: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            provided.add(dest)
    return provided


AMT_ONLY_KEYS = {
    "alpha_base",
    "alpha_max",
    "lambda_pred",
    "pred_coef",
    "reset_strategy",
    "reset_long_fraction",
    "rho_s",
    "rho_l",
    "beta",
    "tau_soft",
    "kappa",
    "tau_on",
    "tau_off",
    "K",
    "cooldown_steps",
    "warmup_steps",
}
CARRACING_ONLY_KEYS = {"carracing_downsample", "carracing_grayscale"}
DRIFT_ONLY_KEYS = {"rho_s", "rho_l", "beta", "tau_soft", "kappa", "tau_on", "tau_off", "K", "cooldown_steps", "warmup_steps"}
PPO_ONLY_KEYS = {"clip_coef", "vf_clip", "target_kl"}
TRPO_ONLY_KEYS = {"trpo_max_kl", "trpo_backtrack_coef", "trpo_backtrack_iters", "trpo_value_epochs"}
VTRACE_ONLY_KEYS = {"vtrace_rho_clip", "vtrace_c_clip"}
VMPO_ONLY_KEYS = {"vmpo_topk_frac", "vmpo_eta", "vmpo_kl_coef", "vmpo_kl_target"}
DQN_ONLY_KEYS = {
    "dqn_replay_size",
    "dqn_batch_size",
    "dqn_learning_starts",
    "dqn_updates_per_iter",
    "dqn_target_update_interval",
    "dqn_double",
    "dqn_eps_start",
    "dqn_eps_end",
    "dqn_eps_decay_steps",
}


def validate_no_unknown_config_keys(*, resolved_cfg: dict, parser: argparse.ArgumentParser) -> None:
    known = {action.dest for action in parser._actions if action.dest and action.dest != argparse.SUPPRESS}
    unknown = sorted(k for k in resolved_cfg.keys() if k not in known)
    if unknown:
        raise ValueError(
            "Unknown config keys detected (likely typo or unsupported parameter): " + ", ".join(unknown)
        )


def _normalized_key_value(key: str, value):
    if key == "algo":
        return normalize_algo_name(str(value))
    if key in {"policy", "env_id"}:
        return str(value).strip()
    return value


def validate_no_strange_params(args, *, resolved_cfg: dict, cli_dests: set[str]) -> None:
    explicit_cfg_keys = {k for k, v in resolved_cfg.items() if v is not None}
    explicit_keys = explicit_cfg_keys | set(cli_dests)

    # Prevent silent experiment mutation when overriding pinned config identity.
    conflicts: list[str] = []
    for key in ("algo", "policy", "env_id"):
        if key not in cli_dests:
            continue
        if key not in resolved_cfg or resolved_cfg.get(key) is None:
            continue
        cfg_value = _normalized_key_value(key, resolved_cfg.get(key))
        arg_value = _normalized_key_value(key, getattr(args, key))
        if cfg_value != arg_value:
            conflicts.append(f"{key} (config={resolved_cfg.get(key)!r}, cli={getattr(args, key)!r})")
    if conflicts:
        raise ValueError(
            "CLI overrides conflict with config-pinned identity keys: "
            + ", ".join(conflicts)
            + ". Use a matching config instead of overriding these keys."
        )

    policy = str(args.policy).strip().lower()
    env_id = str(args.env_id).strip()
    algo = normalize_algo_name(str(args.algo))

    if not env_id.startswith("CarRacing"):
        bad_env_keys = sorted(explicit_keys & CARRACING_ONLY_KEYS)
        if bad_env_keys:
            raise ValueError(
                f"CarRacing-only parameters provided for env `{env_id}`: {', '.join(bad_env_keys)}"
            )

    if policy != "amt":
        bad_amt_keys = sorted(explicit_keys & AMT_ONLY_KEYS)
        if bad_amt_keys:
            raise ValueError(
                f"AMT-only parameters provided for policy `{policy}`: {', '.join(bad_amt_keys)}"
            )
    else:
        reset_strategy = str(args.reset_strategy).strip().lower()
        if reset_strategy == "none":
            bad_reset_keys = sorted(explicit_keys & {"reset_long_fraction"})
            if bad_reset_keys:
                raise ValueError(
                    "Parameters not used with `reset_strategy=none`: " + ", ".join(bad_reset_keys)
                )
        skip_drift = (reset_strategy == "none") and _fixed_alpha_config(args.alpha_base, args.alpha_max)
        if skip_drift:
            bad_drift_keys = sorted(explicit_keys & (DRIFT_ONLY_KEYS | {"lambda_pred", "pred_coef"}))
            if bad_drift_keys:
                raise ValueError(
                    "Drift/prediction parameters are not used when alpha is fixed and reset strategy is `none`: "
                    + ", ".join(bad_drift_keys)
                )

    # Algorithm-specific strictness for explicit CLI args only.
    algo_only = {
        "ppo": PPO_ONLY_KEYS,
        "trpo": TRPO_ONLY_KEYS,
        "v-trace": VTRACE_ONLY_KEYS,
        "v-mpo": VMPO_ONLY_KEYS,
        "dqn": DQN_ONLY_KEYS,
    }
    for owner_algo, keys in algo_only.items():
        if algo == owner_algo:
            continue
        bad_cli = sorted(set(cli_dests) & keys)
        if bad_cli:
            raise ValueError(
                f"Parameters not used by `algo={algo}` (owned by `{owner_algo}`): {', '.join(bad_cli)}"
            )

    if policy == "recurrent":
        bad_recurrent_cli = sorted(set(cli_dests) & {"ema_tau"})
        if bad_recurrent_cli:
            raise ValueError(
                "Parameters not used by recurrent policy: " + ", ".join(bad_recurrent_cli)
            )

    lr_schedule = str(getattr(args, "lr_schedule", "")).strip().lower()
    lr_end = getattr(args, "lr_end", None)
    if lr_schedule == "none":
        if (lr_end is not None) and ("lr_end" in explicit_keys):
            raise ValueError("`lr_end` must be null/omitted when `lr_schedule=none`.")
    elif lr_schedule == "linear":
        if lr_end is not None and float(lr_end) < 0.0:
            raise ValueError("`lr_end` must be >= 0 when `lr_schedule=linear`.")
        if lr_end is not None and float(lr_end) > float(args.lr):
            raise ValueError("`lr_end` must be <= `lr` when `lr_schedule=linear`.")

    # Keep A2C/REINFORCE on-policy in this trainer by enforcing one full-batch update per rollout.
    if algo in {"a2c", "reinforce"}:
        frames_per_update = int(args.num_envs) * int(args.horizon)
        if int(args.epochs) != 1:
            raise ValueError(
                f"`algo={algo}` requires `epochs=1` for on-policy updates "
                "(no repeated passes over the same rollout)."
            )
        if int(args.minibatch_size) < frames_per_update:
            raise ValueError(
                f"`algo={algo}` requires `minibatch_size >= num_envs*horizon` "
                f"(got {args.minibatch_size}, need at least {frames_per_update}) so each rollout is used in one batch."
            )

def _fixed_alpha_config(alpha_base_raw, alpha_max_raw) -> bool:
    alpha_base = parse_floats(alpha_base_raw)
    alpha_max = parse_floats(alpha_max_raw)
    if (len(alpha_base) == 0) or (len(alpha_base) != len(alpha_max)):
        return False
    return all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(alpha_base, alpha_max, strict=True))


def build_required_explicit_keys(args) -> set[str]:
    required = {
        "env_id",
        "num_envs",
        "env_workers",
        "horizon",
        "total_steps",
        "seed",
        "device",
        "algo",
        "policy",
        "hidden_dim",
        "feat_dim",
        "act_embed_dim",
        "mask_indices",
        "phase_len",
        "obs_shift_scale",
        "reward_scale_low",
        "reward_scale_high",
        "gamma",
        "lr",
        "max_grad_norm",
        "vf_coef",
        "ent_coef",
        "epochs",
        "log_interval",
        "wandb",
        "report",
        "report_dir",
    }
    if str(getattr(args, "env_id", "")).startswith("CarRacing"):
        required.update({"carracing_downsample", "carracing_grayscale"})

    algo_name = normalize_algo_name(str(args.algo))
    if algo_name in ON_POLICY_ALGOS or str(args.policy) == "recurrent":
        required.add("gae_lam")
    if algo_name in {"ppo", "a2c", "reinforce", "v-mpo"} or str(args.policy) == "recurrent":
        required.add("minibatch_size")
    if algo_name == "ppo" or str(args.policy) == "recurrent":
        required.add("clip_coef")
        required.add("vf_clip")
        required.add("target_kl")
    if (algo_name in ON_POLICY_ALGOS) and (str(args.policy) != "recurrent"):
        required.add("ema_tau")

    if str(args.policy) == "amt":
        required.update({"alpha_base", "alpha_max", "reset_strategy"})
        if str(args.reset_strategy) != "none":
            required.add("reset_long_fraction")
        skip_drift = (str(args.reset_strategy) == "none") and _fixed_alpha_config(args.alpha_base, args.alpha_max)
        if not skip_drift:
            required.update({"lambda_pred", "pred_coef"})
            required.update({"rho_s", "rho_l", "beta", "tau_soft", "kappa", "tau_on", "tau_off", "K", "cooldown_steps", "warmup_steps"})

    if algo_name == "trpo":
        required.update({"trpo_max_kl", "trpo_backtrack_coef", "trpo_backtrack_iters", "trpo_value_epochs"})
    elif algo_name == "v-trace":
        required.update({"vtrace_rho_clip", "vtrace_c_clip"})
    elif algo_name == "v-mpo":
        required.update({"vmpo_topk_frac", "vmpo_eta", "vmpo_kl_coef", "vmpo_kl_target"})
    elif algo_name == "dqn":
        required.update(
            {
                "dqn_replay_size",
                "dqn_batch_size",
                "dqn_learning_starts",
                "dqn_updates_per_iter",
                "dqn_target_update_interval",
                "dqn_double",
                "dqn_eps_start",
                "dqn_eps_end",
                "dqn_eps_decay_steps",
            }
        )
    return required


def validate_explicit_required_keys(args, *, resolved_cfg: dict, cli_dests: set[str]) -> None:
    required_keys = build_required_explicit_keys(args)
    missing = []
    for key in sorted(required_keys):
        if key in cli_dests:
            continue
        if key not in resolved_cfg:
            missing.append(key)
            continue
        if resolved_cfg.get(key) is None:
            missing.append(key)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Missing required explicit parameters. "
            f"Define these in YAML config sections or pass via CLI: {missing_text}"
        )


def init_wandb(args):
    if not args.wandb:
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed. Run `pip install wandb` or disable --wandb.")
    settings = None
    target_gpu = None
    if str(args.device).startswith("cuda"):
        if args.cuda_id is not None:
            target_gpu = int(args.cuda_id)
        else:
            device_text = str(args.device)
            if ":" in device_text:
                try:
                    target_gpu = int(device_text.split(":", 1)[1])
                except ValueError:
                    target_gpu = None
    if target_gpu is not None:
        settings = wandb.Settings(
            x_stats_gpu_device_ids=[target_gpu],
            x_stats_gpu_count=1,
        )
    tags = parse_strs(args.wandb_tags)
    config_payload = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    source_config = getattr(args, "_source_config", None)
    source_config_path = getattr(args, "_source_config_path", None)
    if isinstance(source_config, dict):
        config_payload["source_config"] = source_config
    if source_config_path:
        config_payload["source_config_path"] = str(source_config_path)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=tags or None,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config=config_payload,
        settings=settings,
    )
    if run is not None and source_config_path:
        cfg_path = Path(str(source_config_path))
        if cfg_path.exists():
            try:
                run.save(str(cfg_path.resolve()), policy="now")
            except Exception as exc:  # pragma: no cover - best effort file attachment
                print(f"Warning: unable to upload config file to W&B ({cfg_path}): {exc}")
    return run


class EarlyStopper:
    def __init__(self, *, metric: str, mode: str, patience: int, min_delta: float, warmup_updates: int):
        self.metric = (metric or "").strip()
        self.mode = mode
        self.patience = int(max(patience, 0))
        self.min_delta = float(max(min_delta, 0.0))
        self.warmup_updates = int(max(warmup_updates, 0))
        self.best: float | None = None
        self.best_update: int | None = None
        self.bad_updates = 0
        self.warned_missing_metric = False

    @property
    def enabled(self) -> bool:
        return bool(self.metric) and (self.patience > 0)

    def _is_improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < (self.best - self.min_delta)
        return value > (self.best + self.min_delta)

    def update(self, *, update_idx: int, metrics: dict) -> tuple[bool, str | None]:
        if not self.enabled:
            return False, None
        if update_idx <= self.warmup_updates:
            return False, None
        value_raw = metrics.get(self.metric)
        if not isinstance(value_raw, (int, float)) or (not math.isfinite(float(value_raw))):
            if not self.warned_missing_metric:
                self.warned_missing_metric = True
                return False, f"Early stop metric `{self.metric}` missing/non-finite; skipping early-stop checks."
            return False, None

        value = float(value_raw)
        if self._is_improved(value):
            self.best = value
            self.best_update = int(update_idx)
            self.bad_updates = 0
        else:
            self.bad_updates += 1

        metrics["early_stop/enabled"] = 1.0
        metrics["early_stop/bad_updates"] = float(self.bad_updates)
        if self.best is not None:
            metrics["early_stop/best"] = float(self.best)
        if self.best_update is not None:
            metrics["early_stop/best_update"] = float(self.best_update)

        if self.bad_updates >= self.patience:
            reason = (
                f"Early stop triggered at update={update_idx}: metric={self.metric}, mode={self.mode}, "
                f"best={self.best}, bad_updates={self.bad_updates}, patience={self.patience}."
            )
            return True, reason
        return False, None


def recent_return_stats(episode_returns: list[tuple[float, int]], window: int = 50) -> tuple[float | None, float | None]:
    if not episode_returns:
        return None, None
    recent = episode_returns[-window:]
    mean_ret = sum(r for r, _ in recent) / len(recent)
    mean_len = sum(l for _, l in recent) / len(recent)
    return mean_ret, mean_len


def _shape_metrics(prefix: str, shape: tuple[int, ...], max_dims: int = 4) -> dict[str, float]:
    metrics = {f"{prefix}/rank": float(len(shape))}
    for i in range(max_dims):
        metrics[f"{prefix}/d{i}"] = float(shape[i]) if i < len(shape) else -1.0
    return metrics


def build_ppo_rollout_debug_metrics(
    *,
    args,
    batch: dict,
    update_idx: int,
    updates_total: int,
    total_steps_target: int,
    ended_episodes: list[tuple[float, int]],
) -> dict[str, float]:
    rollout_batch_size = int(args.num_envs * args.horizon)
    num_minibatches = int(args.epochs * math.ceil(rollout_batch_size / max(int(args.minibatch_size), 1)))

    obs_shape = tuple(int(x) for x in batch["obs"].shape)
    actions_shape = tuple(int(x) for x in batch["actions"].shape)
    rewards_shape = tuple(int(x) for x in batch["rewards"].shape)
    dones_shape = tuple(int(x) for x in batch["dones"].shape)

    terminated_count = int(batch["terminated"].sum().item())
    truncated_count = int(batch["truncated"].sum().item())

    if ended_episodes:
        returns = np.asarray([r for r, _ in ended_episodes], dtype=np.float64)
        lengths = np.asarray([l for _, l in ended_episodes], dtype=np.float64)
        ret_mean = float(returns.mean())
        ret_std = float(returns.std())
        ret_min = float(returns.min())
        ret_max = float(returns.max())
        len_mean = float(lengths.mean())
        len_std = float(lengths.std())
        len_min = float(lengths.min())
        len_max = float(lengths.max())
    else:
        ret_mean = float("nan")
        ret_std = float("nan")
        ret_min = float("nan")
        ret_max = float("nan")
        len_mean = float("nan")
        len_std = float("nan")
        len_min = float("nan")
        len_max = float("nan")

    frames_done = int(update_idx * rollout_batch_size)
    metrics = {
        "debug/bookkeeping/rollout_batch_size": float(rollout_batch_size),
        "debug/bookkeeping/minibatch_size": float(args.minibatch_size),
        "debug/bookkeeping/num_minibatches": float(num_minibatches),
        "debug/bookkeeping/sampling_with_replacement": 0.0,
        "debug/bookkeeping/steps_collected": float(rollout_batch_size),
        "debug/bookkeeping/updates_done": float(update_idx),
        "debug/bookkeeping/updates_total": float(updates_total),
        "debug/bookkeeping/frames_done": float(frames_done),
        "debug/bookkeeping/total_steps_target": float(total_steps_target),
        "debug/episode/terminated_count": float(terminated_count),
        "debug/episode/truncated_count": float(truncated_count),
        "debug/episode/ended_count": float(len(ended_episodes)),
        "debug/episode/return_mean": ret_mean,
        "debug/episode/return_std": ret_std,
        "debug/episode/return_min": ret_min,
        "debug/episode/return_max": ret_max,
        "debug/episode/length_mean": len_mean,
        "debug/episode/length_std": len_std,
        "debug/episode/length_min": len_min,
        "debug/episode/length_max": len_max,
    }
    metrics.update(_shape_metrics("debug/shape/obs", obs_shape))
    metrics.update(_shape_metrics("debug/shape/actions", actions_shape))
    metrics.update(_shape_metrics("debug/shape/rewards", rewards_shape))
    metrics.update(_shape_metrics("debug/shape/dones", dones_shape))
    return metrics


def infer_regime(mask_indices: list[int], phase_len: int, obs_shift_scale: float, reward_scale_low: float, reward_scale_high: float) -> dict:
    partial_observable = len(mask_indices) > 0
    non_stationary = (phase_len > 0) and (
        (obs_shift_scale > 0.0) or (reward_scale_low != 1.0) or (reward_scale_high != 1.0)
    )
    if (not non_stationary) and (not partial_observable):
        name = "stationary_fullobs"
    elif (not non_stationary) and partial_observable:
        name = "stationary_partialobs"
    elif non_stationary and (not partial_observable):
        name = "nonstationary_fullobs"
    else:
        name = "nonstationary_partialobs"
    return {
        "name": name,
        "partial_observable": partial_observable,
        "non_stationary": non_stationary,
        "mask_indices": mask_indices,
        "phase_len": int(phase_len),
        "obs_shift_scale": float(obs_shift_scale),
        "reward_scale_low": float(reward_scale_low),
        "reward_scale_high": float(reward_scale_high),
    }


def add_perf_metrics(metrics: dict, train_start: float, updates_done: int, updates_total: int) -> None:
    elapsed = max(time.perf_counter() - train_start, 1e-6)
    frames = float(metrics.get("loop/frames", 0.0))
    updates_per_sec = updates_done / elapsed
    eta_sec = max(updates_total - updates_done, 0) / max(updates_per_sec, 1e-8)
    metrics["perf/elapsed_sec"] = elapsed
    metrics["perf/fps"] = frames / elapsed
    metrics["perf/updates_per_sec"] = updates_per_sec
    metrics["perf/eta_sec"] = eta_sec


def finite_metric(value) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value_f = float(value)
    if not math.isfinite(value_f):
        return None
    return value_f


def numpy_array_stats(array: np.ndarray) -> dict:
    data = np.asarray(array)
    stats = {"shape": list(data.shape), "dtype": str(data.dtype)}
    if data.size == 0:
        return stats
    stats.update(
        {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
        }
    )
    return stats


def module_param_stats(module: torch.nn.Module) -> dict:
    params = list(module.parameters())
    total_params = int(sum(p.numel() for p in params))
    trainable_params = int(sum(p.numel() for p in params if p.requires_grad))
    total_bytes = int(sum(p.numel() * p.element_size() for p in params))
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in params})
    devices = sorted({str(p.device) for p in params})
    return {
        "class": type(module).__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_bytes": total_bytes,
        "param_mb": round(total_bytes / (1024**2), 3),
        "dtypes": dtypes,
        "devices": devices,
    }


def optimizer_param_stats(optimizer: torch.optim.Optimizer) -> dict:
    unique_params: dict[int, torch.Tensor] = {}
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            unique_params[id(param)] = param
    total_params = int(sum(p.numel() for p in unique_params.values()))
    total_bytes = int(sum(p.numel() * p.element_size() for p in unique_params.values()))
    return {
        "param_tensors": len(unique_params),
        "total_params": total_params,
        "param_bytes": total_bytes,
        "param_mb": round(total_bytes / (1024**2), 3),
    }


def env_wrapper_chain(env: gym.Env, max_depth: int = 32) -> list[str]:
    chain = []
    current_env = env
    seen = set()
    for _ in range(max_depth):
        chain.append(type(current_env).__name__)
        env_id = id(current_env)
        if env_id in seen:
            chain.append("<cycle>")
            break
        seen.add(env_id)
        inner = getattr(current_env, "env", None)
        if inner is None:
            break
        current_env = inner
    return chain


def find_wrapper(env: gym.Env, wrapper_type: type, max_depth: int = 32):
    current_env = env
    seen = set()
    for _ in range(max_depth):
        if isinstance(current_env, wrapper_type):
            return current_env
        env_id = id(current_env)
        if env_id in seen:
            return None
        seen.add(env_id)
        inner = getattr(current_env, "env", None)
        if inner is None:
            return None
        current_env = inner
    return None


def make_adam_optimizer(
    params,
    lr: float,
    *,
    device_type: str,
    adam_foreach: bool | None,
    adam_fused: bool | None,
) -> torch.optim.Adam:
    kwargs: dict = {}
    if device_type == "cuda":
        fused = bool(adam_fused) if adam_fused is not None else True
        if fused:
            if adam_foreach:
                raise ValueError("Cannot enable both --adam-fused and --adam-foreach.")
            kwargs["fused"] = True
        else:
            kwargs["fused"] = False
            foreach = bool(adam_foreach) if adam_foreach is not None else True
            kwargs["foreach"] = foreach
    else:
        if adam_fused:
            raise ValueError("--adam-fused is only supported on CUDA.")
        foreach = bool(adam_foreach) if adam_foreach is not None else True
        kwargs["foreach"] = foreach
    try:
        return torch.optim.Adam(params, lr=lr, **kwargs)
    except Exception as exc:  # pragma: no cover - runtime fallback
        print(f"Warning: failed to create Adam optimizer with {kwargs} ({exc}); falling back to default Adam.")
        return torch.optim.Adam(params, lr=lr)


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_updates: int,
    schedule: str,
    start_lr: float,
    end_lr: float | None,
):
    schedule_name = str(schedule).strip().lower()
    if schedule_name in {"", "none"}:
        return None
    if schedule_name != "linear":
        raise ValueError(f"Unsupported --lr-schedule `{schedule}`. Expected one of: none, linear.")
    start = float(start_lr)
    end = 0.0 if end_lr is None else float(end_lr)
    if start < 0.0:
        raise ValueError("`lr` must be >= 0.")
    if end < 0.0:
        raise ValueError("`lr_end` must be >= 0.")
    if end > start:
        raise ValueError("`lr_end` must be <= `lr` for linear decay.")
    if start == 0.0:
        return LambdaLR(optimizer, lr_lambda=lambda _step_idx: 1.0)
    denom = max(int(total_updates), 1)
    end_ratio = end / start
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step_idx: (
            1.0 + (end_ratio - 1.0) * min(1.0, float(step_idx + 1) / float(denom))
        ),
    )


def maybe_compile_module(module: torch.nn.Module, enabled: bool, mode: str) -> torch.nn.Module:
    if not enabled:
        return module
    try:
        return torch.compile(module, mode=mode)
    except Exception as exc:  # pragma: no cover - optional acceleration
        print(f"Warning: torch.compile failed ({exc}); continuing without compilation.")
        return module


class TorchProfilerController:
    def __init__(self, *, args, reporter, device: str, phase_label: str):
        self.enabled = bool(getattr(args, "torch_profiler", False))
        self._reporter = reporter
        self._profiler = None
        self._summary_path: Path | None = None
        self._sort_by = str(getattr(args, "torch_profiler_sort_by", "") or "").strip()
        self._row_limit = int(getattr(args, "torch_profiler_row_limit", 40))
        if not self.enabled:
            return

        default_sort = "self_cuda_time_total" if torch.device(device).type == "cuda" else "self_cpu_time_total"
        if not self._sort_by:
            self._sort_by = default_sort

        raw_dir = str(getattr(args, "torch_profiler_dir", "") or "").strip()
        if raw_dir:
            output_dir = Path(raw_dir)
        elif reporter.enabled:
            output_dir = reporter.run_dir / "profiler" / phase_label
        else:
            output_dir = Path("reports") / "profiler" / phase_label
        output_dir.mkdir(parents=True, exist_ok=True)
        self._summary_path = output_dir / "key_averages.txt"

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.device(device).type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        schedule = torch.profiler.schedule(
            wait=int(getattr(args, "torch_profiler_wait", 1)),
            warmup=int(getattr(args, "torch_profiler_warmup", 1)),
            active=int(getattr(args, "torch_profiler_active", 3)),
            repeat=int(getattr(args, "torch_profiler_repeat", 1)),
            skip_first=int(getattr(args, "torch_profiler_skip_first", 0)),
        )
        trace_handler = torch.profiler.tensorboard_trace_handler(str(output_dir), worker_name=phase_label)
        profiler_cfg = {
            "enabled": True,
            "phase": phase_label,
            "output_dir": str(output_dir),
            "activities": [str(a).replace("ProfilerActivity.", "") for a in activities],
            "wait": int(getattr(args, "torch_profiler_wait", 1)),
            "warmup": int(getattr(args, "torch_profiler_warmup", 1)),
            "active": int(getattr(args, "torch_profiler_active", 3)),
            "repeat": int(getattr(args, "torch_profiler_repeat", 1)),
            "skip_first": int(getattr(args, "torch_profiler_skip_first", 0)),
            "record_shapes": bool(getattr(args, "torch_profiler_record_shapes", True)),
            "profile_memory": bool(getattr(args, "torch_profiler_profile_memory", True)),
            "with_stack": bool(getattr(args, "torch_profiler_with_stack", False)),
            "with_flops": bool(getattr(args, "torch_profiler_with_flops", False)),
            "sort_by": self._sort_by,
            "row_limit": self._row_limit,
        }
        if reporter.enabled:
            reporter.log_block("torch_profiler", profiler_cfg)
            reporter.update_summary({"torch_profiler": profiler_cfg})
        print(
            f"[torch-profiler] enabled phase={phase_label} output_dir={output_dir} "
            f"schedule=(wait={profiler_cfg['wait']}, warmup={profiler_cfg['warmup']}, "
            f"active={profiler_cfg['active']}, repeat={profiler_cfg['repeat']})"
        )
        try:
            self._profiler = torch.profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=trace_handler,
                record_shapes=bool(getattr(args, "torch_profiler_record_shapes", True)),
                profile_memory=bool(getattr(args, "torch_profiler_profile_memory", True)),
                with_stack=bool(getattr(args, "torch_profiler_with_stack", False)),
                with_flops=bool(getattr(args, "torch_profiler_with_flops", False)),
            )
            self._profiler.__enter__()
        except Exception as exc:
            self.enabled = False
            self._profiler = None
            print(f"Warning: failed to start torch profiler ({exc}); continuing without profiling.")

    def step(self) -> None:
        if self._profiler is None:
            return
        try:
            self._profiler.step()
        except Exception as exc:
            print(f"Warning: torch profiler step failed ({exc}); disabling profiler for remaining updates.")
            self.close()

    def close(self) -> None:
        if self._profiler is None:
            return
        try:
            self._profiler.__exit__(None, None, None)
        except Exception:
            pass
        try:
            table = self._profiler.key_averages().table(sort_by=self._sort_by, row_limit=self._row_limit)
        except Exception:
            try:
                table = self._profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=self._row_limit)
            except Exception as exc:
                table = f"Unable to render profiler key averages: {exc}"
        if self._summary_path is not None:
            self._summary_path.write_text(table + "\n", encoding="utf-8")
            print(f"[torch-profiler] wrote key averages to {self._summary_path}")
            if self._reporter.enabled:
                self._reporter.log_line(f"torch profiler key averages: {self._summary_path}")
                self._reporter.update_summary({"torch_profiler_key_averages": str(self._summary_path)})
        self._profiler = None


def _snapshot_rng_state() -> dict:
    state: dict = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "torch_cuda": None,
    }
    try:
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        state["torch_cuda"] = None
    return state


def _restore_rng_state(state: dict) -> None:
    try:
        torch.set_rng_state(state["torch_cpu"])
    except Exception:
        pass
    try:
        np.random.set_state(state["numpy"])
    except Exception:
        pass
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        try:
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception:
            pass


@torch.no_grad()
def evaluate_trace_policy(
    *,
    args,
    mask_indices: list[int],
    ac: ActorCritic,
    f_mem: FeatureEncoder,
    predictor: Predictor | None,
    alpha_base: torch.Tensor,
    alpha_max: torch.Tensor,
    device: str,
    eval_seed: int,
    eval_num_envs: int,
    eval_episodes: int,
    deterministic: bool,
) -> dict:
    rng_state = _snapshot_rng_state()
    start_time = time.perf_counter()

    ac_was_training = ac.training
    f_mem_was_training = f_mem.training
    predictor_was_training = predictor.training if predictor is not None else None
    ac.eval()
    f_mem.eval()
    if predictor is not None:
        predictor.eval()

    eval_envs = None
    try:
        env_fns = [
            make_env_fn(
                env_id=args.env_id,
                seed=eval_seed + 10_000 * i,
                mask_indices=mask_indices,
                phase_len=args.phase_len,
                obs_shift_scale=args.obs_shift_scale,
                reward_scale_low=args.reward_scale_low,
                reward_scale_high=args.reward_scale_high,
                carracing_downsample=int(getattr(args, "carracing_downsample", 1)),
                carracing_grayscale=bool(getattr(args, "carracing_grayscale", False)),
                frame_stack=int(getattr(args, "frame_stack", 1)),
            )
            for i in range(eval_num_envs)
        ]
        eval_envs = EnvPool(env_fns)
        obs, _ = eval_envs.reset(seed=eval_seed)
        n_envs = eval_envs.num_envs

        skip_drift = (args.reset_strategy == "none") and torch.allclose(alpha_base, alpha_max)
        alpha_const = alpha_base.expand(n_envs, -1).clamp(0.0, 1.0) if skip_drift else None
        drift = None
        if not skip_drift:
            drift = DriftMonitor(
                num_envs=n_envs,
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

        prev_action = torch.zeros(n_envs, device=device, dtype=torch.int64)
        obs0_t = obs_to_tensor(obs, device=device, obs_normalization=args.obs_normalization)
        x_mem0 = encode_mem(f_mem, obs0_t, prev_action)

        M = int(alpha_base.shape[1])
        feat_dim = int(x_mem0.shape[-1])
        traces = torch.zeros((n_envs, M, feat_dim), device=device)
        traces = trace_update(traces, x_mem0, alpha_base.expand(n_envs, -1))

        long_start = int(math.floor((1.0 - args.reset_long_fraction) * M))
        long_mask = torch.zeros(M, device=device, dtype=torch.bool)
        if args.reset_strategy == "partial":
            long_mask[long_start:] = True

        step_calls = 0
        max_steps = int(max(eval_episodes, 1) * 10_000)
        while (len(eval_envs.episode_returns) < eval_episodes) and (step_calls < max_steps):
            obs_t = obs_to_tensor(obs, device=device, obs_normalization=args.obs_normalization)
            traces_flat = traces.reshape(n_envs, -1)

            out, value = ac(obs_t, prev_action, traces_flat)
            if args.algo == "dqn" or deterministic:
                action = out.argmax(dim=-1)
            else:
                action = Categorical(logits=out).sample()

            next_obs, reward, terminated, truncated, _ = eval_envs.step(action.cpu().numpy())
            done_env = terminated | truncated

            rew = torch.as_tensor(reward, device=device, dtype=torch.float32)
            done = torch.as_tensor(done_env, device=device, dtype=torch.bool)

            x_mem_t = encode_mem(f_mem, obs_t, prev_action)
            next_obs_t = obs_to_tensor(next_obs, device=device, obs_normalization=args.obs_normalization)
            next_prev_action = action.clone()
            if done.any():
                next_prev_action[done] = 0
            x_mem_next = encode_mem(f_mem, next_obs_t, next_prev_action)

            if skip_drift:
                traces_next = trace_update(traces, x_mem_next, alpha_const)
            else:
                assert drift is not None
                _, v_next_prov = ac(
                    next_obs_t,
                    next_prev_action,
                    trace_update(traces, x_mem_next, alpha_base.expand(n_envs, -1)).reshape(n_envs, -1),
                )

                delta_prov = rew + args.gamma * (~done).float() * v_next_prov - value
                pred_err = torch.zeros(n_envs, device=device)
                if (predictor is not None) and (args.lambda_pred > 0.0):
                    x_hat = predictor(x_mem_t, action)
                    pred_err = (x_mem_next - x_hat).pow(2).mean(dim=-1)

                e = delta_prov.abs() + args.lambda_pred * pred_err
                gate, reset_event = drift.update(e)
                reset_event = reset_event & (~done)

                alpha = alpha_base + gate.unsqueeze(-1) * (alpha_max - alpha_base)
                alpha = alpha.clamp(0.0, 1.0)

                traces_reset = maybe_reset_traces(traces, reset_event, x_mem_next, args.reset_strategy, long_mask)
                traces_next = trace_update(traces_reset, x_mem_next, alpha)

            if done.any():
                if drift is not None:
                    drift.reset_where(done)
                traces_next[done] = 0.0
                traces_next[done] = trace_update(
                    traces_next[done],
                    x_mem_next[done],
                    alpha_base.expand(done.sum(), -1),
                )

            traces = traces_next
            prev_action = next_prev_action
            obs = next_obs
            step_calls += 1

        episodes_collected = min(len(eval_envs.episode_returns), eval_episodes)
        if episodes_collected > 0:
            episode_data = eval_envs.episode_returns[:episodes_collected]
            rets = np.asarray([r for r, _ in episode_data], dtype=np.float64)
            lens = np.asarray([l for _, l in episode_data], dtype=np.float64)
            ret_mean = float(rets.mean())
            ret_std = float(rets.std())
            len_mean = float(lens.mean())
            len_std = float(lens.std())
        else:
            ret_mean = None
            ret_std = None
            len_mean = None
            len_std = None

        runtime = max(time.perf_counter() - start_time, 1e-9)
        env_steps = int(step_calls * n_envs)
        return {
            "eval/seed": int(eval_seed),
            "eval/num_envs": int(n_envs),
            "eval/deterministic": bool(deterministic),
            "eval/episodes_target": int(eval_episodes),
            "eval/episodes_collected": int(episodes_collected),
            "eval/ret_mean": ret_mean,
            "eval/ret_std": ret_std,
            "eval/len_mean": len_mean,
            "eval/len_std": len_std,
            "eval/env_steps": env_steps,
            "eval/runtime_sec": runtime,
            "eval/fps": (env_steps / runtime) if env_steps > 0 else None,
        }
    finally:
        if eval_envs is not None:
            eval_envs.close()
        ac.train(ac_was_training)
        f_mem.train(f_mem_was_training)
        if predictor is not None and predictor_was_training is not None:
            predictor.train(predictor_was_training)
        _restore_rng_state(rng_state)


@torch.no_grad()
def evaluate_recurrent_policy(
    *,
    args,
    mask_indices: list[int],
    ac: RecurrentActorCritic,
    device: str,
    eval_seed: int,
    eval_num_envs: int,
    eval_episodes: int,
    deterministic: bool,
) -> dict:
    rng_state = _snapshot_rng_state()
    start_time = time.perf_counter()

    ac_was_training = ac.training
    ac.eval()

    eval_envs = None
    try:
        env_fns = [
            make_env_fn(
                env_id=args.env_id,
                seed=eval_seed + 10_000 * i,
                mask_indices=mask_indices,
                phase_len=args.phase_len,
                obs_shift_scale=args.obs_shift_scale,
                reward_scale_low=args.reward_scale_low,
                reward_scale_high=args.reward_scale_high,
                carracing_downsample=int(getattr(args, "carracing_downsample", 1)),
                carracing_grayscale=bool(getattr(args, "carracing_grayscale", False)),
                frame_stack=int(getattr(args, "frame_stack", 1)),
            )
            for i in range(eval_num_envs)
        ]
        eval_envs = EnvPool(env_fns)
        obs, _ = eval_envs.reset(seed=eval_seed)
        n_envs = eval_envs.num_envs

        prev_action = torch.zeros(n_envs, device=device, dtype=torch.int64)
        h, c = ac.init_hidden(n_envs, device)

        step_calls = 0
        max_steps = int(max(eval_episodes, 1) * 10_000)
        while (len(eval_envs.episode_returns) < eval_episodes) and (step_calls < max_steps):
            obs_t = obs_to_tensor(obs, device=device, obs_normalization=args.obs_normalization)
            logits, _value, (h, c) = ac(obs_t, prev_action, (h, c))
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = Categorical(logits=logits).sample()

            next_obs, reward, terminated, truncated, _ = eval_envs.step(action.cpu().numpy())
            done_env = terminated | truncated
            done = torch.as_tensor(done_env, device=device, dtype=torch.bool)
            next_prev_action = action.clone()
            if done.any():
                h[:, done] = 0.0
                c[:, done] = 0.0
                next_prev_action[done] = 0

            prev_action = next_prev_action
            obs = next_obs
            step_calls += 1

        episodes_collected = min(len(eval_envs.episode_returns), eval_episodes)
        if episodes_collected > 0:
            episode_data = eval_envs.episode_returns[:episodes_collected]
            rets = np.asarray([r for r, _ in episode_data], dtype=np.float64)
            lens = np.asarray([l for _, l in episode_data], dtype=np.float64)
            ret_mean = float(rets.mean())
            ret_std = float(rets.std())
            len_mean = float(lens.mean())
            len_std = float(lens.std())
        else:
            ret_mean = None
            ret_std = None
            len_mean = None
            len_std = None

        runtime = max(time.perf_counter() - start_time, 1e-9)
        env_steps = int(step_calls * n_envs)
        return {
            "eval/seed": int(eval_seed),
            "eval/num_envs": int(n_envs),
            "eval/deterministic": bool(deterministic),
            "eval/episodes_target": int(eval_episodes),
            "eval/episodes_collected": int(episodes_collected),
            "eval/ret_mean": ret_mean,
            "eval/ret_std": ret_std,
            "eval/len_mean": len_mean,
            "eval/len_std": len_std,
            "eval/env_steps": env_steps,
            "eval/runtime_sec": runtime,
            "eval/fps": (env_steps / runtime) if env_steps > 0 else None,
        }
    finally:
        if eval_envs is not None:
            eval_envs.close()
        ac.train(ac_was_training)
        _restore_rng_state(rng_state)


def train_recurrent(
    args,
    envs: EnvPool,
    obs0: np.ndarray,
    obs_dim: int,
    obs_shape: tuple[int, ...],
    act_dim: int,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_scaler: torch.amp.GradScaler | None,
    reporter,
    mask_indices: list[int],
    eval_interval: int,
    eval_episodes: int,
    eval_num_envs: int,
    eval_seed: int,
    eval_deterministic: bool,
):
    device_type = torch.device(device).type
    ac = RecurrentActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=args.act_embed_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=args.feat_dim,
        encoder_type=args.encoder,
        obs_shape=obs_shape,
    ).to(device)
    ac = maybe_compile_module(ac, enabled=bool(args.compile), mode=str(args.compile_mode))
    opt = make_adam_optimizer(
        ac.parameters(),
        args.lr,
        device_type=device_type,
        adam_foreach=args.adam_foreach,
        adam_fused=args.adam_fused,
    )
    updates = args.total_steps // (args.num_envs * args.horizon)
    lr_scheduler = make_lr_scheduler(
        opt,
        total_updates=updates,
        schedule=args.lr_schedule,
        start_lr=args.lr,
        end_lr=args.lr_end,
    )

    reporter.log_block(
        "model",
        {
            "policy": args.policy,
            "algorithm": args.algo,
            "encoder": args.encoder,
            "module": module_param_stats(ac),
            "optimizer": optimizer_param_stats(opt),
            "lr_scheduler": {
                "type": str(args.lr_schedule),
                "start_lr": float(args.lr),
                "end_lr": (float(args.lr_end) if args.lr_end is not None else (0.0 if str(args.lr_schedule) == "linear" else None)),
                "total_updates": int(updates),
            },
            "amp": {"enabled": bool(use_amp), "dtype": str(amp_dtype).replace("torch.", "")},
        },
    )
    reporter.log_block("recurrent_actor_critic", str(ac))
    reporter.update_summary({"model_architecture": {"recurrent_actor_critic": str(ac)}})

    early_stopper = EarlyStopper(
        metric=args.early_stop_metric,
        mode=args.early_stop_mode,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        warmup_updates=args.early_stop_warmup_updates,
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    prev_action = torch.zeros(args.num_envs, device=device, dtype=torch.int64)
    hidden = ac.init_hidden(args.num_envs, device)
    obs = obs0

    wandb_run = init_wandb(args)
    if wandb_run is not None:
        reporter.update_summary(
            {
                "wandb": {
                    "run_id": getattr(wandb_run, "id", None),
                    "run_name": getattr(wandb_run, "name", None),
                    "project": getattr(wandb_run, "project", None),
                    "entity": getattr(wandb_run, "entity", None),
                    "url": getattr(wandb_run, "url", None),
                }
            }
        )
    train_start = time.perf_counter()
    progress = None
    if (tqdm is not None) and (not getattr(args, "no_tqdm", False)):
        progress = tqdm(total=updates, dynamic_ncols=True, desc=f"{args.env_id} | recurrent+ppo", leave=True)
    profiler = TorchProfilerController(args=args, reporter=reporter, device=device, phase_label="recurrent")

    final_metrics = None
    final_checkpoint = None
    updates_completed = 0
    best_ret50 = None

    def _recurrent_checkpoint_payload(metrics: dict | None) -> dict:
        return {
            "policy_state": ac.state_dict(),
            "optimizer_state": opt.state_dict(),
            "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "args": vars(args),
            "frames": updates_completed * args.num_envs * args.horizon,
            "metrics": metrics,
        }

    try:
        for upd in range(updates):
            updates_completed = upd + 1
            debug_ppo = bool(args.debug_log)
            episodes_before = len(envs.episode_returns) if debug_ppo else 0
            batch, obs, prev_action, hidden = rollout_recurrent(
                envs=envs,
                ac=ac,
                device=device,
                horizon=args.horizon,
                gamma=args.gamma,
                obs_normalization=args.obs_normalization,
                obs=obs,
                prev_action=prev_action,
                hidden=hidden,
            )
            ended_episodes = envs.episode_returns[episodes_before:] if debug_ppo else []
            ppo_debug_cfg = None
            if debug_ppo:
                ppo_debug_cfg = {
                    "seed": int(args.seed),
                    "update_idx": int(upd + 1),
                    "action_bins": int(act_dim),
                    "ratio_sample_size": 4096,
                    "frame_delta_pairs": 128,
                }

            ppo_stats = ppo_update_recurrent(
                ac=ac,
                opt=opt,
                batch=batch,
                clip_coef=args.clip_coef,
                vf_clip=bool(args.vf_clip),
                target_kl=args.target_kl,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                ent_coef=args.ent_coef,
                epochs=args.epochs,
                lam=args.gae_lam,
                gamma=args.gamma,
                generator=rng,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                grad_scaler=grad_scaler,
                debug_cfg=ppo_debug_cfg,
            )

            metrics = {
                "loop/update": upd + 1,
                "loop/frames": (upd + 1) * args.num_envs * args.horizon,
                "optim/lr": float(opt.param_groups[0]["lr"]),
            }
            for key, value in ppo_stats.items():
                if str(key).startswith("debug/"):
                    metrics[key] = value
                else:
                    metrics[f"loss/{key}"] = value
            if debug_ppo:
                metrics.update(
                    build_ppo_rollout_debug_metrics(
                        args=args,
                        batch=batch,
                        update_idx=upd + 1,
                        updates_total=updates,
                        total_steps_target=int(args.total_steps),
                        ended_episodes=ended_episodes,
                    )
                )
            add_perf_metrics(metrics, train_start=train_start, updates_done=upd + 1, updates_total=updates)

            mean_ret, mean_len = recent_return_stats(envs.episode_returns, window=50)
            if mean_ret is not None:
                metrics["train/ret50"] = mean_ret
                metrics["train/len50"] = mean_len

            do_eval = (eval_interval > 0) and ((upd + 1) % eval_interval == 0)
            if do_eval:
                eval_metrics = evaluate_recurrent_policy(
                    args=args,
                    mask_indices=mask_indices,
                    ac=ac,
                    device=device,
                    eval_seed=eval_seed,
                    eval_num_envs=eval_num_envs,
                    eval_episodes=eval_episodes,
                    deterministic=eval_deterministic,
                )
                metrics.update(eval_metrics)
                eval_ret = eval_metrics.get("eval/ret_mean")
                eval_len = eval_metrics.get("eval/len_mean")
                eval_line = (
                    f"eval update={upd+1:04d}  seed={eval_seed}  "
                    f"episodes={eval_metrics.get('eval/episodes_collected')}/{eval_metrics.get('eval/episodes_target')}"
                )
                if (eval_ret is not None) and (eval_len is not None):
                    eval_line += f"  ret_mean={eval_ret:8.2f}  len_mean={eval_len:6.1f}"
                reporter.log_line(eval_line)

            should_stop, stop_reason = early_stopper.update(update_idx=upd + 1, metrics=metrics)
            if stop_reason:
                reporter.log_line(stop_reason)
                print(stop_reason)

            reporter.log_metrics(metrics)
            ret50_value = finite_metric(metrics.get("train/ret50"))
            if (ret50_value is not None) and ((best_ret50 is None) or (ret50_value > best_ret50)):
                best_ret50 = ret50_value
                reporter.save_checkpoint(_recurrent_checkpoint_payload(metrics), filename="checkpoint_best.pt")
            if mean_ret is not None:
                log_line = (
                    f"update={upd+1:04d}  episodes={len(envs.episode_returns):06d}  "
                    f"ret50={mean_ret:8.2f}  len50={mean_len:6.1f}  "
                    f"kl={ppo_stats['approx_kl']:7.4f}  clipfrac={ppo_stats['clipfrac']:6.3f}"
                )
                eval_ret = metrics.get("eval/ret_mean")
                eval_len = metrics.get("eval/len_mean")
                if (eval_ret is not None) and (eval_len is not None):
                    log_line += f"  eval_ret={eval_ret:8.2f}  eval_len={eval_len:6.1f}"
                reporter.log_line(log_line)
                if (upd + 1) % args.log_interval == 0:
                    print(log_line)

            if wandb_run is not None:
                wandb_run.log(metrics, step=metrics["loop/frames"])
            final_metrics = metrics
            if lr_scheduler is not None:
                lr_scheduler.step()
            if progress is not None:
                progress.set_postfix(
                    {
                        "ret50": f"{metrics.get('train/ret50', float('nan')):.2f}",
                        "fps": f"{metrics['perf/fps']:.1f}",
                        "eta_min": f"{metrics['perf/eta_sec'] / 60.0:.1f}",
                    }
                )
                progress.update(1)
            profiler.step()
            if should_stop:
                break
        final_payload = _recurrent_checkpoint_payload(final_metrics)
        final_checkpoint = reporter.save_checkpoint(final_payload, filename="checkpoint.pt")
        reporter.save_checkpoint(final_payload, filename="checkpoint_last.pt")
    finally:
        profiler.close()
        if wandb_run is not None:
            wandb_run.finish()
        if progress is not None:
            progress.close()
    reporter.finalize(final_metrics, final_checkpoint)


def make_env_fn(
    env_id: str,
    seed: int,
    mask_indices: list[int],
    phase_len: int,
    obs_shift_scale: float,
    reward_scale_low: float,
    reward_scale_high: float,
    carracing_downsample: int = 1,
    carracing_grayscale: bool = False,
    frame_stack: int = 1,
):
    def _thunk():
        env = gym.make(env_id)
        if env_id.startswith("CarRacing"):
            env = DiscreteCarRacingWrapper(env)
            if (carracing_downsample > 1) or carracing_grayscale:
                env = CarRacingPreprocessWrapper(
                    env,
                    downsample=carracing_downsample,
                    grayscale=carracing_grayscale,
                )
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
        if frame_stack > 1:
            env = FrameStackLastAxisWrapper(env, num_stack=frame_stack)
        return env

    return _thunk


def main():
    load_env_file()
    default_device = os.environ.get("AMT_DEVICE") or os.environ.get("DEVICE") or "cuda"

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="YAML config file (overrides positional config).")
    p.add_argument("config_path", nargs="?", default=None, help="YAML config file (positional).")
    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument(
        "--env-workers",
        type=int,
        default=0,
        help="Number of threads used to step/reset envs (0/1 runs serially). Helpful for CarRacing.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=default_device)
    p.add_argument("--cuda-id", type=int, default=None, help="CUDA device index override (e.g., 0-7).")
    p.add_argument("--policy", type=str, default="amt", choices=["amt", "recurrent", "ff"], help="Policy architecture.")
    p.add_argument(
        "--algo",
        type=str,
        default="ppo",
        help="Training algorithm: ppo, a2c, trpo, reinforce, v-trace, v-mpo, dqn.",
    )
    p.add_argument("--carracing-downsample", type=int, default=1, help="Downsample CarRacing image observations by K.")
    p.add_argument(
        "--carracing-grayscale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Convert CarRacing RGB observations to grayscale (1 channel).",
    )
    p.add_argument(
        "--frame-stack",
        type=int,
        default=1,
        help="Stack the last K observations along the last axis (recommended with --encoder cnn).",
    )

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--feat-dim", type=int, default=64)
    p.add_argument("--act-embed-dim", type=int, default=16)
    p.add_argument("--encoder", type=str, default="mlp", choices=["mlp", "cnn"], help="Observation encoder type.")
    p.add_argument(
        "--obs-normalization",
        type=str,
        default="auto",
        choices=["auto", "none", "uint8", "imagenet"],
        help=(
            "Observation normalization before model input: auto (scale uint8 by 1/255), "
            "none, uint8 (always /255), or imagenet (ImageNet mean/std, channel-last)."
        ),
    )

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

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--lr-schedule",
        type=str,
        default="linear",
        choices=["none", "linear"],
        help="Learning-rate schedule applied once per update (linear uses torch.optim.lr_scheduler.LambdaLR).",
    )
    p.add_argument(
        "--lr-end",
        type=float,
        default=None,
        help="Final learning rate for linear decay. Must be null/omitted when --lr-schedule=none.",
    )
    p.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow TF32 matmuls on CUDA for speed (auto-enabled on CUDA if unset).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile on model forward passes (can improve throughput after warmup).",
    )
    p.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode.",
    )
    p.add_argument(
        "--torch-profiler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.profiler to capture per-update CPU/CUDA bottlenecks.",
    )
    p.add_argument(
        "--torch-profiler-dir",
        type=str,
        default=None,
        help="Output directory for profiler traces/key averages. Default: <run_dir>/profiler/<phase> when --report.",
    )
    p.add_argument("--torch-profiler-wait", type=int, default=1, help="Profiler schedule wait steps.")
    p.add_argument("--torch-profiler-warmup", type=int, default=1, help="Profiler schedule warmup steps.")
    p.add_argument("--torch-profiler-active", type=int, default=3, help="Profiler schedule active steps.")
    p.add_argument("--torch-profiler-repeat", type=int, default=1, help="Profiler schedule repeat count.")
    p.add_argument("--torch-profiler-skip-first", type=int, default=0, help="Skip this many initial profiler steps.")
    p.add_argument(
        "--torch-profiler-record-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record operator input shapes in profiler.",
    )
    p.add_argument(
        "--torch-profiler-profile-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track memory usage in profiler.",
    )
    p.add_argument(
        "--torch-profiler-with-stack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture Python stack traces in profiler (higher overhead).",
    )
    p.add_argument(
        "--torch-profiler-with-flops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Estimate FLOPs for supported operators.",
    )
    p.add_argument(
        "--torch-profiler-sort-by",
        type=str,
        default="",
        help="Sort key for key_averages table (default: self_cuda_time_total on CUDA, self_cpu_time_total on CPU).",
    )
    p.add_argument("--torch-profiler-row-limit", type=int, default=40, help="Row count in key_averages table.")
    p.add_argument(
        "--adam-foreach",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use foreach Adam implementation (auto-enabled when not using fused Adam).",
    )
    p.add_argument(
        "--adam-fused",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use fused Adam implementation (CUDA only; auto-enabled on CUDA if unset).",
    )
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument(
        "--target-kl",
        type=float,
        default=0.01,
        help="PPO KL early-stop threshold per update. If approx_kl exceeds this, remaining PPO epochs are skipped.",
    )
    p.add_argument(
        "--vf-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable PPO value-function clipping using --clip-coef.",
    )
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--ema-tau", type=float, default=0.995)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar and ETA display.")
    p.add_argument("--debug-log", action=argparse.BooleanOptionalAction, default=False, help="Enable PPO debug metrics logging.")
    p.add_argument(
        "--early-stop-metric",
        type=str,
        default="",
        help="Metric name for early stopping (e.g., train/ret50 or eval/ret_mean). Empty disables early stopping.",
    )
    p.add_argument(
        "--early-stop-mode",
        type=str,
        default="max",
        choices=["max", "min"],
        help="Early-stop direction: max (higher is better) or min (lower is better).",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving updates. <=0 disables early stopping.",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum absolute improvement required to reset early-stop patience.",
    )
    p.add_argument(
        "--early-stop-warmup-updates",
        type=int,
        default=0,
        help="Ignore early-stop checks for the first N updates.",
    )

    p.add_argument("--eval-interval", type=int, default=0, help="Run evaluation every N updates (0 disables).")
    p.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes per evaluation call.")
    p.add_argument(
        "--eval-num-envs",
        type=int,
        default=None,
        help="Number of parallel envs for evaluation (default: min(num_envs, eval_episodes)).",
    )
    p.add_argument("--eval-seed", type=int, default=None, help="Base seed for evaluation envs (default: seed + eval_seed_offset).")
    p.add_argument("--eval-seed-offset", type=int, default=1_000_000, help="Offset added to --seed when --eval-seed is not set.")
    p.add_argument("--eval-stochastic", action="store_true", help="Sample actions during eval (default: greedy/argmax).")

    p.add_argument("--trpo-max-kl", type=float, default=0.01)
    p.add_argument("--trpo-backtrack-coef", type=float, default=0.5)
    p.add_argument("--trpo-backtrack-iters", type=int, default=10)
    p.add_argument("--trpo-value-epochs", type=int, default=2)

    p.add_argument("--vtrace-rho-clip", type=float, default=1.0)
    p.add_argument("--vtrace-c-clip", type=float, default=1.0)

    p.add_argument("--vmpo-topk-frac", type=float, default=0.5)
    p.add_argument("--vmpo-eta", type=float, default=1.0)
    p.add_argument("--vmpo-kl-coef", type=float, default=1.0)
    p.add_argument("--vmpo-kl-target", type=float, default=0.01)

    p.add_argument("--dqn-replay-size", type=int, default=100_000)
    p.add_argument("--dqn-batch-size", type=int, default=256)
    p.add_argument("--dqn-learning-starts", type=int, default=2_048)
    p.add_argument("--dqn-updates-per-iter", type=int, default=1)
    p.add_argument("--dqn-target-update-interval", type=int, default=200)
    p.add_argument("--dqn-double", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dqn-eps-start", type=float, default=1.0)
    p.add_argument("--dqn-eps-end", type=float, default=0.05)
    p.add_argument("--dqn-eps-decay-steps", type=int, default=100_000)

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

    p.add_argument("--report", action="store_true", help="Write a run summary + checkpoint to the report dir.")
    p.add_argument("--report-dir", type=str, default="reports")
    p.add_argument("--report-run-name", type=str, default=None)
    p.add_argument("--run-postfix", type=str, default="", help="Optional postfix appended to run names.")
    p.add_argument(
        "--run-note",
        type=str,
        default=None,
        help="Required run note describing intent, hypothesis, and key distinguishing details.",
    )

    cli_dests = collect_cli_provided_dests(p, sys.argv[1:])

    # two-pass parse to apply YAML defaults then allow CLI overrides
    partial_args, _ = p.parse_known_args()
    if partial_args.config and partial_args.config_path and (partial_args.config != partial_args.config_path):
        raise ValueError("Provide a single config path (either positional or --config).")
    config_input = partial_args.config or partial_args.config_path
    if config_input is None:
        raise ValueError("A config file is required. Pass it as positional <config.yaml> or via --config <path>.")
    config_path = Path(config_input)
    cfg, root_cfg = load_config_file(config_path)
    validate_no_unknown_config_keys(resolved_cfg=cfg, parser=p)
    p.set_defaults(**{k: v for k, v in cfg.items() if hasattr(partial_args, k)})
    args = p.parse_args()
    args._source_config = {"root": root_cfg, "resolved": cfg}
    args._source_config_path = str(config_path.resolve())
    args.algo = normalize_algo_name(args.algo)
    if args.algo not in ALL_ALGOS:
        choices = ", ".join(sorted(ALL_ALGOS))
        raise ValueError(f"Unsupported --algo '{args.algo}'. Expected one of: {choices}")
    validate_explicit_required_keys(args, resolved_cfg=cfg, cli_dests=cli_dests)
    validate_no_strange_params(args, resolved_cfg=cfg, cli_dests=cli_dests)
    args.run_note = (args.run_note or "").strip()
    args.run_postfix = sanitize_postfix(args.run_postfix)
    if not args.run_note:
        raise ValueError(
            "Run note is required. Pass --run-note with a concise note that captures intent and distinguishing details."
        )
    if args.run_postfix:
        args.report_run_name = with_postfix(args.report_run_name, args.run_postfix)
        args.wandb_run_name = with_postfix(args.wandb_run_name, args.run_postfix)
    set_seed(args.seed)

    if str(args.device).startswith("cuda") and args.cuda_id is None:
        raise ValueError(
            "GPU selection is required. Pass `--cuda-id <index>` in the command, e.g. `--device cuda --cuda-id 0`."
        )

    device = resolve_device(args.device, args.cuda_id)
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tf32_enabled = bool(args.tf32) if args.tf32 is not None else (device_type == "cuda")
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        if tf32_enabled:
            torch.set_float32_matmul_precision("high")
    if args.tf32 is None:
        args.tf32 = tf32_enabled
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
    if args.algo == "dqn":
        # DQN path does not optimize predictor parameters.
        args.lambda_pred = 0.0
        args.pred_coef = 0.0
        if args.encoder == "cnn":
            raise ValueError("`--encoder cnn` is not supported with `--algo dqn` yet (replay stores flattened observations).")
    if args.policy == "recurrent" and args.algo != "ppo":
        raise ValueError("`--policy recurrent` currently supports only `--algo ppo`.")
    if (args.algo == "ppo" or args.policy == "recurrent") and float(args.target_kl) <= 0.0:
        raise ValueError("`--target-kl` must be > 0 for PPO training.")
    if int(args.frame_stack) < 1:
        raise ValueError("`--frame-stack` must be >= 1.")
    if int(args.frame_stack) > 1 and args.encoder != "cnn":
        raise ValueError("`--frame-stack > 1` is supported only with `--encoder cnn`.")
    alpha_base_list = parse_floats(args.alpha_base)
    alpha_max_list = parse_floats(args.alpha_max)
    assert len(alpha_base_list) == len(alpha_max_list)
    M = len(alpha_base_list)

    mask_indices = parse_ints(args.mask_indices)

    eval_interval = int(args.eval_interval)
    if eval_interval < 0:
        raise ValueError("--eval-interval must be >= 0.")
    eval_episodes = int(args.eval_episodes)
    if (eval_interval > 0) and (eval_episodes < 1):
        raise ValueError("--eval-episodes must be >= 1 when --eval-interval > 0.")
    eval_num_envs = int(args.eval_num_envs) if args.eval_num_envs is not None else min(args.num_envs, max(eval_episodes, 1))
    if (eval_interval > 0) and (eval_num_envs < 1):
        raise ValueError("--eval-num-envs must be >= 1 when --eval-interval > 0.")
    eval_seed = int(args.eval_seed) if args.eval_seed is not None else int(args.seed + int(args.eval_seed_offset))
    eval_deterministic = not bool(args.eval_stochastic)
    if args.eval_seed is None:
        args.eval_seed = eval_seed
    if args.eval_num_envs is None:
        args.eval_num_envs = eval_num_envs

    if int(args.torch_profiler_wait) < 0:
        raise ValueError("--torch-profiler-wait must be >= 0.")
    if int(args.torch_profiler_warmup) < 0:
        raise ValueError("--torch-profiler-warmup must be >= 0.")
    if int(args.torch_profiler_active) < 1:
        raise ValueError("--torch-profiler-active must be >= 1.")
    if int(args.torch_profiler_repeat) < 1:
        raise ValueError("--torch-profiler-repeat must be >= 1.")
    if int(args.torch_profiler_skip_first) < 0:
        raise ValueError("--torch-profiler-skip-first must be >= 0.")
    if int(args.torch_profiler_row_limit) < 1:
        raise ValueError("--torch-profiler-row-limit must be >= 1.")
    if args.torch_profiler and (eval_interval > 0):
        print("Note: torch profiler is enabled with evaluation; eval steps are included in profiled updates.")

    env_workers = int(args.env_workers)
    if env_workers < 0:
        raise ValueError("--env-workers must be >= 0.")
    if env_workers > 1 and str(args.env_id).startswith("CarRacing"):
        print(
            "Note: CarRacing with --env-workers > 1 uses threaded stepping; "
            "if you observe nondeterminism or rare crashes, try --env-workers 0/1."
        )
    frames_per_update = int(args.num_envs) * int(args.horizon)
    if frames_per_update <= 0:
        raise ValueError("--num-envs and --horizon must both be > 0.")
    if int(args.total_steps) < frames_per_update:
        raise ValueError(
            f"--total-steps ({args.total_steps}) must be >= num_envs*horizon ({frames_per_update}) for a full update."
        )
    remainder = int(args.total_steps) % frames_per_update
    if remainder != 0:
        raise ValueError(
            f"--total-steps ({args.total_steps}) must be divisible by num_envs*horizon ({frames_per_update}) "
            f"for full-training runs (remainder={remainder})."
        )

    env_fns = [
        make_env_fn(
            env_id=args.env_id,
            seed=args.seed + 10_000 * i,
            mask_indices=mask_indices,
            phase_len=args.phase_len,
            obs_shift_scale=args.obs_shift_scale,
            reward_scale_low=args.reward_scale_low,
            reward_scale_high=args.reward_scale_high,
            carracing_downsample=int(args.carracing_downsample),
            carracing_grayscale=bool(args.carracing_grayscale),
            frame_stack=int(args.frame_stack),
        )
        for i in range(args.num_envs)
    ]
    envs = EnvPool(env_fns, workers=env_workers)

    obs0, _ = envs.reset(seed=args.seed)
    if not isinstance(envs.single_observation_space, gym.spaces.Box):
        raise TypeError(f"Expected Box observation space, got {type(envs.single_observation_space)}")
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise TypeError(
            "Only Discrete action spaces are supported in this trainer. "
            "For CarRacing, use `env_id=CarRacing-v3` (auto-discretized)."
        )

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(envs.single_action_space.n)
    feat_dim = args.feat_dim
    mem_dim = M * feat_dim

    reporter = start_run_report(
        repo_root=Path(__file__).resolve().parent,
        report_dir=Path(args.report_dir),
        run_name=args.report_run_name or args.wandb_run_name,
        args=vars(args),
        device=device,
        obs_dim=obs_dim,
        act_dim=act_dim,
        mask_indices=mask_indices,
        config_path=str(config_input) if config_input else None,
        enabled=args.report,
    )

    benchmark = infer_regime(
        mask_indices=mask_indices,
        phase_len=args.phase_len,
        obs_shift_scale=args.obs_shift_scale,
        reward_scale_low=args.reward_scale_low,
        reward_scale_high=args.reward_scale_high,
    )
    if reporter.enabled:
        env0 = envs.envs[0]
        car_racing_wrapper = find_wrapper(env0, DiscreteCarRacingWrapper)
        reporter.log_block(
            "data",
            {
                "env_id": args.env_id,
                "num_envs": int(args.num_envs),
                "env_workers": int(env_workers),
                "horizon": int(args.horizon),
                "total_steps": int(args.total_steps),
                "updates": int(args.total_steps // max(args.num_envs * args.horizon, 1)),
                "observation_space": str(envs.single_observation_space),
                "observation_dtype": str(getattr(envs.single_observation_space, "dtype", None)),
                "obs_normalization": str(args.obs_normalization),
                "action_space": str(envs.single_action_space),
                "wrappers": env_wrapper_chain(env0),
                "mask_indices": mask_indices,
                "benchmark": benchmark,
                "carracing_preprocess": {
                    "downsample": int(args.carracing_downsample),
                    "grayscale": bool(args.carracing_grayscale),
                    "frame_stack": int(args.frame_stack),
                },
                "obs0": numpy_array_stats(obs0),
                "car_racing_action_table": (
                    {
                        "num_actions": int(car_racing_wrapper.action_space.n),
                        "table_shape": list(car_racing_wrapper.action_table.shape),
                    }
                    if car_racing_wrapper is not None
                    else None
                ),
                "amp": {"enabled": bool(use_amp), "dtype": str(amp_dtype).replace("torch.", "")},
                "resolved_device": device,
            },
        )
    reporter.update_summary(
        {
            "model": {
                "policy": args.policy,
                "algorithm": args.algo,
                "encoder": args.encoder,
                "hidden_dim": args.hidden_dim,
                "feat_dim": args.feat_dim,
                "act_embed_dim": args.act_embed_dim,
            },
            "benchmark": benchmark,
        }
    )

    if args.policy == "recurrent":
        train_recurrent(
            args=args,
            envs=envs,
            obs0=obs0,
            obs_dim=obs_dim,
            obs_shape=tuple(envs.single_observation_space.shape),
            act_dim=act_dim,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            reporter=reporter,
            mask_indices=mask_indices,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            eval_num_envs=eval_num_envs,
            eval_seed=eval_seed,
            eval_deterministic=eval_deterministic,
        )
        return

    ac = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=args.act_embed_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=feat_dim,
        mem_dim=mem_dim,
        encoder_type=args.encoder,
        obs_shape=tuple(envs.single_observation_space.shape),
    ).to(device)

    f_mem = FeatureEncoder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=args.act_embed_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=feat_dim,
        encoder_type=args.encoder,
        obs_shape=tuple(envs.single_observation_space.shape),
    ).to(device)
    f_mem.load_state_dict(ac.f_pol.state_dict())

    predictor = None
    if (args.lambda_pred > 0.0) or (args.pred_coef > 0.0):
        predictor = Predictor(feat_dim=feat_dim, act_dim=act_dim, hidden_dim=args.hidden_dim).to(device)

    ac = maybe_compile_module(ac, enabled=bool(args.compile), mode=str(args.compile_mode))
    f_mem = maybe_compile_module(f_mem, enabled=bool(args.compile), mode=str(args.compile_mode))
    if predictor is not None:
        predictor = maybe_compile_module(predictor, enabled=bool(args.compile), mode=str(args.compile_mode))

    alpha_base = torch.tensor(alpha_base_list, device=device, dtype=torch.float32).unsqueeze(0)  # (1, M)
    alpha_max = torch.tensor(alpha_max_list, device=device, dtype=torch.float32).unsqueeze(0)

    traces = torch.zeros((args.num_envs, M, feat_dim), device=device)
    prev_action = torch.zeros(args.num_envs, device=device, dtype=torch.int64)

    with torch.no_grad():
        obs0_t = obs_to_tensor(obs0, device=device, obs_normalization=args.obs_normalization)
        x_mem0 = encode_mem(f_mem, obs0_t, prev_action)
        traces = trace_update(traces, x_mem0, alpha_base.expand(args.num_envs, -1))

    skip_drift = (args.reset_strategy == "none") and torch.allclose(alpha_base, alpha_max)
    drift = None
    if not skip_drift:
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
    opt = make_adam_optimizer(
        params,
        args.lr,
        device_type=device_type,
        adam_foreach=args.adam_foreach,
        adam_fused=args.adam_fused,
    )
    updates = args.total_steps // (args.num_envs * args.horizon)
    lr_scheduler = make_lr_scheduler(
        opt,
        total_updates=updates,
        schedule=args.lr_schedule,
        start_lr=args.lr,
        end_lr=args.lr_end,
    )

    reporter.log_block(
        "model",
        {
            "policy": args.policy,
            "algorithm": args.algo,
            "encoder": args.encoder,
            "actor_critic": module_param_stats(ac),
            "feature_encoder_ema": module_param_stats(f_mem),
            "predictor": module_param_stats(predictor) if predictor is not None else None,
            "optimizer": optimizer_param_stats(opt),
            "lr_scheduler": {
                "type": str(args.lr_schedule),
                "start_lr": float(args.lr),
                "end_lr": (float(args.lr_end) if args.lr_end is not None else (0.0 if str(args.lr_schedule) == "linear" else None)),
                "total_updates": int(updates),
            },
            "memory": {"M": int(M), "feat_dim": int(feat_dim), "mem_dim": int(mem_dim)},
            "drift_monitor": (
                {
                    "enabled": True,
                    "rho_s": float(args.rho_s),
                    "rho_l": float(args.rho_l),
                    "beta": float(args.beta),
                    "tau_soft": float(args.tau_soft),
                    "kappa": float(args.kappa),
                    "tau_on": float(args.tau_on),
                    "tau_off": float(args.tau_off),
                    "K": int(args.K),
                    "cooldown_steps": int(args.cooldown_steps),
                    "warmup_steps": int(args.warmup_steps),
                }
                if drift is not None
                else {"enabled": False, "reason": "fixed-trace/no-reset configuration"}
            ),
            "amp": {"enabled": bool(use_amp), "dtype": str(amp_dtype).replace("torch.", "")},
        },
    )
    reporter.log_block("actor_critic", str(ac))
    reporter.log_block("feature_encoder_ema", str(f_mem))
    if predictor is not None:
        reporter.log_block("predictor", str(predictor))
    reporter.update_summary(
        {
            "model_architecture": {
                "actor_critic": str(ac),
                "feature_encoder_ema": str(f_mem),
                "predictor": str(predictor) if predictor is not None else None,
            }
        }
    )

    early_stopper = EarlyStopper(
        metric=args.early_stop_metric,
        mode=args.early_stop_mode,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        warmup_updates=args.early_stop_warmup_updates,
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    dqn_action_rng = torch.Generator(device=device)
    dqn_action_rng.manual_seed(args.seed + 12345)

    obs = obs0
    wandb_run = init_wandb(args)
    dqn_target_ac = None
    replay = None
    dqn_opt_steps = 0
    if args.algo == "dqn":
        dqn_target_ac = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_embed_dim=args.act_embed_dim,
            hidden_dim=args.hidden_dim,
            feat_dim=feat_dim,
            mem_dim=mem_dim,
            encoder_type=args.encoder,
            obs_shape=tuple(envs.single_observation_space.shape),
        ).to(device)
        hard_update_(dqn_target_ac, ac)
        dqn_target_ac = maybe_compile_module(dqn_target_ac, enabled=bool(args.compile), mode=str(args.compile_mode))
        replay = DQNReplayBuffer(capacity=args.dqn_replay_size, obs_dim=obs_dim, trace_dim=mem_dim)
        reporter.log_block(
            "dqn",
            {
                "target_actor_critic": module_param_stats(dqn_target_ac),
                "replay": {
                    "capacity": int(args.dqn_replay_size),
                    "obs_dim": int(obs_dim),
                    "trace_dim": int(mem_dim),
                },
            },
        )
    train_start = time.perf_counter()
    progress = None
    if (tqdm is not None) and (not args.no_tqdm):
        progress = tqdm(
            total=updates,
            dynamic_ncols=True,
            desc=f"{args.env_id} | {args.policy}+{args.algo}",
            leave=True,
        )
    profiler = TorchProfilerController(
        args=args,
        reporter=reporter,
        device=device,
        phase_label=f"{str(args.policy)}_{str(args.algo)}",
    )

    final_metrics = None
    final_checkpoint = None
    updates_completed = 0
    best_ret50 = None

    def _trace_checkpoint_payload(metrics: dict | None) -> dict:
        return {
            "policy_state": ac.state_dict(),
            "optimizer_state": opt.state_dict(),
            "lr_scheduler_state": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "predictor_state": predictor.state_dict() if predictor is not None else None,
            "f_mem_state": f_mem.state_dict(),
            "target_policy_state": dqn_target_ac.state_dict() if dqn_target_ac is not None else None,
            "replay_size": int(len(replay)) if replay is not None else None,
            "args": vars(args),
            "frames": updates_completed * args.num_envs * args.horizon,
            "metrics": metrics,
        }

    try:
        for upd in range(updates):
            updates_completed = upd + 1
            should_stop = False
            if args.algo in ON_POLICY_ALGOS:
                debug_ppo = bool(args.debug_log and (args.algo == "ppo"))
                episodes_before = len(envs.episode_returns) if debug_ppo else 0
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
                    obs_normalization=args.obs_normalization,
                    alpha_base=alpha_base,
                    alpha_max=alpha_max,
                    reset_strategy=args.reset_strategy,
                    reset_long_fraction=args.reset_long_fraction,
                    obs=obs,
                    prev_action=prev_action,
                    traces=traces,
                )
                ended_episodes = envs.episode_returns[episodes_before:] if debug_ppo else []
                ppo_debug_cfg = None
                if debug_ppo:
                    ppo_debug_cfg = {
                        "seed": int(args.seed),
                        "update_idx": int(upd + 1),
                        "action_bins": int(act_dim),
                        "ratio_sample_size": 4096,
                        "frame_delta_pairs": 128,
                    }

                algo_stats = update_on_policy(
                    algo=args.algo,
                    ac=ac,
                    opt=opt,
                    batch=batch,
                    clip_coef=args.clip_coef,
                    vf_clip=bool(args.vf_clip),
                    target_kl=args.target_kl,
                    vf_coef=args.vf_coef,
                    max_grad_norm=args.max_grad_norm,
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
                    trpo_max_kl=args.trpo_max_kl,
                    trpo_backtrack_coef=args.trpo_backtrack_coef,
                    trpo_backtrack_iters=args.trpo_backtrack_iters,
                    trpo_value_epochs=args.trpo_value_epochs,
                    vtrace_rho_clip=args.vtrace_rho_clip,
                    vtrace_c_clip=args.vtrace_c_clip,
                    vmpo_topk_frac=args.vmpo_topk_frac,
                    vmpo_eta=args.vmpo_eta,
                    vmpo_kl_coef=args.vmpo_kl_coef,
                    vmpo_kl_target=args.vmpo_kl_target,
                    debug_cfg=ppo_debug_cfg,
                )
                ema_update_(f_mem, ac.f_pol, tau=args.ema_tau)

                metrics = {
                    "loop/update": upd + 1,
                    "loop/frames": (upd + 1) * args.num_envs * args.horizon,
                    "diagnostics/gate_mean": (float(drift.state.gate.mean().item()) if drift is not None else 0.0),
                    "optim/lr": float(opt.param_groups[0]["lr"]),
                }
                for key, value in algo_stats.items():
                    if str(key).startswith("debug/"):
                        metrics[key] = value
                    else:
                        metrics[f"loss/{key}"] = value
                if debug_ppo:
                    metrics.update(
                        build_ppo_rollout_debug_metrics(
                            args=args,
                            batch=batch,
                            update_idx=upd + 1,
                            updates_total=updates,
                            total_steps_target=int(args.total_steps),
                            ended_episodes=ended_episodes,
                        )
                    )
                add_perf_metrics(metrics, train_start=train_start, updates_done=upd + 1, updates_total=updates)

                mean_ret, mean_len = recent_return_stats(envs.episode_returns, window=50)
                if mean_ret is not None:
                    metrics["train/ret50"] = mean_ret
                    metrics["train/len50"] = mean_len

                do_eval = (eval_interval > 0) and ((upd + 1) % eval_interval == 0)
                if do_eval:
                    eval_metrics = evaluate_trace_policy(
                        args=args,
                        mask_indices=mask_indices,
                        ac=ac,
                        f_mem=f_mem,
                        predictor=predictor,
                        alpha_base=alpha_base,
                        alpha_max=alpha_max,
                        device=device,
                        eval_seed=eval_seed,
                        eval_num_envs=eval_num_envs,
                        eval_episodes=eval_episodes,
                        deterministic=eval_deterministic,
                    )
                    metrics.update(eval_metrics)
                    eval_ret = eval_metrics.get("eval/ret_mean")
                    eval_len = eval_metrics.get("eval/len_mean")
                    eval_line = (
                        f"eval update={upd+1:04d}  seed={eval_seed}  "
                        f"episodes={eval_metrics.get('eval/episodes_collected')}/{eval_metrics.get('eval/episodes_target')}"
                    )
                    if (eval_ret is not None) and (eval_len is not None):
                        eval_line += f"  ret_mean={eval_ret:8.2f}  len_mean={eval_len:6.1f}"
                    reporter.log_line(eval_line)

                should_stop, stop_reason = early_stopper.update(update_idx=upd + 1, metrics=metrics)
                if stop_reason:
                    reporter.log_line(stop_reason)
                    print(stop_reason)

                reporter.log_metrics(metrics)
                if mean_ret is not None:
                    log_line = (
                        f"update={upd+1:04d}  episodes={len(envs.episode_returns):06d}  "
                        f"ret50={mean_ret:8.2f}  len50={mean_len:6.1f}  gate_mean={metrics['diagnostics/gate_mean']:6.3f}  "
                        f"kl={algo_stats['approx_kl']:7.4f}  clipfrac={algo_stats['clipfrac']:6.3f}"
                    )
                    eval_ret = metrics.get("eval/ret_mean")
                    eval_len = metrics.get("eval/len_mean")
                    if (eval_ret is not None) and (eval_len is not None):
                        log_line += f"  eval_ret={eval_ret:8.2f}  eval_len={eval_len:6.1f}"
                    reporter.log_line(log_line)
                    if (upd + 1) % args.log_interval == 0:
                        print(log_line)

                if wandb_run is not None:
                    wandb_run.log(metrics, step=metrics["loop/frames"])
                final_metrics = metrics
            else:
                assert replay is not None
                assert dqn_target_ac is not None

                frames_before = upd * args.num_envs * args.horizon
                epsilon = linear_schedule(
                    step=frames_before,
                    start=args.dqn_eps_start,
                    end=args.dqn_eps_end,
                    decay_steps=args.dqn_eps_decay_steps,
                )
                obs, prev_action, traces, collect_stats = dqn_collect_rollout(
                    envs=envs,
                    ac=ac,
                    f_mem=f_mem,
                    drift=drift,
                    predictor=predictor,
                    replay=replay,
                    device=device,
                    horizon=args.horizon,
                    gamma=args.gamma,
                    lambda_pred=args.lambda_pred,
                    obs_normalization=args.obs_normalization,
                    alpha_base=alpha_base,
                    alpha_max=alpha_max,
                    reset_strategy=args.reset_strategy,
                    reset_long_fraction=args.reset_long_fraction,
                    obs=obs,
                    prev_action=prev_action,
                    traces=traces,
                    epsilon=epsilon,
                    action_generator=dqn_action_rng,
                )
                ema_update_(f_mem, ac.f_pol, tau=args.ema_tau)

                dqn_stats_hist = []
                ready = len(replay) >= max(args.dqn_batch_size, args.dqn_learning_starts)
                if ready:
                    for _ in range(max(args.dqn_updates_per_iter, 1)):
                        step_stats = dqn_update(
                            ac=ac,
                            target_ac=dqn_target_ac,
                            opt=opt,
                            replay=replay,
                            batch_size=args.dqn_batch_size,
                            gamma=args.gamma,
                            double_dqn=args.dqn_double,
                            generator=rng,
                            device=device,
                            use_amp=use_amp,
                            amp_dtype=amp_dtype,
                            grad_scaler=grad_scaler,
                            max_grad_norm=args.max_grad_norm,
                        )
                        if step_stats is None:
                            continue
                        dqn_stats_hist.append(step_stats)
                        dqn_opt_steps += 1
                        if dqn_opt_steps % max(args.dqn_target_update_interval, 1) == 0:
                            hard_update_(dqn_target_ac, ac)

                if dqn_stats_hist:
                    dqn_stats = {
                        k: sum(s[k] for s in dqn_stats_hist) / len(dqn_stats_hist) for k in dqn_stats_hist[0].keys()
                    }
                else:
                    dqn_stats = {
                        "q_loss": 0.0,
                        "q_mean": collect_stats["q_mean"],
                        "target_q_mean": 0.0,
                        "td_abs": 0.0,
                    }

                metrics = {
                    "loop/update": upd + 1,
                    "loop/frames": (upd + 1) * args.num_envs * args.horizon,
                    "diagnostics/gate_mean": (float(drift.state.gate.mean().item()) if drift is not None else 0.0),
                    "optim/lr": float(opt.param_groups[0]["lr"]),
                    "dqn/epsilon": collect_stats["epsilon"],
                    "dqn/replay_size": float(len(replay)),
                    "dqn/q_collect": collect_stats["q_mean"],
                }
                metrics.update({f"loss/{k}": v for k, v in dqn_stats.items()})
                add_perf_metrics(metrics, train_start=train_start, updates_done=upd + 1, updates_total=updates)

                mean_ret, mean_len = recent_return_stats(envs.episode_returns, window=50)
                if mean_ret is not None:
                    metrics["train/ret50"] = mean_ret
                    metrics["train/len50"] = mean_len

                do_eval = (eval_interval > 0) and ((upd + 1) % eval_interval == 0)
                if do_eval:
                    eval_metrics = evaluate_trace_policy(
                        args=args,
                        mask_indices=mask_indices,
                        ac=ac,
                        f_mem=f_mem,
                        predictor=predictor,
                        alpha_base=alpha_base,
                        alpha_max=alpha_max,
                        device=device,
                        eval_seed=eval_seed,
                        eval_num_envs=eval_num_envs,
                        eval_episodes=eval_episodes,
                        deterministic=eval_deterministic,
                    )
                    metrics.update(eval_metrics)
                    eval_ret = eval_metrics.get("eval/ret_mean")
                    eval_len = eval_metrics.get("eval/len_mean")
                    eval_line = (
                        f"eval update={upd+1:04d}  seed={eval_seed}  "
                        f"episodes={eval_metrics.get('eval/episodes_collected')}/{eval_metrics.get('eval/episodes_target')}"
                    )
                    if (eval_ret is not None) and (eval_len is not None):
                        eval_line += f"  ret_mean={eval_ret:8.2f}  len_mean={eval_len:6.1f}"
                    reporter.log_line(eval_line)

                should_stop, stop_reason = early_stopper.update(update_idx=upd + 1, metrics=metrics)
                if stop_reason:
                    reporter.log_line(stop_reason)
                    print(stop_reason)

                reporter.log_metrics(metrics)
                if mean_ret is not None:
                    log_line = (
                        f"update={upd+1:04d}  episodes={len(envs.episode_returns):06d}  "
                        f"ret50={mean_ret:8.2f}  len50={mean_len:6.1f}  gate_mean={metrics['diagnostics/gate_mean']:6.3f}  "
                        f"eps={metrics['dqn/epsilon']:5.3f}  qloss={dqn_stats['q_loss']:7.4f}  td={dqn_stats['td_abs']:7.4f}"
                    )
                    eval_ret = metrics.get("eval/ret_mean")
                    eval_len = metrics.get("eval/len_mean")
                    if (eval_ret is not None) and (eval_len is not None):
                        log_line += f"  eval_ret={eval_ret:8.2f}  eval_len={eval_len:6.1f}"
                    reporter.log_line(log_line)
                    if (upd + 1) % args.log_interval == 0:
                        print(log_line)

                if wandb_run is not None:
                    wandb_run.log(metrics, step=metrics["loop/frames"])
                final_metrics = metrics

            ret50_value = finite_metric(metrics.get("train/ret50"))
            if (ret50_value is not None) and ((best_ret50 is None) or (ret50_value > best_ret50)):
                best_ret50 = ret50_value
                reporter.save_checkpoint(_trace_checkpoint_payload(metrics), filename="checkpoint_best.pt")
            if lr_scheduler is not None:
                lr_scheduler.step()
            if progress is not None:
                progress.set_postfix(
                    {
                        "ret50": f"{metrics.get('train/ret50', float('nan')):.2f}",
                        "fps": f"{metrics['perf/fps']:.1f}",
                        "eta_min": f"{metrics['perf/eta_sec'] / 60.0:.1f}",
                    }
                )
                progress.update(1)
            profiler.step()
            if should_stop:
                break
        final_payload = _trace_checkpoint_payload(final_metrics)
        final_checkpoint = reporter.save_checkpoint(final_payload, filename="checkpoint.pt")
        reporter.save_checkpoint(final_payload, filename="checkpoint_last.pt")
    finally:
        profiler.close()
        if wandb_run is not None:
            wandb_run.finish()
        if progress is not None:
            progress.close()
    reporter.finalize(final_metrics, final_checkpoint)


if __name__ == "__main__":
    main()
