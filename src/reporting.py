from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_git_sha(repo_root: Path, *, short: bool = True) -> str | None:
    args = ["git", "rev-parse"]
    if short:
        args.append("--short")
    args.append("HEAD")
    try:
        result = subprocess.run(
            args,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _sanitize(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def extract_snippet(path: Path, start_pattern: str, num_lines: int) -> tuple[int, list[str]] | None:
    lines = path.read_text().splitlines()
    for idx, line in enumerate(lines):
        if start_pattern in line:
            start = idx + 1
            return start, lines[idx : idx + num_lines]
    return None


def format_snippet(path: Path, start_line: int, lines: list[str]) -> str:
    header = f"# {path.as_posix()}:{start_line}"
    body = "\n".join(lines)
    return "\n".join([header, body])


def _slugify(value: str | None, *, max_len: int = 40) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
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
    slug = "".join(out).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug


def _infer_regime_from_args(args: dict) -> str | None:
    mask_indices = args.get("mask_indices", [])
    if isinstance(mask_indices, str):
        mask_indices = [x for x in mask_indices.split(",") if x.strip()]
    partial_observable = len(mask_indices) > 0
    phase_len = int(args.get("phase_len", 0))
    obs_shift_scale = float(args.get("obs_shift_scale", 0.0))
    reward_scale_low = float(args.get("reward_scale_low", 1.0))
    reward_scale_high = float(args.get("reward_scale_high", 1.0))
    non_stationary = (phase_len > 0) and (
        (obs_shift_scale > 0.0) or (reward_scale_low != 1.0) or (reward_scale_high != 1.0)
    )
    if (not non_stationary) and (not partial_observable):
        return "stationary_fullobs"
    if (not non_stationary) and partial_observable:
        return "stationary_partialobs"
    if non_stationary and (not partial_observable):
        return "nonstationary_fullobs"
    return "nonstationary_partialobs"


def _build_active_args(args: dict) -> dict:
    common_keys = [
        "env_id",
        "seed",
        "device",
        "cuda_id",
        "policy",
        "algo",
        "encoder",
        "obs_normalization",
        "num_envs",
        "env_workers",
        "frame_stack",
        "horizon",
        "total_steps",
        "lr",
        "lr_schedule",
        "lr_end",
        "gamma",
        "gae_lam",
        "run_note",
        "run_postfix",
        "torch_profiler",
        "torch_profiler_dir",
        "torch_profiler_wait",
        "torch_profiler_warmup",
        "torch_profiler_active",
        "torch_profiler_repeat",
        "torch_profiler_skip_first",
        "torch_profiler_record_shapes",
        "torch_profiler_profile_memory",
        "torch_profiler_with_stack",
        "torch_profiler_with_flops",
        "torch_profiler_sort_by",
        "torch_profiler_row_limit",
    ]
    policy_keys = {
        "ff": ["hidden_dim", "feat_dim", "act_embed_dim"],
        "amt": [
            "hidden_dim",
            "feat_dim",
            "act_embed_dim",
            "alpha_base",
            "alpha_max",
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
            "reset_strategy",
            "reset_long_fraction",
            "lambda_pred",
            "pred_coef",
            "ema_tau",
        ],
        "recurrent": ["hidden_dim", "feat_dim", "act_embed_dim"],
    }
    algo_keys = {
        "ppo": ["clip_coef", "vf_clip", "target_kl", "vf_coef", "max_grad_norm", "ent_coef", "epochs", "minibatch_size"],
        "a2c": ["vf_coef", "max_grad_norm", "ent_coef", "epochs", "minibatch_size"],
        "trpo": [
            "vf_coef",
            "max_grad_norm",
            "ent_coef",
            "trpo_max_kl",
            "trpo_backtrack_coef",
            "trpo_backtrack_iters",
            "trpo_value_epochs",
        ],
        "reinforce": ["vf_coef", "max_grad_norm", "ent_coef", "epochs", "minibatch_size"],
        "v-trace": ["vf_coef", "max_grad_norm", "ent_coef", "epochs", "vtrace_rho_clip", "vtrace_c_clip"],
        "v-mpo": [
            "vf_coef",
            "max_grad_norm",
            "ent_coef",
            "epochs",
            "minibatch_size",
            "vmpo_topk_frac",
            "vmpo_eta",
            "vmpo_kl_coef",
            "vmpo_kl_target",
        ],
        "dqn": [
            "vf_coef",
            "max_grad_norm",
            "ent_coef",
            "dqn_replay_size",
            "dqn_batch_size",
            "dqn_learning_starts",
            "dqn_updates_per_iter",
            "dqn_target_update_interval",
            "dqn_double",
            "dqn_eps_start",
            "dqn_eps_end",
            "dqn_eps_decay_steps",
            "dqn_pin_memory",
        ],
    }

    policy = str(args.get("policy", "")).strip().lower()
    algo = str(args.get("algo", "")).strip().lower()
    env_id = str(args.get("env_id", "")).strip()

    selected = list(common_keys)
    selected.extend(policy_keys.get(policy, []))
    selected.extend(algo_keys.get(algo, []))

    if env_id.startswith("CarRacing"):
        selected.extend(["carracing_downsample", "carracing_grayscale"])

    selected.extend(["phase_len", "obs_shift_scale", "reward_scale_low", "reward_scale_high", "mask_indices"])
    selected.extend(
        [
            "wandb",
            "wandb_project",
            "wandb_entity",
            "wandb_run_name",
            "wandb_mode",
            "wandb_tags",
            "wandb_dir",
            "report",
            "report_dir",
            "report_run_name",
            "debug_log",
            "eval_interval",
            "eval_episodes",
            "eval_num_envs",
            "eval_seed",
            "eval_seed_offset",
            "eval_stochastic",
            "early_stop_metric",
            "early_stop_mode",
            "early_stop_patience",
            "early_stop_min_delta",
            "early_stop_warmup_updates",
            "amp",
            "amp_dtype",
            "compile",
            "compile_mode",
            "tf32",
            "adam_foreach",
            "adam_fused",
            "log_interval",
            "no_tqdm",
        ]
    )

    active = {}
    seen = set()
    for key in selected:
        if key in seen:
            continue
        seen.add(key)
        if key in args:
            active[key] = args.get(key)
    return active


def build_run_name(args: dict, run_name: str | None) -> str:
    if run_name:
        return run_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    env_slug = _slugify(args.get("env_id", "env"), max_len=24)
    policy_slug = _slugify(args.get("policy", "policy"), max_len=16)
    algo_slug = _slugify(args.get("algo", "algo"), max_len=16)
    seed = args.get("seed")
    seed_slug = f"s{int(seed)}" if isinstance(seed, (int, np.integer)) else ""
    regime_slug = _slugify(_infer_regime_from_args(args), max_len=28)
    note_slug = _slugify(args.get("run_note"), max_len=24)
    parts = [stamp, env_slug, regime_slug, policy_slug, algo_slug, seed_slug, note_slug]
    parts = [p for p in parts if p]
    return "_".join(parts)


def _safe_cuda_index(device: str) -> int | None:
    try:
        device_obj = torch.device(device)
    except (TypeError, ValueError):
        return None
    if device_obj.type != "cuda":
        return None
    if not torch.cuda.is_available():
        return None
    if device_obj.index is not None:
        return int(device_obj.index)
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return None


@dataclass
class RunReporter:
    run_dir: Path
    summary_path: Path
    metrics_path: Path
    metrics_csv_path: Path
    logs_path: Path
    summary: dict
    start_time: float
    enabled: bool = True
    last_metrics: dict | None = None
    last_log_line: str | None = None
    metric_rows: int = 0
    best_ret50: float | None = None
    second_best_ret50: float | None = None
    best_ret50_frames: float | None = None
    last_ret50: float | None = None

    def log_metrics(self, metrics: dict) -> None:
        if not self.enabled:
            return
        metrics_clean = _sanitize(metrics)
        self.last_metrics = metrics_clean
        row = {"time/elapsed_sec": round(time.time() - self.start_time, 6)}
        row.update(metrics_clean)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        self.metric_rows += 1
        ret50 = row.get("train/ret50")
        if isinstance(ret50, (int, float)) and math.isfinite(ret50):
            ret50 = float(ret50)
            self.last_ret50 = ret50
            if (self.best_ret50 is None) or (ret50 > self.best_ret50):
                self.second_best_ret50 = self.best_ret50
                self.best_ret50 = float(ret50)
                frames = row.get("loop/frames")
                self.best_ret50_frames = float(frames) if isinstance(frames, (int, float)) else None
            elif (self.second_best_ret50 is None) or (ret50 > self.second_best_ret50):
                self.second_best_ret50 = ret50

    def log_line(self, line: str) -> None:
        if not self.enabled:
            return
        self.last_log_line = line
        with self.logs_path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

    def log_block(self, title: str, payload: dict | str, *, prefix: str = "# ") -> None:
        if not self.enabled:
            return
        header = f"{prefix}=== {title} ==="
        footer = f"{prefix}=== end {title} ==="
        if isinstance(payload, dict):
            payload_text = json.dumps(_sanitize(payload), indent=2, sort_keys=True)
        else:
            payload_text = str(payload)
        lines = payload_text.splitlines() or [""]
        with self.logs_path.open("a", encoding="utf-8") as f:
            f.write(header.rstrip() + "\n")
            for line in lines:
                f.write(f"{prefix}{line}".rstrip() + "\n")
            f.write(footer.rstrip() + "\n")

    def save_checkpoint(self, payload: dict, filename: str = "checkpoint.pt") -> Path | None:
        if not self.enabled:
            return None
        ckpt_path = self.run_dir / filename
        torch.save(payload, ckpt_path)
        return ckpt_path

    def finalize(self, metrics: dict | None = None, checkpoint_path: Path | None = None) -> None:
        if not self.enabled:
            return
        if metrics is None:
            metrics = self.last_metrics
        if metrics is not None:
            self.summary["final_metrics"] = _sanitize(metrics)
        if self.last_log_line:
            self.summary["final_log_line"] = self.last_log_line
        if checkpoint_path is not None:
            self.summary["checkpoint"] = str(checkpoint_path)
        self.summary["runtime_sec"] = round(time.time() - self.start_time, 3)
        self._export_metrics_csv()
        self.summary["artifacts"] = {
            "metrics_jsonl": str(self.metrics_path),
            "metrics_csv": str(self.metrics_csv_path),
            "logs_txt": str(self.logs_path),
        }
        self.summary["training_stats"] = {
            "metric_rows": int(self.metric_rows),
            "best_ret50": self.best_ret50,
            "second_best_ret50": self.second_best_ret50,
            "best_ret50_frames": self.best_ret50_frames,
            "last_ret50": self.last_ret50,
        }
        self._write_summary()

    def update_summary(self, updates: dict) -> None:
        if not self.enabled:
            return
        self.summary.update(_sanitize(updates))
        self._write_summary()

    def _write_summary(self) -> None:
        self.summary_path.write_text(json.dumps(self.summary, indent=2, sort_keys=True))

    def _export_metrics_csv(self) -> None:
        fieldnames: list[str] = []
        rows: list[dict] = []
        seen = set()
        with self.metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                rows.append(row)
                for key in row.keys():
                    if key in seen:
                        continue
                    seen.add(key)
                    fieldnames.append(key)

        with self.metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
            if not fieldnames:
                f.write("")
                return
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                serializable_row = {}
                for key in fieldnames:
                    value = row.get(key)
                    if isinstance(value, (dict, list)):
                        serializable_row[key] = json.dumps(value, sort_keys=True)
                    else:
                        serializable_row[key] = value
                writer.writerow(serializable_row)


def start_run_report(
    *,
    repo_root: Path,
    report_dir: Path,
    run_name: str | None,
    args: dict,
    device: str,
    obs_dim: int,
    act_dim: int,
    mask_indices: list[int],
    config_path: str | None,
    enabled: bool = True,
) -> RunReporter:
    if not enabled:
        return RunReporter(
            run_dir=Path("."),
            summary_path=Path("."),
            metrics_path=Path("."),
            metrics_csv_path=Path("."),
            logs_path=Path("."),
            summary={},
            start_time=time.time(),
            enabled=False,
        )

    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = build_run_name(args=args, run_name=run_name)
    run_dir = report_dir / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Use a different --report-run-name or --run-postfix."
        )
    run_dir.mkdir(parents=True, exist_ok=False)

    ram_gb = None
    try:
        if hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            ram_gb = round((page_size * pages) / (1024**3), 2)
    except (OSError, ValueError):
        ram_gb = None

    system = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "ram_gb": ram_gb,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_index": _safe_cuda_index(device),
        "gpu_name": None,
        "gpu_total_memory_gb": None,
        "gpu_compute_capability": None,
        "gpu_multi_processor_count": None,
        "device": device,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "hostname": platform.node() or None,
    }

    cuda_index = system.get("cuda_device_index")
    if isinstance(cuda_index, int):
        try:
            system["gpu_name"] = torch.cuda.get_device_name(cuda_index)
            props = torch.cuda.get_device_properties(cuda_index)
            system["gpu_total_memory_gb"] = round(float(props.total_memory) / (1024**3), 3)
            system["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            system["gpu_multi_processor_count"] = int(props.multi_processor_count)
        except Exception:
            pass

    command = "python3 amg.py " + " ".join(sys.argv[1:])
    run_identity = {
        "env_id": args.get("env_id"),
        "policy": args.get("policy"),
        "algorithm": args.get("algo"),
        "seed": args.get("seed"),
        "regime": _infer_regime_from_args(args),
    }
    active_args = _sanitize(_build_active_args(args))
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "timestamp": _utc_now(),
        "command": command.strip(),
        "run_note": args.get("run_note"),
        "run_identity": run_identity,
        "config_path": config_path,
        "git_sha": _safe_git_sha(repo_root, short=True),
        "git_commit_hash": _safe_git_sha(repo_root, short=False),
        # Keep only run-relevant arguments in summary to avoid noisy defaults
        # from inactive algorithms/policies.
        "args": active_args,
        # Backward-compatible alias for downstream scripts.
        "active_args": active_args,
        "env": {
            "env_id": args.get("env_id"),
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "mask_indices": mask_indices,
        },
        "system": system,
    }

    summary_path = run_dir / "run_summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    metrics_csv_path = run_dir / "metrics.csv"
    logs_path = run_dir / "train.log"
    metrics_path.write_text("", encoding="utf-8")
    metrics_csv_path.write_text("", encoding="utf-8")
    logs_path.write_text("", encoding="utf-8")
    reporter = RunReporter(
        run_dir=run_dir,
        summary_path=summary_path,
        metrics_path=metrics_path,
        metrics_csv_path=metrics_csv_path,
        logs_path=logs_path,
        summary=summary,
        start_time=time.time(),
    )
    reporter._write_summary()
    reporter.log_line("# AMT run log")
    reporter.log_line(f"# run_id: {run_id}")
    reporter.log_line(f"# timestamp_utc: {summary.get('timestamp')}")
    reporter.log_line(f"# run_dir: {run_dir.as_posix()}")
    reporter.log_line(f"# git_sha: {summary.get('git_sha')}")
    reporter.log_line(f"# git_commit_hash: {summary.get('git_commit_hash')}")
    reporter.log_line(f"# command: {summary.get('command')}")
    reporter.log_line(f"# run_note: {summary.get('run_note')}")
    reporter.log_line(f"# config_path: {summary.get('config_path')}")
    reporter.log_block("run_identity", summary.get("run_identity", {}))
    reporter.log_block("system", system)
    reporter.log_block("env", summary.get("env", {}))
    reporter.log_block("args", summary.get("args", {}))
    return reporter
