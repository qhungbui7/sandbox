#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


BENCHMARK_ROOT = Path("configs/benchmarks")
DEFAULT_ENVS = ["cartpole", "acrobot", "mountaincar"]
DEFAULT_REGIMES = [
    "stationary_fullobs",
    "stationary_partialobs",
    "nonstationary_fullobs",
    "nonstationary_partialobs",
]
DEFAULT_ALGOS = ["ppo", "a2c", "trpo", "reinforce", "v-trace", "v-mpo", "dqn"]
ALL_ALGOS = set(DEFAULT_ALGOS)


def parse_csv(value: str, default: list[str]) -> list[str]:
    if not value.strip():
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def find_config(env_key: str, regime: str) -> Path:
    cfg = BENCHMARK_ROOT / env_key / f"{regime}.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing benchmark config: {cfg}")
    return cfg


def infer_env_regime_from_config(cfg: Path) -> tuple[str, str]:
    env_key = cfg.parent.name.strip().lower() or "env"
    regime = cfg.stem.strip().lower() or "regime"
    return env_key, regime


def resolve_config_specs(*, envs: list[str], regimes: list[str], config_paths: list[str]) -> list[tuple[str, str, Path]]:
    if config_paths:
        specs: list[tuple[str, str, Path]] = []
        for raw in config_paths:
            cfg = Path(raw)
            if not cfg.exists():
                raise FileNotFoundError(f"Config path not found: {cfg}")
            env_key, regime = infer_env_regime_from_config(cfg)
            specs.append((env_key, regime, cfg))
        return specs

    specs = []
    for env_key in envs:
        for regime in regimes:
            cfg = find_config(env_key, regime)
            specs.append((env_key, regime, cfg))
    return specs


def run_command(cmd: list[str], dry_run: bool) -> int:
    print(">>>", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def load_report_metrics(report_dir: Path, run_name: str) -> dict:
    summary_path = report_dir / run_name / "run_summary.json"
    if not summary_path.exists():
        return {}
    data = json.loads(summary_path.read_text())
    final_metrics = data.get("final_metrics", {}) or {}
    training_stats = data.get("training_stats", {}) if isinstance(data.get("training_stats"), dict) else {}
    if training_stats:
        final_metrics = dict(final_metrics)
        final_metrics["train/best_ret50"] = training_stats.get("best_ret50")
        final_metrics["train/second_best_ret50"] = training_stats.get("second_best_ret50")
        final_metrics["train/last_ret50"] = training_stats.get("last_ret50")
    return final_metrics


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean_std_ci95(values: list[float]) -> dict:
    clean = [float(v) for v in values if is_number(v)]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    mean = sum(clean) / n
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95": 0.0}
    var = sum((x - mean) ** 2 for x in clean) / (n - 1)
    std = math.sqrt(max(var, 0.0))
    ci95 = 1.96 * std / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "ci95": ci95}


def contains_training_override(extra_args: list[str]) -> bool:
    tracked = {"--total-steps", "--num-envs", "--horizon"}
    return any(token in tracked for token in extra_args)


def has_arg(extra_args: list[str], name: str) -> bool:
    return any(token == name or token.startswith(f"{name}=") for token in extra_args)


def sanitize_postfix(value: str) -> str:
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


def build_run_note(base_note: str, env_key: str, regime: str, algo: str, seed: int, cfg: Path) -> str:
    return (
        f"{base_note.strip()} | "
        f"env={env_key} regime={regime} algo={algo} seed={seed} cfg={cfg.as_posix()}"
    )


def run_postprocess(
    *,
    run_name: str,
    report_dir: Path,
    plot_dir: Path | None,
    gameplay_dir: Path | None,
    gameplay_episodes: int,
    device: str,
    cuda_id: int | None,
    dry_run: bool,
) -> None:
    run_dir = report_dir / run_name
    resolved_plot_dir = plot_dir if plot_dir is not None else (run_dir / "plots")
    resolved_gameplay_dir = gameplay_dir if gameplay_dir is not None else (run_dir / "gameplay")
    if not dry_run:
        resolved_plot_dir.mkdir(parents=True, exist_ok=True)
        resolved_gameplay_dir.mkdir(parents=True, exist_ok=True)

    plot_cmd = [
        sys.executable,
        "scripts/plot_training.py",
        "--report-dir",
        str(report_dir),
        "--run-names",
        run_name,
        "--output-dir",
        str(resolved_plot_dir),
    ]
    code = run_command(plot_cmd, dry_run=dry_run)
    if code != 0:
        raise SystemExit(code)

    gameplay_cmd = [
        sys.executable,
        "scripts/record_gameplay.py",
        "--run-dir",
        str(run_dir),
        "--episodes",
        str(gameplay_episodes),
        "--deterministic",
        "--output-dir",
        str(resolved_gameplay_dir),
        "--name",
        run_name,
        "--overwrite",
        "--device",
        device,
    ]
    if str(device).startswith("cuda"):
        assert cuda_id is not None
        gameplay_cmd.extend(["--cuda-id", str(cuda_id)])
    code = run_command(gameplay_cmd, dry_run=dry_run)
    if code != 0:
        raise SystemExit(code)


def main() -> None:
    p = argparse.ArgumentParser(description="Run benchmark matrix across environment/regime presets.")
    p.add_argument("--algo", type=str, default=None, help="Optional single algorithm override.")
    p.add_argument(
        "--algos",
        type=str,
        default=",".join(DEFAULT_ALGOS),
        help="Comma-separated algorithms. Default runs every model: ppo,a2c,trpo,reinforce,v-trace,v-mpo,dqn",
    )
    p.add_argument(
        "--envs",
        type=str,
        default=",".join(DEFAULT_ENVS),
        help="Comma-separated env keys (cartpole, acrobot, mountaincar, carracing).",
    )
    p.add_argument(
        "--regimes",
        type=str,
        default=",".join(DEFAULT_REGIMES),
        help="Comma-separated regimes (stationary_fullobs, stationary_partialobs, nonstationary_fullobs, nonstationary_partialobs).",
    )
    p.add_argument(
        "--config-paths",
        type=str,
        default="",
        help="Optional comma-separated config YAML paths to run directly (overrides --envs/--regimes).",
    )
    p.add_argument("--seeds", type=str, default="0", help="Comma-separated random seeds.")
    p.add_argument("--run-note", type=str, required=True, help="Required base note. Script appends env/regime/model/seed.")
    p.add_argument("--run-postfix", type=str, default="", help="Optional postfix appended to run names.")
    p.add_argument("--device", type=str, default="cuda", help="Device override (default: cuda).")
    p.add_argument("--cuda-id", type=int, default=None, help="CUDA device index for training (required when --device cuda).")
    p.add_argument("--total-steps", type=int, default=None, help="Optional total_steps override.")
    p.add_argument("--num-envs", type=int, default=None, help="Optional num_envs override.")
    p.add_argument("--horizon", type=int, default=None, help="Optional horizon override.")
    p.add_argument("--frame-stack", type=int, default=None, help="Optional frame stack override (recommended with --encoder cnn).")
    p.add_argument(
        "--encoder",
        type=str,
        default=None,
        choices=["mlp", "cnn"],
        help="Model encoder override passed to amg.py.",
    )
    p.add_argument(
        "--obs-normalization",
        type=str,
        default=None,
        choices=["auto", "none", "uint8"],
        help="Observation normalization override. Default: uint8 for CarRacing, auto otherwise.",
    )
    p.add_argument("--report-dir", type=str, default="reports/benchmarks")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    p.add_argument("--wandb-project", type=str, default="amt")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-dir", type=str, default="wandb")
    p.add_argument(
        "--extra-args",
        type=str,
        default="",
        help='Extra raw CLI args appended to each amg.py command. Example: "--dqn-learning-starts 512 --dqn-batch-size 64"',
    )
    p.add_argument(
        "--require-full-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disallow rollout-size overrides so each run follows full config training.",
    )
    p.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After each run, generate plots and gameplay videos for inspection.",
    )
    p.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Optional shared plot output dir override. Default is per-run: <report-dir>/<run-name>/plots.",
    )
    p.add_argument(
        "--gameplay-dir",
        type=str,
        default=None,
        help="Optional shared gameplay output dir override. Default is per-run: <report-dir>/<run-name>/gameplay.",
    )
    p.add_argument("--analysis-dir", type=str, default=None, help="Summary output dir (default: <report-dir>/analysis).")
    p.add_argument("--gameplay-episodes", type=int, default=2, help="Episodes recorded per run when --visualize is enabled.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()

    if str(args.device).startswith("cuda") and args.cuda_id is None:
        raise ValueError("--cuda-id is required when --device is CUDA.")
    if not str(args.run_note).strip():
        raise ValueError("--run-note must be a non-empty string.")
    run_postfix = sanitize_postfix(args.run_postfix)

    envs = parse_csv(args.envs, DEFAULT_ENVS)
    regimes = parse_csv(args.regimes, DEFAULT_REGIMES)
    config_paths = parse_csv(args.config_paths, [])
    if args.algo:
        algos = [args.algo.strip().lower().replace("_", "-")]
    else:
        algos = [a.strip().lower().replace("_", "-") for a in parse_csv(args.algos, DEFAULT_ALGOS)]
    invalid_algos = sorted({a for a in algos if a not in ALL_ALGOS})
    if invalid_algos:
        raise ValueError(f"Unsupported --algos entries: {invalid_algos}. Choose from: {', '.join(sorted(ALL_ALGOS))}")
    seeds = [int(x) for x in parse_csv(args.seeds, ["0"])]
    report_dir = Path(args.report_dir)
    extra_args = shlex.split(args.extra_args) if args.extra_args else []
    plot_dir = Path(args.plot_dir) if args.plot_dir else None
    gameplay_dir = Path(args.gameplay_dir) if args.gameplay_dir else None
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else (report_dir / "analysis")

    if args.require_full_training:
        if args.total_steps is not None or args.num_envs is not None or args.horizon is not None:
            raise ValueError(
                "Full training is enforced: remove --total-steps/--num-envs/--horizon or pass --no-require-full-training."
            )
        if contains_training_override(extra_args):
            raise ValueError(
                "Full training is enforced: remove training-size overrides from --extra-args "
                "or pass --no-require-full-training."
            )

    config_specs = resolve_config_specs(envs=envs, regimes=regimes, config_paths=config_paths)

    results = []
    for env_key, regime, cfg in config_specs:
        for algo in algos:
            for seed in seeds:
                run_name = f"{env_key}_{regime}_{algo}_s{seed}"
                if run_postfix:
                    run_name = f"{run_name}_{run_postfix}"
                run_note = build_run_note(
                    base_note=args.run_note,
                    env_key=env_key,
                    regime=regime,
                    algo=algo,
                    seed=seed,
                    cfg=cfg,
                )
                cmd = [
                    sys.executable,
                    "amg.py",
                    str(cfg),
                    "--algo",
                    algo,
                    "--seed",
                    str(seed),
                    "--run-note",
                    run_note,
                    "--report",
                    "--report-dir",
                    str(report_dir),
                    "--report-run-name",
                    run_name,
                    "--device",
                    args.device,
                ]
                if str(args.device).startswith("cuda"):
                    cmd.extend(["--cuda-id", str(args.cuda_id)])
                if args.total_steps is not None:
                    cmd.extend(["--total-steps", str(args.total_steps)])
                if args.num_envs is not None:
                    cmd.extend(["--num-envs", str(args.num_envs)])
                if args.horizon is not None:
                    cmd.extend(["--horizon", str(args.horizon)])
                if args.frame_stack is not None:
                    cmd.extend(["--frame-stack", str(args.frame_stack)])
                if args.wandb:
                    cmd.extend(["--wandb", "--wandb-project", args.wandb_project])
                    if args.wandb_entity:
                        cmd.extend(["--wandb-entity", args.wandb_entity])
                    cmd.extend(["--wandb-run-name", run_name])
                    cmd.extend(["--wandb-mode", args.wandb_mode])
                    cmd.extend(["--wandb-dir", args.wandb_dir])
                else:
                    cmd.extend(["--wandb-mode", "disabled"])
                if extra_args:
                    cmd.extend(extra_args)
                if args.encoder is not None:
                    cmd.extend(["--encoder", str(args.encoder)])

                obs_normalization = args.obs_normalization
                if obs_normalization is None:
                    obs_normalization = "uint8" if env_key == "carracing" else "auto"
                if (not has_arg(extra_args, "--obs-normalization")) or (args.obs_normalization is not None):
                    cmd.extend(["--obs-normalization", str(obs_normalization)])

                code = run_command(cmd, dry_run=args.dry_run)
                if code != 0:
                    print(f"[error] run failed (exit={code}) for {run_name}")
                    if not args.continue_on_error:
                        raise SystemExit(code)
                    continue

                if args.visualize:
                    try:
                        run_postprocess(
                            run_name=run_name,
                            report_dir=report_dir,
                            plot_dir=plot_dir,
                            gameplay_dir=gameplay_dir,
                            gameplay_episodes=int(args.gameplay_episodes),
                            device=args.device,
                            cuda_id=args.cuda_id,
                            dry_run=args.dry_run,
                        )
                    except SystemExit as exc:
                        print(f"[error] post-processing failed for {run_name}: {exc}")
                        if not args.continue_on_error:
                            raise
                        continue

                metrics = load_report_metrics(report_dir=report_dir, run_name=run_name)
                results.append(
                    {
                        "run_name": run_name,
                        "env": env_key,
                        "regime": regime,
                        "algo": algo,
                        "seed": seed,
                        "best_ret50": metrics.get("train/best_ret50"),
                        "second_best_ret50": metrics.get("train/second_best_ret50"),
                        "last_ret50": metrics.get("train/last_ret50"),
                        "ret50": metrics.get("train/ret50"),
                        "len50": metrics.get("train/len50"),
                        "frames": metrics.get("loop/frames"),
                    }
                )

    if not args.dry_run:
        summary_cmd = [
            sys.executable,
            "scripts/summarize_benchmarks.py",
            "--report-dir",
            str(report_dir),
            "--output-dir",
            str(analysis_dir),
        ]
        code = run_command(summary_cmd, dry_run=False)
        if code != 0 and not args.continue_on_error:
            raise SystemExit(code)

    if not args.dry_run and results:
        print("\n=== Per-Run Results ===")
        print("run_name,env,regime,algo,seed,best_ret50,second_best_ret50,last_ret50,ret50,len50,frames")
        for row in results:
            print(
                f"{row['run_name']},{row['env']},{row['regime']},{row['algo']},{row['seed']},"
                f"{row['best_ret50']},{row['second_best_ret50']},{row['last_ret50']},"
                f"{row['ret50']},{row['len50']},{row['frames']}",
            )

        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in results:
            grouped[(row["env"], row["regime"], row["algo"])].append(row)

        print("\n=== Aggregated Results (mean/std/95% CI) ===")
        print(
            "env,regime,algo,n,ret50_mean,ret50_std,ret50_ci95,len50_mean,len50_std,len50_ci95,frames_mean,frames_std,frames_ci95"
        )
        for key in sorted(grouped):
            rows = grouped[key]
            ret_stats = mean_std_ci95([r["ret50"] for r in rows if is_number(r["ret50"])])
            len_stats = mean_std_ci95([r["len50"] for r in rows if is_number(r["len50"])])
            frame_stats = mean_std_ci95([r["frames"] for r in rows if is_number(r["frames"])])

            def fmt(stats: dict, name: str) -> str:
                value = stats.get(name)
                if value is None:
                    return "None"
                return f"{value:.6f}"

            n = max(ret_stats["n"], len_stats["n"], frame_stats["n"])
            env_key, regime, algo = key
            print(
                f"{env_key},{regime},{algo},{n},"
                f"{fmt(ret_stats, 'mean')},{fmt(ret_stats, 'std')},{fmt(ret_stats, 'ci95')},"
                f"{fmt(len_stats, 'mean')},{fmt(len_stats, 'std')},{fmt(len_stats, 'ci95')},"
                f"{fmt(frame_stats, 'mean')},{fmt(frame_stats, 'std')},{fmt(frame_stats, 'ci95')}"
            )


if __name__ == "__main__":
    main()
