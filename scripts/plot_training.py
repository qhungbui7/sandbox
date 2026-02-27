#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _discover_run_dirs(report_dir: Path, run_names: list[str] | None) -> list[Path]:
    if run_names:
        dirs = [report_dir / name for name in run_names]
        return [d for d in dirs if d.exists()]
    return sorted([p for p in report_dir.iterdir() if p.is_dir() and (p / "run_summary.json").exists()])


def _series(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=np.float64)


def _get_metrics_path(run_dir: Path, summary: dict) -> Path:
    artifacts = summary.get("artifacts", {}) if isinstance(summary, dict) else {}
    candidate = artifacts.get("metrics_jsonl")
    if isinstance(candidate, str) and candidate:
        p = Path(candidate)
        if p.exists():
            return p
    return run_dir / "metrics.jsonl"


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for i in range(values.shape[0]):
        start = max(0, i - window + 1)
        seg = values[start : i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size > 0:
            out[i] = float(seg.mean())
    return out


def _load_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_name = summary.get("run_id", run_dir.name)
    metrics_path = _get_metrics_path(run_dir, summary)
    rows = _load_jsonl(metrics_path)
    if not rows:
        return None
    frames = _series(rows, "loop/frames")
    ret50 = _series(rows, "train/ret50")
    len50 = _series(rows, "train/len50")
    approx_kl = _series(rows, "loss/approx_kl")
    clipfrac = _series(rows, "loss/clipfrac")
    value_loss = _series(rows, "loss/value_loss")
    entropy = _series(rows, "loss/entropy")
    explained_variance = _series(rows, "debug/value/explained_variance")

    hist = np.asarray([[float(row.get(f"debug/action/hist_{idx}", float("nan"))) for idx in range(7)] for row in rows], dtype=np.float64)
    probs = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1e-12)
    action_dom = probs.max(axis=1)
    action_entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)

    kl_abs = np.abs(approx_kl)
    ret50_smooth = _rolling_mean(ret50, window=10)
    collapse_flag = (entropy < 1.2) & (action_dom > 0.6)
    kl_event = kl_abs > 0.02
    clip_event = clipfrac > 0.2
    value_loss_p95 = float(np.quantile(value_loss, 0.95))
    v_event = value_loss > value_loss_p95

    return {
        "run_dir": run_dir,
        "summary": summary,
        "run_name": run_name,
        "frames": frames,
        "ret50": ret50,
        "ret50_smooth": ret50_smooth,
        "len50": len50,
        "kl_abs": kl_abs,
        "clipfrac": clipfrac,
        "value_loss": value_loss,
        "entropy": entropy,
        "action_dom": action_dom,
        "action_entropy": action_entropy,
        "explained_variance": explained_variance,
        "collapse_flag": collapse_flag,
        "kl_event": kl_event,
        "clip_event": clip_event,
        "v_event": v_event,
    }


def _plot_single_run(run_data: dict, output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required for plotting. Install with `pip install matplotlib`.")

    summary = run_data["summary"]
    run_name = run_data["run_name"]
    frames = run_data["frames"]
    ret50 = run_data["ret50"]
    ret50_smooth = run_data["ret50_smooth"]
    len50 = run_data["len50"]
    kl_abs = run_data["kl_abs"]
    clipfrac = run_data["clipfrac"]
    entropy = run_data["entropy"]
    action_dom = run_data["action_dom"]
    action_entropy = run_data["action_entropy"]
    value_loss = run_data["value_loss"]
    explained_variance = run_data["explained_variance"]
    event_any = run_data["kl_event"] | run_data["clip_event"] | run_data["v_event"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_ret, ax_klclip, ax_action, ax_value = axes.flatten()

    ax_ret_len = ax_ret.twinx()
    ax_ret.plot(frames, ret50, label="ret50", color="tab:blue", alpha=0.35)
    ax_ret.plot(frames, ret50_smooth, label="ret50_smooth(w=10)", color="tab:blue", linewidth=1.8)
    ax_ret_len.plot(frames, len50, label="len50", color="tab:orange", alpha=0.7)
    for x in frames[event_any]:
        ax_ret.axvline(x=x, color="tab:red", alpha=0.1, linewidth=0.8)
    ax_ret.set_title("Returns / Length + Events")
    ax_ret.set_xlabel("Frames")
    ax_ret.set_ylabel("ret50")
    ax_ret_len.set_ylabel("len50")
    h1, l1 = ax_ret.get_legend_handles_labels()
    h2, l2 = ax_ret_len.get_legend_handles_labels()
    ax_ret.legend(h1 + h2, l1 + l2, loc="best")
    ax_ret.grid(alpha=0.25)

    ax_clip = ax_klclip.twinx()
    ax_klclip.plot(frames, kl_abs, label="kl_abs", color="tab:purple")
    ax_clip.plot(frames, clipfrac, label="clipfrac", color="tab:red")
    for thr in [0.01, 0.02]:
        ax_klclip.axhline(y=thr, color="tab:purple", linestyle="--", alpha=0.45, linewidth=0.9)
    for thr in [0.2, 0.4]:
        ax_clip.axhline(y=thr, color="tab:red", linestyle="--", alpha=0.45, linewidth=0.9)
    ax_klclip.set_title("KL(abs) / Clipfrac")
    ax_klclip.set_xlabel("Frames")
    ax_klclip.set_ylabel("kl_abs")
    ax_clip.set_ylabel("clipfrac")
    h1, l1 = ax_klclip.get_legend_handles_labels()
    h2, l2 = ax_clip.get_legend_handles_labels()
    ax_klclip.legend(h1 + h2, l1 + l2, loc="best")
    ax_klclip.grid(alpha=0.25)

    ax_action.plot(frames, entropy, label="entropy", color="tab:green")
    ax_action.plot(frames, action_dom, label="action_dom", color="tab:orange")
    ax_action.plot(frames, action_entropy, label="action_entropy", color="tab:brown")
    ax_action.set_title("Entropy / Action Dominance")
    ax_action.set_xlabel("Frames")
    ax_action.legend(loc="best")
    ax_action.grid(alpha=0.25)

    ax_ev = ax_value.twinx()
    ax_value.plot(frames, np.log1p(np.maximum(value_loss, 0.0)), label="log1p(value_loss)", color="tab:red")
    ax_ev.plot(frames, explained_variance, label="explained_variance", color="tab:blue")
    ax_ev.axhline(y=0.0, color="tab:blue", linestyle="--", alpha=0.45, linewidth=0.9)
    ax_value.set_title("Value Loss / Explained Variance")
    ax_value.set_xlabel("Frames")
    ax_value.set_ylabel("log1p(value_loss)")
    ax_ev.set_ylabel("explained_variance")
    h1, l1 = ax_value.get_legend_handles_labels()
    h2, l2 = ax_ev.get_legend_handles_labels()
    ax_value.legend(h1 + h2, l1 + l2, loc="best")
    ax_value.grid(alpha=0.25)

    fig.suptitle(
        f"{run_name}\n{summary.get('env', {}).get('env_id', 'env')} | "
        f"{summary.get('model', {}).get('policy', '?')}+{summary.get('model', {}).get('algorithm', '?')}"
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / f"{run_name}_dashboard.png"
    compat_path = output_dir / f"{run_name}_training.png"
    fig.savefig(dashboard_path, dpi=150)
    fig.savefig(compat_path, dpi=150)
    plt.close(fig)
    return [dashboard_path, compat_path]


def _get_group_value(summary: dict, group_by: str) -> str:
    cur = summary
    for key in group_by.split("."):
        cur = cur[key]
    if isinstance(cur, (str, int, float, bool)):
        return str(cur)
    return json.dumps(cur, sort_keys=True)


def _stack_group_curves(group_runs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(int(run["ret50_smooth"].shape[0]) for run in group_runs)
    y_stack = np.full((len(group_runs), max_len), np.nan, dtype=np.float64)
    x_stack = np.full((len(group_runs), max_len), np.nan, dtype=np.float64)
    for i, run in enumerate(group_runs):
        n = int(run["ret50_smooth"].shape[0])
        y_stack[i, :n] = run["ret50_smooth"]
        x_stack[i, :n] = run["frames"]
    x_med = np.nanmedian(x_stack, axis=0)
    y_mean = np.nanmean(y_stack, axis=0)
    y_q25 = np.nanquantile(y_stack, 0.25, axis=0)
    y_q75 = np.nanquantile(y_stack, 0.75, axis=0)
    valid = np.isfinite(x_med) & np.isfinite(y_mean)
    return x_med[valid], y_mean[valid], y_q25[valid], y_q75[valid]


def _plot_compare_ret50(run_data: list[dict], output_dir: Path, group_by: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required for plotting. Install with `pip install matplotlib`.")

    if not run_data:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    plotted = 0
    if group_by:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for run in run_data:
            grouped[_get_group_value(run["summary"], group_by)].append(run)
        for group_key in sorted(grouped):
            xs, mean, q25, q75 = _stack_group_curves(grouped[group_key])
            if xs.size == 0:
                continue
            ax.plot(xs, mean, linewidth=2.0, label=f"{group_key} (mean)")
            ax.fill_between(xs, q25, q75, alpha=0.2)
            plotted += 1
    else:
        for run in run_data:
            xs = run["frames"]
            ys = run["ret50_smooth"]
            label = (
                f"{run['run_name']} | "
                f"{run['summary'].get('env', {}).get('env_id', 'env')} | "
                f"{run['summary'].get('model', {}).get('algorithm', '?')}"
            )
            ax.plot(xs, ys, label=label)
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    title = "ret50_smooth comparison"
    if group_by:
        title += f" (group_by={group_by})"
    ax.set_title(title)
    ax.set_xlabel("Frames")
    ax.set_ylabel("ret50_smooth(w=10)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comparison_ret50_smooth.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Plot training curves from report metrics.jsonl files.")
    p.add_argument("--report-dir", type=str, default="reports", help="Root directory containing run folders.")
    p.add_argument("--run-names", type=str, default="", help="Optional comma-separated run folder names.")
    p.add_argument("--output-dir", type=str, default="reports/plots", help="Where PNG plots are written.")
    p.add_argument(
        "--group-by",
        type=str,
        default="",
        help="Optional summary key for grouped overlay with mean/quantile bands (supports dotted keys, e.g. config_path).",
    )
    args = p.parse_args()

    run_names = [x.strip() for x in args.run_names.split(",") if x.strip()] or None
    run_dirs = _discover_run_dirs(Path(args.report_dir), run_names=run_names)
    if not run_dirs:
        raise SystemExit(f"No runs found in {args.report_dir}")

    output_dir = Path(args.output_dir)
    produced = []
    run_data = []
    for run_dir in run_dirs:
        data = _load_run(run_dir)
        if data is None:
            continue
        run_data.append(data)
        produced.extend(_plot_single_run(data, output_dir))

    compare_out = _plot_compare_ret50(run_data, output_dir, group_by=str(args.group_by).strip())
    if compare_out is not None:
        produced.append(compare_out)

    if not produced:
        raise SystemExit("No plottable metrics were found.")

    print("Generated plots:")
    for path in produced:
        print(path)


if __name__ == "__main__":
    main()
