#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _infer_regime(args: dict) -> str:
    mask_indices = args.get("mask_indices", [])
    if isinstance(mask_indices, str):
        mask_indices = [x for x in mask_indices.split(",") if x.strip()]
    partial_observable = len(mask_indices) > 0
    phase_len = int(args.get("phase_len", 0))
    obs_shift_scale = float(args.get("obs_shift_scale", 0.0))
    reward_scale_low = float(args.get("reward_scale_low", 1.0))
    reward_scale_high = float(args.get("reward_scale_high", 1.0))
    non_stationary = (phase_len > 0) and ((obs_shift_scale > 0.0) or (reward_scale_low != 1.0) or (reward_scale_high != 1.0))
    if (not non_stationary) and (not partial_observable):
        return "stationary_fullobs"
    if (not non_stationary) and partial_observable:
        return "stationary_partialobs"
    if non_stationary and (not partial_observable):
        return "nonstationary_fullobs"
    return "nonstationary_partialobs"


def _get_metrics_path(run_dir: Path, summary: dict) -> Path:
    artifacts = summary.get("artifacts", {}) if isinstance(summary, dict) else {}
    metrics_jsonl = artifacts.get("metrics_jsonl")
    if isinstance(metrics_jsonl, str) and metrics_jsonl:
        candidate = Path(metrics_jsonl)
        if candidate.exists():
            return candidate
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


def _tail_mean(values: np.ndarray, n: int) -> float:
    tail = values[max(0, values.shape[0] - n) :]
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        return float("nan")
    return float(tail.mean())


def _last_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite[-1])


def _nanmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.max())


def _nansecondmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan")
    sorted_desc = np.sort(finite)[::-1]
    return float(sorted_desc[1])


def _nanquantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def _fraction_true(condition: np.ndarray, valid: np.ndarray | None = None) -> float:
    if valid is None:
        valid = np.ones(condition.shape, dtype=bool)
    denom = int(valid.sum())
    if denom <= 0:
        return float("nan")
    return float((condition & valid).sum() / float(denom))


def _auc(xs: np.ndarray, ys: np.ndarray) -> float:
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs_f = xs[mask]
    ys_f = ys[mask]
    if xs_f.size < 2:
        if xs_f.size == 1:
            return float(ys_f[0])
        return float("nan")
    return float(np.trapezoid(ys_f, xs_f))


def _slope_last_k(xs: np.ndarray, ys: np.ndarray, k: int) -> float:
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs_f = xs[mask]
    ys_f = ys[mask]
    if xs_f.size < 2:
        return float("nan")
    xs_tail = xs_f[max(0, xs_f.size - k) :]
    ys_tail = ys_f[max(0, ys_f.size - k) :]
    if xs_tail.size < 2:
        return float("nan")
    x0 = xs_tail[0]
    slope, _intercept = np.polyfit(xs_tail - x0, ys_tail, 1)
    return float(slope)


def _delta_last_k(values: np.ndarray, k: int) -> float:
    tail = values[max(0, values.shape[0] - k) :]
    tail = tail[np.isfinite(tail)]
    if tail.size < 2:
        return float("nan")
    half = tail.size // 2
    if half < 1:
        return float("nan")
    prev = tail[-(2 * half) : -half]
    last = tail[-half:]
    if (prev.size == 0) or (last.size == 0):
        return float("nan")
    return float(last.mean() - prev.mean())


def _tail_quantile(values: np.ndarray, k: int, q: float) -> float:
    tail = values[max(0, values.shape[0] - k) :]
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        return float("nan")
    return float(np.quantile(tail, q))


def _tail_fraction_gt(values: np.ndarray, k: int, threshold: float) -> float:
    tail = values[max(0, values.shape[0] - k) :]
    valid = np.isfinite(tail)
    if int(valid.sum()) == 0:
        return float("nan")
    return float((tail[valid] > threshold).mean())


def _run_row_with_derived(summary_path: Path) -> tuple[dict, dict]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir = summary_path.parent
    args = data.get("args", {})
    final = data.get("final_metrics", {})
    benchmark = data.get("benchmark", {}) if isinstance(data.get("benchmark"), dict) else {}
    metrics_rows = _load_jsonl(_get_metrics_path(run_dir, data))

    frames = np.asarray([float(row.get("loop/frames", float("nan"))) for row in metrics_rows], dtype=np.float64)
    ret50 = np.asarray([float(row.get("train/ret50", float("nan"))) for row in metrics_rows], dtype=np.float64)
    len50 = np.asarray([float(row.get("train/len50", float("nan"))) for row in metrics_rows], dtype=np.float64)
    approx_kl = np.asarray([float(row.get("loss/approx_kl", float("nan"))) for row in metrics_rows], dtype=np.float64)
    clipfrac = np.asarray([float(row.get("loss/clipfrac", float("nan"))) for row in metrics_rows], dtype=np.float64)
    value_loss = np.asarray([float(row.get("loss/value_loss", float("nan"))) for row in metrics_rows], dtype=np.float64)
    entropy = np.asarray([float(row.get("loss/entropy", float("nan"))) for row in metrics_rows], dtype=np.float64)
    explained_variance = np.asarray(
        [float(row.get("debug/value/explained_variance", float("nan"))) for row in metrics_rows],
        dtype=np.float64,
    )

    hist_rows = [[float(row.get(f"debug/action/hist_{idx}", float("nan"))) for idx in range(7)] for row in metrics_rows]
    hist = np.asarray(hist_rows, dtype=np.float64).reshape((-1, 7))
    hist_total = hist.sum(axis=1, keepdims=True)
    probs = hist / np.maximum(hist_total, 1e-12)
    action_dom = probs.max(axis=1)
    action_entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)

    kl_abs = np.abs(approx_kl)
    ret50_smooth = _rolling_mean(ret50, window=10)
    collapse_flag = (entropy < 1.2) & (action_dom > 0.6)

    kl_event = kl_abs > 0.02
    clip_event = clipfrac > 0.2
    value_loss_p95 = _nanquantile(value_loss, 0.95)
    v_event = value_loss > value_loss_p95

    runtime = data.get("runtime_sec")
    frames_final = float(final.get("loop/frames", float("nan")))
    fps_est = None
    if isinstance(runtime, (int, float)) and runtime > 0:
        fps_est = frames_final / float(runtime)

    collapse_indices = np.flatnonzero(collapse_flag)
    if collapse_indices.size > 0:
        time_to_collapse = float(frames[collapse_indices[0]])
    else:
        time_to_collapse = float("nan")

    last_k = 15
    summary_row = {
        "run_id": data.get("run_id", run_dir.name),
        "env_id": data.get("env", {}).get("env_id", args.get("env_id")),
        "regime": benchmark.get("name") or _infer_regime(args),
        "algorithm": data.get("model", {}).get("algorithm", args.get("algo", "ppo")),
        "policy": data.get("model", {}).get("policy", args.get("policy", "amt")),
        "seed": args.get("seed"),
        "frames_total": _last_finite(frames),
        "best_model_ret50": _nanmax(ret50),
        "second_model_ret50": _nansecondmax(ret50),
        "last_model_ret50": _last_finite(ret50),
        "best_ret50": _nanmax(ret50),
        "final_ret50_mean_last10": _tail_mean(ret50, 10),
        "auc_ret50": _auc(frames, ret50),
        "ret_slope_last20": _slope_last_k(frames, ret50, 20),
        "len_final_mean_last10": _tail_mean(len50, 10),
        "frac_len_below_900": _fraction_true(len50 < 900.0, np.isfinite(len50)),
        "kl_abs_max": _nanmax(kl_abs),
        "frac_kl_abs_gt_0.02": _fraction_true(kl_abs > 0.02, np.isfinite(kl_abs)),
        "clipfrac_max": _nanmax(clipfrac),
        "frac_clipfrac_gt_0.2": _fraction_true(clipfrac > 0.2, np.isfinite(clipfrac)),
        "value_loss_p95": value_loss_p95,
        "value_loss_max": _nanmax(value_loss),
        "frac_value_loss_gt_20": _fraction_true(value_loss > 20.0, np.isfinite(value_loss)),
        "entropy_end": _last_finite(entropy),
        "action_dom_end": _last_finite(action_dom),
        "collapse_frac": _fraction_true(collapse_flag, np.isfinite(entropy) & np.isfinite(action_dom)),
        "time_to_collapse": time_to_collapse,
        "explained_variance_end": _last_finite(explained_variance),
        "frac_ev_lt_0": _fraction_true(explained_variance < 0.0, np.isfinite(explained_variance)),
        "score": (
            _auc(frames, ret50)
            - 50.0 * _fraction_true(clipfrac > 0.2, np.isfinite(clipfrac))
            - 50.0 * _fraction_true(kl_abs > 0.02, np.isfinite(kl_abs))
            - 100.0 * _fraction_true(collapse_flag, np.isfinite(entropy) & np.isfinite(action_dom))
        ),
        "ret50": float(final.get("train/ret50", float("nan"))),
        "len50": float(final.get("train/len50", float("nan"))),
        "frames": frames_final,
        "runtime_sec": runtime,
        "fps_est": fps_est,
        "ret50_smooth_slope_lastK": _slope_last_k(frames, ret50_smooth, last_k),
        "ret50_smooth_delta_lastK": _delta_last_k(ret50_smooth, last_k),
        "value_loss_p95_lastK": _tail_quantile(value_loss, last_k, 0.95),
        "frac_clip_gt_0.2_lastK": _tail_fraction_gt(clipfrac, last_k, 0.2),
        "frac_kl_gt_0.02_lastK": _tail_fraction_gt(kl_abs, last_k, 0.02),
    }

    derived = {
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
    return summary_row, derived


def _collect_rows(report_dir: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(report_dir.glob("*/run_summary.json")):
        row, _derived = _run_row_with_derived(summary_path)
        rows.append(row)
    return rows


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean_std_ci95(values: list[float]) -> dict:
    clean = [float(v) for v in values if _is_number(v)]
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


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("env_id")),
            str(row.get("regime")),
            str(row.get("algorithm")),
            str(row.get("policy")),
        )
        grouped[key].append(row)

    out = []
    for key in sorted(grouped):
        env_id, regime, algorithm, policy = key
        entries = grouped[key]
        ret_stats = _mean_std_ci95([r.get("ret50") for r in entries])
        len_stats = _mean_std_ci95([r.get("len50") for r in entries])
        fps_stats = _mean_std_ci95([r.get("fps_est") for r in entries])
        out.append(
            {
                "env_id": env_id,
                "regime": regime,
                "algorithm": algorithm,
                "policy": policy,
                "runs": len(entries),
                "ret50_mean": ret_stats["mean"],
                "ret50_std": ret_stats["std"],
                "ret50_ci95": ret_stats["ci95"],
                "len50_mean": len_stats["mean"],
                "len50_std": len_stats["std"],
                "len50_ci95": len_stats["ci95"],
                "fps_est_mean": fps_stats["mean"],
                "fps_est_std": fps_stats["std"],
                "fps_est_ci95": fps_stats["ci95"],
            }
        )
    return out


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "env_id", "regime", "algorithm", "policy", "seed", "ret50", "len50", "frames", "runtime_sec", "fps_est"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "env_id",
        "regime",
        "algorithm",
        "policy",
        "seed",
        "frames_total",
        "best_model_ret50",
        "second_model_ret50",
        "last_model_ret50",
        "best_ret50",
        "final_ret50_mean_last10",
        "auc_ret50",
        "ret_slope_last20",
        "len_final_mean_last10",
        "frac_len_below_900",
        "kl_abs_max",
        "frac_kl_abs_gt_0.02",
        "clipfrac_max",
        "frac_clipfrac_gt_0.2",
        "value_loss_p95",
        "value_loss_max",
        "frac_value_loss_gt_20",
        "entropy_end",
        "action_dom_end",
        "collapse_frac",
        "time_to_collapse",
        "explained_variance_end",
        "frac_ev_lt_0",
        "score",
        "ret50_smooth_slope_lastK",
        "ret50_smooth_delta_lastK",
        "value_loss_p95_lastK",
        "frac_clip_gt_0.2_lastK",
        "frac_kl_gt_0.02_lastK",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Summary",
        "",
        "| run_id | env_id | regime | algorithm | policy | seed | ret50 | len50 | frames | runtime_sec | fps_est |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_id']} | {r['env_id']} | {r['regime']} | {r['algorithm']} | {r['policy']} | "
            f"{r['seed']} | {r['ret50']} | {r['len50']} | {r['frames']} | {r['runtime_sec']} | {r['fps_est']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_aggregate_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "env_id",
        "regime",
        "algorithm",
        "policy",
        "runs",
        "ret50_mean",
        "ret50_std",
        "ret50_ci95",
        "len50_mean",
        "len50_std",
        "len50_ci95",
        "fps_est_mean",
        "fps_est_std",
        "fps_est_ci95",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_aggregate_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Aggregate Summary",
        "",
        "| env_id | regime | algorithm | policy | runs | ret50 mean | ret50 std | ret50 95% ci | len50 mean | len50 std | len50 95% ci | fps mean | fps std | fps 95% ci |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(value) -> str:
        if value is None:
            return "None"
        return f"{float(value):.6f}"

    for r in rows:
        lines.append(
            f"| {r['env_id']} | {r['regime']} | {r['algorithm']} | {r['policy']} | {r['runs']} | "
            f"{fmt(r['ret50_mean'])} | {fmt(r['ret50_std'])} | {fmt(r['ret50_ci95'])} | "
            f"{fmt(r['len50_mean'])} | {fmt(r['len50_std'])} | {fmt(r['len50_ci95'])} | "
            f"{fmt(r['fps_est_mean'])} | {fmt(r['fps_est_std'])} | {fmt(r['fps_est_ci95'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize benchmark runs into CSV + Markdown for easy comparison.")
    p.add_argument("--report-dir", type=str, default="reports/benchmarks")
    p.add_argument("--output-dir", type=str, default="reports/analysis")
    args = p.parse_args()

    rows = _collect_rows(Path(args.report_dir))
    if not rows:
        raise SystemExit(f"No run_summary.json files found under {args.report_dir}")
    aggregate_rows = _aggregate_rows(rows)

    out_dir = Path(args.output_dir)
    csv_path = out_dir / "benchmark_summary.csv"
    summary_csv_path = out_dir / "summary.csv"
    md_path = out_dir / "benchmark_summary.md"
    agg_csv_path = out_dir / "benchmark_aggregate_summary.csv"
    agg_md_path = out_dir / "benchmark_aggregate_summary.md"
    _write_csv(rows, csv_path)
    _write_summary_csv(rows, summary_csv_path)
    _write_markdown(rows, md_path)
    _write_aggregate_csv(aggregate_rows, agg_csv_path)
    _write_aggregate_markdown(aggregate_rows, agg_md_path)
    print(f"Wrote {len(rows)} rows")
    print(csv_path)
    print(summary_csv_path)
    print(md_path)
    print(agg_csv_path)
    print(agg_md_path)


if __name__ == "__main__":
    main()
