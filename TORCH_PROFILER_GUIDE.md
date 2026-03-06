# Torch Profiler Guide

This guide explains how to profile training in `amg.py` to find CPU/CUDA bottlenecks.

## What it does

When enabled, training uses `torch.profiler` and writes:

- TensorBoard trace files (`*.pt.trace.json`) for timeline analysis.
- A text summary table (`key_averages.txt`) with top operators by time.

Default output directory:

- If `--report` is enabled: `<run_dir>/profiler/<phase>/`
- Otherwise: `reports/profiler/<phase>/`

## Quick start

Run a short profiling job first:

```bash
python3 amg.py configs/benchmarks/cartpole/stationary_fullobs.yaml \
  --run-note "torch profiler quick check" \
  --device cuda \
  --cuda-id 0 \
  --report \
  --report-dir reports/benchmarks \
  --report-run-name cartpole_profiler_quick \
  --torch-profiler \
  --torch-profiler-wait 1 \
  --torch-profiler-warmup 1 \
  --torch-profiler-active 5 \
  --torch-profiler-repeat 1 \
  --eval-interval 0
```

After the run, inspect:

- `reports/benchmarks/cartpole_profiler_quick/profiler/<phase>/key_averages.txt`
- `reports/benchmarks/cartpole_profiler_quick/profiler/<phase>/` trace files

`<phase>` is:

- `recurrent` for recurrent training path.
- `<policy>_<algo>` for the main training path (example: `amt_ppo`, `ff_dqn`).

## View traces in TensorBoard

```bash
tensorboard --logdir reports/benchmarks/cartpole_profiler_quick/profiler
```

Then open TensorBoard and use the **Profile** tab.

## Main flags

- `--torch-profiler`: enable profiling.
- `--torch-profiler-dir`: custom output dir.
- `--torch-profiler-wait`: steps to skip before warmup.
- `--torch-profiler-warmup`: warmup steps (discarded).
- `--torch-profiler-active`: steps recorded per cycle.
- `--torch-profiler-repeat`: number of cycles.
- `--torch-profiler-skip-first`: global initial steps to skip.
- `--torch-profiler-record-shapes`: include tensor shapes.
- `--torch-profiler-profile-memory`: include memory usage.
- `--torch-profiler-with-stack`: include Python stack traces (higher overhead).
- `--torch-profiler-with-flops`: estimate FLOPs where supported.
- `--torch-profiler-sort-by`: sort key for `key_averages.txt`.
- `--torch-profiler-row-limit`: number of rows in `key_averages.txt`.

## Schedule behavior

Each profiled cycle is:

1. `wait` steps (not recorded)
2. `warmup` steps (instrumentation warmup)
3. `active` steps (recorded)

The cycle repeats `repeat` times.

Example:

- `wait=1, warmup=1, active=5, repeat=2` records `10` updates total.

## Tips for cleaner bottleneck data

- Disable eval during profiling: `--eval-interval 0`.
- Use a short run dedicated to profiling.
- Keep `--torch-profiler-with-stack` off unless needed.
- For CUDA runs, profile on the same GPU and settings you use in real training.

## Common issues

- Large overhead: reduce `active`, disable `with_stack`, and keep `repeat` low.
- Huge trace files: lower `active`/`repeat`, or disable shape/memory capture.
- No CUDA kernels in trace: confirm `--device cuda --cuda-id <id>` and CUDA availability.
