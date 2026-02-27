# Example: Use All Utilities End-to-End

This is a one-command smoke workflow that exercises most major utilities in the repo on short settings.

Prereq: complete the one-time setup in [`INSTALL.md`](INSTALL.md).

For individual copy/paste commands (setup, training, benchmark matrix, plots, gameplay, baseline QA), see [`COMMANDS_TO_RUN.md`](COMMANDS_TO_RUN.md).

## Quick start

```bash
GPU_ID=0 \
RUN_NOTE="utility smoke run with short steps" \
WANDB_PROJECT=amt \
WANDB_ENTITY=your-team \
WANDB_DIR=wandb \
WANDB_MODE=online \
bash scripts/example_use_all_utilities.sh
```

By default it uses:

1. `reports/example`
2. `reports/example/plots`
3. `reports/example/gameplay`
4. `reports/example/analysis`

## What this script does

The script [`scripts/example_use_all_utilities.sh`](scripts/example_use_all_utilities.sh) runs:

1. Fast tests (`pytest`)
2. Short PPO + DQN training runs (with `--report`)
3. A small benchmark matrix via `scripts/run_benchmark_matrix.py`
4. Training plots via `scripts/plot_training.py`
5. Summary tables via `scripts/summarize_benchmarks.py`
6. A baseline answer sheet via `scripts/generate_baseline_answers.py`
7. Optional: a tiny CarRacing benchmark (only if CarRacing is available)
8. Optional: gameplay recording (only if `moviepy` is installed)

---

## Override python/output locations

```bash
PY=.venv/bin/python \
GPU_ID=0 \
RUN_NOTE="my utility run" \
WANDB_PROJECT=amt \
WANDB_ENTITY=your-team \
WANDB_DIR=wandb \
WANDB_MODE=online \
REPORT_DIR=reports/my_run \
PLOT_DIR=reports/my_run/plots \
GAMEPLAY_DIR=reports/my_run/gameplay \
ANALYSIS_DIR=reports/my_run/analysis \
bash scripts/example_use_all_utilities.sh
```

---

## Expected outputs

1. Run folders + checkpoints + `metrics.jsonl`: `reports/example/...`
2. PNG training plots: `reports/example/plots`
   - `<run_name>_dashboard.png`
   - `<run_name>_training.png`
   - `comparison_ret50_smooth.png`
3. Gameplay MP4 files: `reports/example/gameplay` (if `moviepy` installed)
4. Comparison tables:
   - `reports/example/analysis/summary.csv`
   - `reports/example/analysis/benchmark_summary.csv`
   - `reports/example/analysis/benchmark_summary.md`
   - `reports/example/analysis/benchmark_aggregate_summary.csv`
   - `reports/example/analysis/benchmark_aggregate_summary.md`
5. Baseline form draft:
   - `reports/example/analysis/BASELINE_ANSWERS.md`
