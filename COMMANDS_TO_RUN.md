# AMT Runbook (Copy/Paste)

This is the single place for runnable commands. Other `.md` files link here to avoid duplication.

Prereq: complete the one-time setup in [`INSTALL.md`](INSTALL.md).

## 0) Choose a GPU (or use CPU)

Most commands below assume CUDA. Export a GPU index once (so both Python commands and shell scripts can see it):

```bash
export GPU_ID=0
```

CPU-only runs are supported too (replace `--device cuda --cuda-id ${GPU_ID}` with `--device cpu`).

## 1) Common run settings (W&B + report outputs)

```bash
export WANDB_PROJECT=amt
export WANDB_ENTITY=your-team
export WANDB_DIR=wandb
export WANDB_MODE=online

export REPORT_DIR=reports/benchmarks
```

All training commands below:

- use tqdm by default (disable with `--no-tqdm`)
- enable W&B logging (`--wandb --wandb-mode ${WANDB_MODE} --wandb-dir ...`)
- write local report artifacts (`--report --report-dir ... --report-run-name ...`)
- require a run note (`--run-note "..."`)
- require a config path when invoking `amg.py` (positional `<config.yaml>` or `--config <path>`)
- use `--run-postfix` to safely create a distinct run name; existing run folders are not overwritten
- W&B monitors only the selected target GPU (`--cuda-id`) instead of all visible devices

## 2) Verify installation (fast tests)

```bash
.venv/bin/python -m pytest -q tests/test_components.py tests/test_algorithms.py tests/test_reporting.py
```

## 3) First training run (recommended)

```bash
.venv/bin/python amg.py configs/benchmarks/cartpole/stationary_fullobs.yaml \
  --algo ppo \
  --run-note "baseline stationary cartpole ppo run" \
  --early-stop-metric eval/ret_mean \
  --early-stop-mode max \
  --early-stop-patience 20 \
  --early-stop-min-delta 1.0 \
  --early-stop-warmup-updates 20 \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --wandb \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-run-name cartpole_stationary_fullobs_ppo_s0 \
  --wandb-mode ${WANDB_MODE} \
  --wandb-dir ${WANDB_DIR} \
  --report \
  --report-dir ${REPORT_DIR} \
  --report-run-name cartpole_stationary_fullobs_ppo_s0
```

## 4) Run all algorithms on one task (quick sweep)

```bash
for algo in ppo a2c trpo reinforce v-trace v-mpo dqn; do
  .venv/bin/python amg.py configs/benchmarks/cartpole/stationary_fullobs.yaml \
    --algo "$algo" \
    --run-note "single-task sweep algo=${algo} stationary cartpole" \
    --device cuda \
    --cuda-id ${GPU_ID} \
    --total-steps 10240 \
    --wandb \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-entity ${WANDB_ENTITY} \
    --wandb-run-name "cartpole_stationary_fullobs_${algo}_s0" \
    --wandb-mode ${WANDB_MODE} \
    --wandb-dir ${WANDB_DIR} \
    --report \
    --report-dir ${REPORT_DIR} \
    --report-run-name "cartpole_stationary_fullobs_${algo}_s0"
done
```

## 5) Benchmark matrix (multiple envs/regimes/seeds)

```bash
.venv/bin/python scripts/run_benchmark_matrix.py \
  --algos ppo,a2c,trpo,reinforce,v-trace,v-mpo,dqn \
  --run-note "full benchmark matrix with auto plots/videos" \
  --envs cartpole,acrobot,mountaincar \
  --regimes stationary_fullobs,stationary_partialobs,nonstationary_fullobs,nonstationary_partialobs \
  --seeds 0 \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --report-dir ${REPORT_DIR} \
  --wandb \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-mode ${WANDB_MODE} \
  --wandb-dir ${WANDB_DIR}
```

### Dry run (print commands only)

```bash
.venv/bin/python scripts/run_benchmark_matrix.py \
  --run-note "dry run matrix planning" \
  --dry-run \
  --device cuda \
  --cuda-id ${GPU_ID}
```

### Run benchmark using specific config file(s)

```bash
.venv/bin/python scripts/run_benchmark_matrix.py \
  --run-note "direct config benchmark run" \
  --run-postfix "trial1" \
  --config-paths configs/benchmarks/carracing/stationary_fullobs.yaml \
  --algos ppo \
  --seeds 0 \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --report-dir ${REPORT_DIR} \
  --wandb \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-mode ${WANDB_MODE} \
  --wandb-dir ${WANDB_DIR}
```

Re-run the same setup with a different suffix:

```bash
.venv/bin/python scripts/run_benchmark_matrix.py \
  --run-note "direct config benchmark run" \
  --run-postfix "trial2" \
  --config-paths configs/benchmarks/carracing/stationary_fullobs.yaml \
  --algos ppo \
  --seeds 0 \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --report-dir ${REPORT_DIR}
```

### DQN matrix with DQN-specific overrides

```bash
.venv/bin/python scripts/run_benchmark_matrix.py \
  --algo dqn \
  --run-note "dqn benchmark with custom replay hyperparameters" \
  --envs cartpole \
  --regimes stationary_fullobs \
  --seeds 0 \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --report-dir ${REPORT_DIR} \
  --wandb \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-mode ${WANDB_MODE} \
  --wandb-dir ${WANDB_DIR} \
  --extra-args="--dqn-learning-starts 256 --dqn-batch-size 64 --dqn-updates-per-iter 2"
```

### CarRacing benchmark (discrete-action wrapper)

Full-training run with auto plots/gameplay:

```bash
.venv/bin/python scripts/run_benchmark_matrix.py \
  --algo ppo \
  --run-note "carracing full-training benchmark with visualization" \
  --envs carracing \
  --regimes stationary_fullobs \
  --seeds 0 \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --report-dir ${REPORT_DIR} \
  --wandb \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-mode ${WANDB_MODE} \
  --wandb-dir ${WANDB_DIR}
```

CarRacing recurrent + CNN + frame stack + PPO debug logs:

```bash
.venv/bin/python scripts/run_benchmark_matrix.py \
  --run-note "carracing recurrent cnn frame-stack debug run" \
  --run-postfix "trial1" \
  --envs carracing \
  --regimes stationary_fullobs \
  --seeds 0 \
  --algos ppo \
  --encoder cnn \
  --frame-stack 4 \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --report-dir ${REPORT_DIR} \
  --wandb \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-mode ${WANDB_MODE} \
  --wandb-dir ${WANDB_DIR} \
  --extra-args="--policy recurrent --debug-log"
```

Note: when `--extra-args` starts with a flag (e.g., `--debug-log`), pass it as `--extra-args="..."` (with `=`).

## 6) Paper-style sweeps (CartPole + W&B)

`run_paper.sh` is a small launcher for the CartPole paper baselines (GPU-only):

```bash
CUDA_ID=${GPU_ID} SEEDS="0 1 2" PROJECT=${WANDB_PROJECT} ENTITY=${WANDB_ENTITY} \
RUN_NOTE="paper sweep baseline" WANDB_MODE=online WANDB_DIR=${WANDB_DIR} REPORT_DIR=reports/paper PY=.venv/bin/python \
bash run_paper.sh baseline

CUDA_ID=${GPU_ID} SEEDS="0 1 2" PROJECT=${WANDB_PROJECT} ENTITY=${WANDB_ENTITY} \
RUN_NOTE="paper sweep ppo-ff" WANDB_MODE=online WANDB_DIR=${WANDB_DIR} REPORT_DIR=reports/paper PY=.venv/bin/python \
bash run_paper.sh ppo-ff

CUDA_ID=${GPU_ID} SEEDS="0 1 2" PROJECT=${WANDB_PROJECT} ENTITY=${WANDB_ENTITY} \
RUN_NOTE="paper sweep fixed trace" WANDB_MODE=online WANDB_DIR=${WANDB_DIR} REPORT_DIR=reports/paper PY=.venv/bin/python \
bash run_paper.sh ppo-fixed-trace

CUDA_ID=${GPU_ID} SEEDS="0 1 2" PROJECT=${WANDB_PROJECT} ENTITY=${WANDB_ENTITY} \
RUN_NOTE="paper sweep no amp" WANDB_MODE=online WANDB_DIR=${WANDB_DIR} REPORT_DIR=reports/paper PY=.venv/bin/python \
bash run_paper.sh no-amp

CUDA_ID=${GPU_ID} SEEDS="0 1 2" PROJECT=${WANDB_PROJECT} ENTITY=${WANDB_ENTITY} \
RUN_NOTE="paper sweep no predictor" WANDB_MODE=online WANDB_DIR=${WANDB_DIR} REPORT_DIR=reports/paper PY=.venv/bin/python \
bash run_paper.sh no-pred

CUDA_ID=${GPU_ID} SEEDS="0 1 2" PROJECT=${WANDB_PROJECT} ENTITY=${WANDB_ENTITY} \
RUN_NOTE="paper sweep zero reset" WANDB_MODE=online WANDB_DIR=${WANDB_DIR} REPORT_DIR=reports/paper PY=.venv/bin/python \
bash run_paper.sh zero-reset
```

## 7) End-to-end smoke workflow (all utilities)

Runs tests, trains short PPO + DQN, runs a small matrix, plots, summarizes, generates baseline answers, and records a short gameplay video (if optional deps are installed):

```bash
RUN_NOTE="utility smoke run" \
bash scripts/example_use_all_utilities.sh
```

Override python/output locations:

```bash
PY=.venv/bin/python \
RUN_NOTE="utility smoke run" \
WANDB_PROJECT=${WANDB_PROJECT} \
WANDB_ENTITY=${WANDB_ENTITY} \
WANDB_DIR=${WANDB_DIR} \
REPORT_DIR=reports/example \
PLOT_DIR=reports/example/plots \
GAMEPLAY_DIR=reports/example/gameplay \
ANALYSIS_DIR=reports/example/analysis \
bash scripts/example_use_all_utilities.sh
```

## 8) Plots, tables, and gameplay from reports

Plot training curves:

```bash
.venv/bin/python scripts/plot_training.py --report-dir ${REPORT_DIR} --output-dir reports/plots
```

Grouped smooth-return overlay by config:

```bash
.venv/bin/python scripts/plot_training.py \
  --report-dir ${REPORT_DIR} \
  --output-dir reports/plots \
  --group-by config_path
```

Build comparison tables:

```bash
.venv/bin/python scripts/summarize_benchmarks.py --report-dir ${REPORT_DIR} --output-dir reports/analysis
```

Key outputs:
- `reports/plots/<run_name>_dashboard.png`
- `reports/plots/comparison_ret50_smooth.png`
- `reports/analysis/summary.csv`

Record gameplay videos:

```bash
.venv/bin/python scripts/record_gameplay.py \
  --run-dir ${REPORT_DIR}/cartpole_stationary_fullobs_ppo_s0 \
  --episodes 3 \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --output-dir reports/gameplay
```

Record benchmark gameplay for both checkpoints (`best_train` + `best_eval`, 4 videos total with `--episodes 2`):

```bash
BENCH_RUN_DIR=${REPORT_DIR}/cartpole_stationary_fullobs_ppo_s0

.venv/bin/python scripts/record_gameplay.py \
  --run-dir ${BENCH_RUN_DIR} \
  --checkpoint-kind both \
  --episodes 2 \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --output-dir reports/benchmarks/gameplay
```

Long inference-style video:

```bash
.venv/bin/python scripts/record_gameplay.py \
  --run-dir ${REPORT_DIR}/cartpole_stationary_fullobs_ppo_s0 \
  --episodes 50 \
  --target-steps 500 \
  --video-length 500 \
  --no-record-all-episodes \
  --eval-stationary \
  --eval-fullobs \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --output-dir reports/gameplay \
  --name cartpole_inference_long \
  --overwrite
```

## 9) Baseline QA (PPO-FF) answer sheet

Run the baseline and write a report folder:

```bash
.venv/bin/python amg.py configs/cartpole/ppo_ff.yaml \
  --run-note "ppo-ff baseline for qa questionnaire" \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --wandb \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-run-name ppo_ff_baseline \
  --wandb-mode ${WANDB_MODE} \
  --wandb-dir ${WANDB_DIR} \
  --report \
  --report-dir reports \
  --report-run-name ppo_ff_baseline
```

Generate the prefilled answer sheet:

```bash
.venv/bin/python scripts/generate_baseline_answers.py \
  --run-dir reports/ppo_ff_baseline \
  --out reports/BASELINE_ANSWERS.md
```

## 10) Optional: W&B logging and AMP

CUDA mixed precision (AMP):

```bash
.venv/bin/python amg.py configs/benchmarks/cartpole/stationary_fullobs.yaml \
  --algo ppo \
  --run-note "amp ablation stationary cartpole ppo" \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --wandb \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-run-name cartpole_stationary_fullobs_ppo_amp_s0 \
  --wandb-mode ${WANDB_MODE} \
  --wandb-dir ${WANDB_DIR} \
  --report \
  --report-dir ${REPORT_DIR} \
  --report-run-name cartpole_stationary_fullobs_ppo_amp_s0 \
  --amp \
  --amp-dtype float16
```

Disable tqdm progress bar (optional):

```bash
.venv/bin/python amg.py configs/benchmarks/cartpole/stationary_fullobs.yaml \
  --algo ppo \
  --run-note "tqdm-off logging behavior check" \
  --device cuda \
  --cuda-id ${GPU_ID} \
  --wandb \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-run-name cartpole_stationary_fullobs_ppo_no_tqdm_s0 \
  --wandb-mode ${WANDB_MODE} \
  --wandb-dir ${WANDB_DIR} \
  --report \
  --report-dir ${REPORT_DIR} \
  --report-run-name cartpole_stationary_fullobs_ppo_no_tqdm_s0 \
  --no-tqdm
```

## 11) Useful paths and self-serve help

Outputs:

```bash
ls -la reports
ls -la ${REPORT_DIR}
ls -la reports/plots
ls -la reports/gameplay
ls -la reports/analysis
```

CLI help:

```bash
.venv/bin/python amg.py --help
.venv/bin/python scripts/run_benchmark_matrix.py --help
.venv/bin/python scripts/record_gameplay.py --help
```

If plot/video dependencies are missing, see [`INSTALL.md`](INSTALL.md).
