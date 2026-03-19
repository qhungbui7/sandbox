# amt

Adaptive Memory Traces (AMT) is a compact research playground that tests how well a PPO-style agent can react to non-stationary dynamics. The single entry point, `amg.py`, wires together partial observability wrappers, a piecewise drift generator, a recurrent-style memory made from exponential traces, and an adaptive drift monitor.

## Docs (start here)

- [`COMMANDS_TO_RUN.md`](COMMANDS_TO_RUN.md): all copy/paste commands (setup, training, benchmark matrix, plots, gameplay, baseline QA, paper sweeps).
- [`BENCHMARKS_GUIDE.md`](BENCHMARKS_GUIDE.md): what the benchmark regimes/algorithms mean (beginner friendly).
- [`EXAMPLE_USE_ALL_UTILITIES.md`](EXAMPLE_USE_ALL_UTILITIES.md): one-command end-to-end smoke run using the repo utilities.
- [`DOCKER.md`](DOCKER.md): containerized setup (CPU/GPU) for reproducible runs.
- [`BASELINE_QA_GUIDE.md`](BASELINE_QA_GUIDE.md): baseline-only questionnaire workflow (PPO-FF + answer sheet generator).
- [`paper/main.tex`](paper/main.tex): paper source (NeurIPS 2025 submission template).

## How the training stack fits together

1. **Environment wrappers (`PartialObsWrapper`, `PiecewiseDriftWrapper`)** hide state and inject controlled reward/observation shifts to create drift events.  
2. **`EnvPool`** fans out multiple Gymnasium environments so `rollout` can gather `num_envs × horizon` transitions per update while tracking episode statistics.  
3. **Representation learning (`FeatureEncoder`, `ActorCritic`, `Predictor`)** embeds observations + previous actions, rolls them into short/long traces, and predicts next embeddings to expose surprise when drift hits.  
4. **`DriftMonitor`** keeps short/long exponential error averages, opens a gate when the short-term deviation crosses `tau_on`, and triggers trace resets after `K` consecutive alarms (with cooldowns).  
5. **`rollout` + algorithm updater** cooperate to gather batches, compute returns/advantages, and optimize the selected method (`ppo`, `a2c`, `trpo`, `reinforce`, `v-trace`, `v-mpo`, or `dqn`) with optional auxiliary prediction loss.  
6. **Logging**: console prints summarize the last 50 episode returns, mean trace gate value, KL, and clip fraction. When `--wandb` is set, the same metrics plus loss scalars stream to W&B using the step count `num_envs × horizon × updates`.

Each of these components lives in `amg.py`, so you can trace data flow without jumping between modules.

## Environment & dependency setup

See [`INSTALL.md`](INSTALL.md) for the one-time virtualenv + dependency setup.
For containerized reproducibility, use [`DOCKER.md`](DOCKER.md).

> *Note:* Install a CUDA-enabled PyTorch build if you plan to use GPUs or mixed precision.

Optionally, store API keys or defaults in `.env` (already tracked locally). For example:

```
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_ENTITY=your-team
```

Load them before running (e.g., `source .env`).

## Running a baseline experiment

All runnable commands live in [`COMMANDS_TO_RUN.md`](COMMANDS_TO_RUN.md) (single runs, matrix sweeps, plots, gameplay, baseline QA, paper sweeps).

Training commands now require:
- a config path (`amg.py <config.yaml>` or `amg.py --config <path>`)
- a run note (`--run-note "..."`)
- use `--run-postfix` when repeating the same setup

Minimal example (one benchmark config):

```bash
export GPU_ID=0
export WANDB_PROJECT=amt
export WANDB_ENTITY=your-team
export WANDB_DIR=wandb
export WANDB_MODE=online
export REPORT_DIR=reports/benchmarks

.venv/bin/python amg.py configs/benchmarks/cartpole/stationary_fullobs.yaml \
  --algo ppo \
  --run-note "baseline full training with detailed logging" \
  --run-postfix "trial1" \
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

### Key CLI flags (see `.venv/bin/python amg.py --help` for the full list)

| Flag | What it controls |
| --- | --- |
| `--reset-strategy` / `--reset-long-fraction` | How adaptive traces are cleared when drift fires. |
| `--lambda-pred`, `--pred-coef` | Toggles the auxiliary predictor loss. |
| `--rho-s`, `--rho-l`, `--tau-on/off`, `--K` | Drift monitor sensitivity and persistence thresholds. |
| `--wandb`, `--wandb-project`, `--wandb-tags` | Opt-in telemetry to Weights & Biases. |
| `--cuda-id` + `--wandb` | W&B GPU stats are constrained to the selected target GPU. |
| `--run-note` | Required run note; saved in logs/summary to differentiate experiments. |
| `--run-postfix` | Optional postfix appended to report/W&B run names. |
| `--encoder` | Observation encoder: `mlp` (default) or `cnn` (image-based). |
| `--early-stop-*` | Optional early stopping (`metric`, `mode`, `patience`, `min_delta`, warmup updates). |
| `--amp`, `--amp-dtype` | CUDA mixed precision (half precision or bfloat16). |
| `--log-interval` | How often (in updates) to print the rolling metrics. |
| `--cuda-id` | Explicit CUDA device index (e.g., 0–7) to pin the run to a specific GPU. |
| `--env-workers` | Threaded env stepping/reset (helps CPU-heavy envs like `CarRacing-v3`). |
| `--config` / `<config.yaml>` | YAML file to override defaults (flag or positional). CLI flags still win. |
| `--policy` | `amt` (default), `recurrent` (LSTM core), or `ff` (feed-forward proxy). |
| `--algo` | `ppo` (default), `a2c`, `trpo`, `reinforce`, `v-trace`, `v-mpo`, `dqn`. |
| `--action-space` | Action mode: `auto`, `discrete`, or `continuous` (`continuous` currently supports PPO paths). |
| `--carracing-downsample`, `--carracing-grayscale` | Optional CarRacing observation preprocessing to reduce input size. |
| `--frame-stack` | Stack last K observations along channel/last axis (recommended with `--encoder cnn`). |
| `--debug-log` | Enable extra per-update PPO debug diagnostics in `metrics.jsonl`/W&B. |
| `--tf32`, `--adam-fused`, `--compile`, `--compile-mode` | CUDA speed knobs (`--compile-mode reduce-overhead` is lower warmup overhead; `max-autotune` may win on long runs). |

## Modular config layout

Configs are now split into reusable section files:

- `configs/env/` (environment + observability/drift setup)
- `configs/model/` (policy/algo + AMT model settings)
- `configs/training/` (rollout/optimizer schedule)
- `configs/other/` (runtime/logging/reporting)

Run entries (for example `configs/cartpole/amt.yaml` and `configs/benchmarks/*/*.yaml`) are manifests that point to these sections:

```yaml
config_paths:
  env: ../env/cartpole_nonstationary_partialobs.yaml
  model: ../model/amt.yaml
  training: ../training/full_501760.yaml
  other: ../other/default_cuda.yaml
overrides:
  other:
    wandb_run_name: amt_baseline
```

The run command is unchanged: choose the manifest path you want to run:

```bash
.venv/bin/python amg.py <path-to-config.yaml> --run-note \"...\"
```

Supported config composition keys are:

- `config_paths` (or `configs`): map of section name to YAML path
- `include` / `includes` / `inherit` / `inherits`: optional extra base configs
- `overrides`: final values applied last

## W&B logging guide

1. Make sure `wandb` is installed and `WANDB_API_KEY` is exported (or present in `.env`).  
2. Start a run with `--wandb`. Optional extras:
   - `--wandb-project` sets the destination project (default `amt`).  
   - `--wandb-entity` routes logs to a specific team.  
   - `--wandb-run-name` and `--wandb-tags` make runs easier to search (comma-separated tags).  
   - `--wandb-mode {online,offline,disabled}` and `--wandb-dir` help when you need air-gapped logging.  
3. Metrics logged each update: rolling return/length, gate mean, algorithm-specific losses, plus the cumulative frame count.

W&B logging is optional; for setup details see [`INSTALL.md`](INSTALL.md).

## Baseline variants (paper runs)

Configs live under `configs/<env>/` (e.g., `configs/cartpole/`). Paper-style multi-seed sweeps are launched via `run_paper.sh` (expects `CUDA_ID`, `SEEDS`, `PROJECT`, and `RUN_NOTE`; optional `ENTITY`).

Copy/paste commands for single runs and sweeps live in [`COMMANDS_TO_RUN.md`](COMMANDS_TO_RUN.md).

## Benchmark matrix (stationary + non-stationary)

Beyond the original non-stationary partial-observation setup, this repo now includes a benchmark matrix under `configs/benchmarks/`:

- `stationary_fullobs`
- `stationary_partialobs`
- `nonstationary_fullobs`
- `nonstationary_partialobs`

for `CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`, and `CarRacing-v3` (via discrete-action wrapper).

The runner script is `scripts/run_benchmark_matrix.py`. It now runs full-training configs by default, auto-generates per-run plots/gameplay (best + last checkpoints), and writes aggregated mean/std/95% CI summaries.

Detailed beginner guide: [`BENCHMARKS_GUIDE.md`](BENCHMARKS_GUIDE.md). End-to-end smoke workflow: [`EXAMPLE_USE_ALL_UTILITIES.md`](EXAMPLE_USE_ALL_UTILITIES.md).

## Training metrics, plots, gameplay videos

Training now logs richer metrics to each reported run folder:

- `metrics.jsonl` (per-update metrics history)
- `train.log` (console-style update lines)
- `run_summary.json` (final summary + artifact pointers)
- `run_summary.json["active_args"]` (filtered active config, separate from full `args`)

`tqdm` progress bars show update progress, FPS, and ETA by default (disable with `--no-tqdm`).

`scripts/plot_training.py` now writes per-run dashboards (`*_dashboard.png`) plus an aggregate smooth-return plot (`comparison_ret50_smooth.png`).

`scripts/summarize_benchmarks.py` now also writes `summary.csv` with derived collapse/instability metrics (KL/clip/value-loss/event fractions and run score).

For plotting (`scripts/plot_training.py`), gameplay (`scripts/record_gameplay.py`), and summary tables (`scripts/summarize_benchmarks.py`), use the commands in [`COMMANDS_TO_RUN.md`](COMMANDS_TO_RUN.md).

## Mixed precision (AMP) details

- Turn on with `--amp`; choose the compute dtype via `--amp-dtype {float16,bfloat16}` (default `float16`).  
- AMP is only enabled when `--device` points to CUDA; otherwise the flag is ignored with a warning.  
- `ppo_update` wraps forward/backward passes in `torch.cuda.amp.autocast`, and automatically uses `GradScaler` when training in `float16`.  
- The rollout / environment interaction path stays in fp32 for numerical stability, so only the optimizer-heavy section benefits from AMP.

Mixed precision typically yields a 1.3–1.6× throughput bump on consumer GPUs for this workload, while keeping policy/value training curves close to the fp32 baseline.

## Troubleshooting tips

- **Missing dependencies**: `ModuleNotFoundError` for `numpy`, `gymnasium`, etc. means you need to install the libraries into your active environment.  
- **Slow convergence**: Inspect `diagnostics/gate_mean`—if it sticks near 0, consider lowering `--tau-on` or `--K`; if it saturates at 1, increase `--tau-on` or extend cooldown.  
- **Unstable PPO loss**: Reduce `--clip-coef`, shrink `--lr`, or lower `--amp-dtype` to `bfloat16` if numerical issues appear when using float16 AMP.  
- **W&B offline runs**: Use `--wandb-mode offline` to buffer logs locally, then run `wandb sync` later.

With these additions the repository now documents how AMT works end-to-end, and you can toggle W&B plus mixed precision straight from the CLI without editing the source again.

Run directories are unique: if `reports/.../<run_name>` already exists, the run errors out. Use a different `--report-run-name` or `--run-postfix`.
