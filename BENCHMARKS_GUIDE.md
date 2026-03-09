# RL Benchmarks and Algorithms Guide (Beginner Friendly)

This guide explains the benchmark suite added to this repository and how to run it.
It is written for readers who are new to reinforcement learning (RL) and want a practical map from concept to runnable code.

For copy/paste commands, use [`COMMANDS_TO_RUN.md`](COMMANDS_TO_RUN.md) (single runs, matrix sweeps, plots, gameplay, and baseline QA).

---

## 1) What was added

### New benchmark regimes

The project originally focused on **non-stationary + partially observable** training.
Now it also supports:

1. `stationary_fullobs`
2. `stationary_partialobs`
3. `nonstationary_fullobs`
4. `nonstationary_partialobs`

for three classic-control environments plus one pixel-control environment:

1. `CartPole-v1`
2. `Acrobot-v1`
3. `MountainCar-v0`
4. `CarRacing-v3` (continuous env discretized by wrapper for this repo)

Benchmark config files live in:

- `configs/benchmarks/cartpole/`
- `configs/benchmarks/acrobot/`
- `configs/benchmarks/mountaincar/`
- `configs/benchmarks/carracing/`

### New benchmark runner

A matrix runner script was added:

- `scripts/run_benchmark_matrix.py`

This script can run combinations of:

1. environment(s)
2. regime(s)
3. seed(s)
4. algorithm (`ppo`, `a2c`, `trpo`, `reinforce`, `v-trace`, `v-mpo`, `dqn`)

and prints a compact summary from report files at the end.

---

## 2) Environment cards

These are the exact Gymnasium tasks currently used for the benchmark suite.

## `CartPole-v1`

- Observation space: 4D continuous vector (`Box`, shape `(4,)`)
- Action space: discrete (`Discrete(2)`): move left or right
- Episode limit: 500 steps
- Reward intuition:
  - Usually +1 per time-step the pole stays balanced
- Why it is useful:
  - Fast sanity check
  - Easy to see if training loop/logging is correct
  - Good first environment for debugging algorithm changes

## `Acrobot-v1`

- Observation space: 6D continuous vector (`Box`, shape `(6,)`)
- Action space: discrete (`Discrete(3)`)
- Episode limit: 500 steps
- Reward intuition:
  - Typically negative per step until terminal objective is reached
- Why it is useful:
  - Harder than CartPole
  - Useful for checking stability of policy-gradient methods
  - Good for evaluating memory under partial observability

## `MountainCar-v0`

- Observation space: 2D continuous vector (`Box`, shape `(2,)`)
- Action space: discrete (`Discrete(3)`): push left, no push, push right
- Episode limit: 200 steps
- Reward intuition:
  - Sparse/harder signal (typically -1 per step until goal)
- Why it is useful:
  - Sensitive to exploration quality
  - Helpful for comparing value-based vs policy-gradient behavior

## `CarRacing-v3`

- Observation space: image (`Box`, shape `(96, 96, 3)`)
- Native action space: continuous (`Box`, shape `(3,)`) = `[steer, gas, brake]`
- In this repo:
  - Wrapped with `DiscreteCarRacingWrapper` so training uses discrete actions
  - Default discrete action table includes: steer left/right, gas, brake, and turn+gas combos
  - Supports `--encoder cnn` + `--frame-stack 4` for temporal pixel input
- Why it is useful:
  - Pixel-based control benchmark
  - Harder than vector-state classic-control tasks
  - Useful for stress-testing feature encoders and exploration
- Dependency note:
  - Requires Box2D extras; see [`INSTALL.md`](INSTALL.md) if CarRacing fails to import/build.

---

## 3) Benchmark regime cards

All regimes are produced by changing two wrappers/settings:

1. **Partial observability** via `mask_indices`
2. **Non-stationarity** via `phase_len`, `obs_shift_scale`, and reward scaling range

## `stationary_fullobs`

- `mask_indices: []`
- `phase_len: 0`
- `obs_shift_scale: 0.0`
- `reward_scale_low/high: 1.0 / 1.0`
- Meaning:
  - Agent sees full observation
  - Dynamics/reward distribution does not drift

## `stationary_partialobs`

- `mask_indices`: non-empty (environment-specific)
- Drift disabled (`phase_len: 0`, no shifts/scaling)
- Meaning:
  - Agent sees incomplete observation
  - Environment itself is stationary

## `nonstationary_fullobs`

- `mask_indices: []`
- Drift enabled (`phase_len > 0`, `obs_shift_scale > 0`, reward scaling range not fixed at 1)
- Meaning:
  - Agent sees full observation
  - Observation/reward statistics can change over time

## `nonstationary_partialobs`

- `mask_indices`: non-empty
- Drift enabled
- Meaning:
  - Hardest regime in this suite
  - Incomplete observations plus moving target distribution

---

## 4) Algorithm cards

This repo now supports the following algorithms via `--algo`.

## `ppo` (Proximal Policy Optimization)

- Family: on-policy actor-critic
- Core idea:
  - Uses a clipped policy-ratio objective to avoid overly large updates
- In this code:
  - Uses GAE-based advantages
  - Logs KL and clip fraction
- Good default for first experiments

## `a2c` (Advantage Actor-Critic)

- Family: on-policy actor-critic
- Core idea:
  - Policy gradient weighted by advantage (`log pi(a|s) * A`)
- In this code:
  - Uses same rollout/GAE infrastructure
  - No PPO clipping
- Simpler baseline than PPO

## `trpo` (Trust Region Policy Optimization)

- Family: on-policy trust-region method
- Core idea:
  - Constrains update size under KL divergence
- In this code:
  - Uses backtracking line search with `trpo_*` controls
  - Value function still optimized with gradient steps
- More conservative updates than vanilla policy gradient

## `reinforce`

- Family: Monte Carlo policy gradient
- Core idea:
  - Optimizes expected return directly from discounted returns
- In this code:
  - Uses discounted returns + value baseline regression
- Useful educational baseline (simple objective, typically high variance)

## `v-trace`

- Family: off-policy corrected actor-critic target
- Core idea:
  - Corrects policy/value targets with clipped importance weights
- In this code:
  - Controlled by `vtrace_rho_clip` and `vtrace_c_clip`
- Useful when behavior and update policy are not perfectly aligned

## `v-mpo`

- Family: maximum-a-posteriori policy optimization style update
- Core idea:
  - Focuses policy update on top-advantage actions with soft weighting
- In this code:
  - Uses top-k fraction, temperature-like `eta`, and KL penalty controls
- Useful for comparing alternative policy-improvement dynamics

## `dqn`

- Family: off-policy value-based method
- Core idea:
  - Learns Q-values with replay buffer and target network
- In this code:
  - Epsilon-greedy collection
  - Replay buffer + target network hard updates
  - Optional Double-DQN behavior (`--dqn-double`)
- Good contrast against policy-gradient family

---

## 5) Config layout and naming

Each environment has four files:

1. `stationary_fullobs.yaml`
2. `stationary_partialobs.yaml`
3. `nonstationary_fullobs.yaml`
4. `nonstationary_partialobs.yaml`

Example:

- `configs/benchmarks/cartpole/stationary_fullobs.yaml`

By default these configs use:

1. `policy: amt`
2. `algo: ppo`
3. `wandb: false`
4. `report: true`
5. `report_dir: reports/benchmarks`

You can override algorithm (or any flag) on CLI.

---

## 6) How to run

Copy/paste commands live in [`COMMANDS_TO_RUN.md`](COMMANDS_TO_RUN.md). Minimal example (one benchmark config with W&B + report outputs):

```bash
export GPU_ID=0
export WANDB_PROJECT=amt
export WANDB_ENTITY=your-team
export WANDB_DIR=wandb
export WANDB_MODE=online
export REPORT_DIR=reports/benchmarks

.venv/bin/python amg.py configs/benchmarks/cartpole/stationary_fullobs.yaml \
  --algo ppo \
  --run-note "benchmark guide baseline run" \
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

To sweep environment/regime/seed combinations, use `scripts/run_benchmark_matrix.py` (examples in [`COMMANDS_TO_RUN.md`](COMMANDS_TO_RUN.md)).
The matrix runner requires `--run-note` and automatically appends env/regime/algo/seed details per run.
Use `--run-postfix` when repeating the same sweep to avoid run-name collisions.
If a run folder already exists, execution fails early to protect previous artifacts.

---

## 7) How to check results

Each run writes:

1. `reports/benchmarks/<run_name>/run_summary.json`
2. `reports/benchmarks/<run_name>/checkpoint.pt`
3. `reports/benchmarks/<run_name>/metrics.jsonl`
4. `reports/benchmarks/<run_name>/train.log`

The matrix runner prints per-run rows and aggregate summaries with mean/std/95% CI:

1. per-run: `run_name`, `ret50`, `len50`, `frames`
2. aggregate: `ret50_mean/std/ci95`, `len50_mean/std/ci95`, `frames_mean/std/ci95`

Utility scripts now add:

1. `scripts/plot_training.py`:
   - per-run dashboard: `reports/.../plots/<run_name>_dashboard.png`
   - aggregate smooth-return overlay: `reports/.../plots/comparison_ret50_smooth.png`
2. `scripts/summarize_benchmarks.py`:
   - `reports/.../analysis/summary.csv` (derived metrics like `kl_abs_max`, `collapse_frac`, `time_to_collapse`, `score`)
   - plus the existing benchmark summary CSV/Markdown files

For deeper analysis, compare final metrics by:

1. algorithm (fixed env/regime)
2. regime (fixed env/algo)
3. environment (fixed algo/regime)

Training progress also shows a `tqdm` ETA bar by default (updates completed, FPS, and estimated remaining time).
Disable it with `--no-tqdm` if needed.

Plotting, gameplay recording, and summary-table commands live in [`COMMANDS_TO_RUN.md`](COMMANDS_TO_RUN.md).

---

## 8) Suggested beginner reading order

1. Start with `CartPole-v1` + `stationary_fullobs` + `ppo`
2. Switch only algorithm (`a2c`, `reinforce`, then `dqn`)
3. Turn on partial observability (`stationary_partialobs`)
4. Turn on non-stationarity (`nonstationary_fullobs`)
5. Finally run `nonstationary_partialobs`

This order isolates one source of difficulty at a time.

---

## 9) Notes and limitations

1. Current setup targets **discrete-action** environments (categorical policy head).
2. `policy=recurrent` is currently tied to `algo=ppo`.
3. `dqn` path uses replay/target updates and ignores predictor-loss training.
4. For fair comparisons, keep shared hyperparameters (steps/seeds/device) fixed when comparing algorithms.
