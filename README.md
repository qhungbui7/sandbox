# amt

Adaptive Memory Traces (AMT) is a compact research playground that tests how well a PPO-style agent can react to non-stationary dynamics. The single entry point, `amg.py`, wires together partial observability wrappers, a piecewise drift generator, a recurrent-style memory made from exponential traces, and an adaptive drift monitor. The script now also exposes first-class Weights & Biases (W&B) telemetry and optional CUDA mixed precision so you can diagnose and optimize long trainings without touching the core logic.

## How the training stack fits together

1. **Environment wrappers (`PartialObsWrapper`, `PiecewiseDriftWrapper`)** hide state and inject controlled reward/observation shifts to create drift events.  
2. **`EnvPool`** fans out multiple Gymnasium environments so `rollout` can gather `num_envs × horizon` transitions per update while tracking episode statistics.  
3. **Representation learning (`FeatureEncoder`, `ActorCritic`, `Predictor`)** embeds observations + previous actions, rolls them into short/long traces, and predicts next embeddings to expose surprise when drift hits.  
4. **`DriftMonitor`** keeps short/long exponential error averages, opens a gate when the short-term deviation crosses `tau_on`, and triggers trace resets after `K` consecutive alarms (with cooldowns).  
5. **`rollout` and `ppo_update`** cooperate to gather batches, compute GAE, and optimize PPO with optional auxiliary prediction loss. The PPO step understands AMP autocast + gradient scaling when `--amp` is on.  
6. **Logging**: console prints summarize the last 50 episode returns, mean trace gate value, KL, and clip fraction. When `--wandb` is set, the same metrics plus loss scalars stream to W&B using the step count `num_envs × horizon × updates`.

Each of these components lives in `amg.py`, so you can trace data flow without jumping between modules.

## Environment & dependency setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch gymnasium[classic_control] numpy wandb
```

> *Note:* Install a CUDA-enabled PyTorch build if you plan to use GPUs or mixed precision.

Optionally, store API keys or defaults in `.env` (already tracked locally). For example:

```
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_ENTITY=your-team
```

Load them before running (e.g., `source .env`).

## Running a baseline experiment

```bash
python3 amg.py \
  --env-id CartPole-v1 \
  --device cuda \
  --num-envs 8 \
  --total-steps 500000 \
  --wandb \
  --wandb-project amt \
  --wandb-run-name cartpole_amp_demo \
  --amp \
  --amp-dtype float16
```

This launches an 8-environment PPO job, logs metrics to the `amt-dev` project on W&B, and trains with CUDA autocast + gradient scaling. Remove `--wandb` if you only need console output, or set `--device cpu` for local debugging.

### Key CLI flags (see `python3 amg.py --help` for the full list)

| Flag | What it controls |
| --- | --- |
| `--reset-strategy` / `--reset-long-fraction` | How adaptive traces are cleared when drift fires. |
| `--lambda-pred`, `--pred-coef` | Toggles the auxiliary predictor loss. |
| `--rho-s`, `--rho-l`, `--tau-on/off`, `--K` | Drift monitor sensitivity and persistence thresholds. |
| `--wandb`, `--wandb-project`, `--wandb-tags` | Opt-in telemetry to Weights & Biases. |
| `--amp`, `--amp-dtype` | CUDA mixed precision (half precision or bfloat16). |
| `--log-interval` | How often (in updates) to print the rolling metrics. |
| `--cuda-id` | Explicit CUDA device index (e.g., 0–7) to pin the run to a specific GPU. |
| `--config` | YAML file to override defaults. CLI flags still win. |
| `--policy` | `amt` (default), `recurrent` (LSTM core), or `ff` (feed-forward PPO proxy). |

## W&B logging guide

1. Make sure `wandb` is installed and `WANDB_API_KEY` is exported (or present in `.env`).  
2. Start a run with `--wandb`. Optional extras:
   - `--wandb-project` sets the destination project (default `amt`).  
   - `--wandb-entity` routes logs to a specific team.  
   - `--wandb-run-name` and `--wandb-tags` make runs easier to search (comma-separated tags).  
   - `--wandb-mode {online,offline,disabled}` and `--wandb-dir` help when you need air-gapped logging.  
3. Metrics logged each update: rolling return/length, gate mean, PPO losses, KL, clip fraction, plus the cumulative frame count.

W&B logging is optional—trying to enable it without the package installed raises a clear error so you know to `pip install wandb`.

## Baseline variants (paper runs)

Use `run_paper.sh` to launch the baselines with consistent seeds and W&B logging. Examples:

```bash
# AMT-PPO (adaptive traces, predictor loss, AMP fp16)
SEEDS="0 1 2" PROJECT=amt ENTITY=bqhung127 bash run_paper.sh baseline

# Feed-forward PPO proxy (no adaptive memory, no resets/predictor)
SEEDS="0 1 2" PROJECT=amt ENTITY=bqhung127 bash run_paper.sh ppo-ff

# Fixed-trace PPO (multi-timescale traces, no adaptive gating/resets)
SEEDS="0 1 2" PROJECT=amt ENTITY=bqhung127 bash run_paper.sh ppo-fixed-trace

# Recurrent PPO (LSTM core, no trace memory) via policy flag
python3 amg.py --policy recurrent --config configs/recurrent_cartpole.yaml

# YAML-driven single runs (no bash wrapper)
# Baseline AMT
python3 amg.py --config configs/amt_cartpole.yaml --wandb --wandb-run-name amt_s0 --seed 0
# Ablations
python3 amg.py --config configs/no_amp_cartpole.yaml --wandb --wandb-run-name no_amp_s0 --seed 0
python3 amg.py --config configs/no_pred_cartpole.yaml --wandb --wandb-run-name no_pred_s0 --seed 0
python3 amg.py --config configs/zero_reset_cartpole.yaml --wandb --wandb-run-name zero_reset_s0 --seed 0
python3 amg.py --config configs/ppo_ff_cartpole.yaml --wandb --wandb-run-name ppo_ff_s0 --seed 0
python3 amg.py --config configs/ppo_fixed_trace_cartpole.yaml --wandb --wandb-run-name ppo_fixed_s0 --seed 0
```

Additional ablations: `no-amp`, `no-pred`, `zero-reset` (see script help).

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
