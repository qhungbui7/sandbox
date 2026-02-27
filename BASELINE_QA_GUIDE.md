# Baseline QA Guide

Use this guide to generate baseline-only answers for the questionnaire (PPO-FF), not the novel AMT model.

Prereq: complete the one-time setup in [`INSTALL.md`](INSTALL.md).

This workflow is also listed in [`COMMANDS_TO_RUN.md`](COMMANDS_TO_RUN.md) (Baseline QA section).

## 1) Run the baseline (PPO-FF) with reporting enabled (GPU-first)

Choose GPU index:

```bash
export GPU_ID=0
export WANDB_PROJECT=amt
export WANDB_ENTITY=your-team
export WANDB_DIR=wandb
export WANDB_MODE=online
```

```bash
.venv/bin/python amg.py configs/cartpole/ppo_ff.yaml \
  --run-note "ppo-ff baseline for questionnaire answers" \
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

This creates `reports/ppo_ff_baseline/run_summary.json` and a `checkpoint.pt`.

## 2) Generate the answer sheet

```bash
.venv/bin/python scripts/generate_baseline_answers.py \
  --run-dir reports/ppo_ff_baseline \
  --out reports/BASELINE_ANSWERS.md
```

## 3) Fill the remaining TODO fields

`reports/BASELINE_ANSWERS.md` is prefilled from the baseline run. Edit the TODO items for:
- Hyperparameter tuning details.
- Failure mode + error analysis.
- Profiling excerpt and optimization.
- Experiment tracking decision.
- PR link and reviewer comment.
- Team contribution.
- Dataset bias/limitation and mitigation.
- Dependency licenses confirmation.
- Final confirmation checkbox.

If you already have a different baseline run, pass its report directory to `--run-dir`.
