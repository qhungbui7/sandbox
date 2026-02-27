#!/usr/bin/env bash
set -euo pipefail

# Reproducible sweep launcher for AMT-PPO paper experiments.
# Usage: CUDA_ID=0 SEEDS="0 1 2" PROJECT=amt ENTITY=your_team RUN_NOTE="paper sweep note" bash run_paper.sh baseline

CONFIG=${1:-help}
SEEDS=${SEEDS:-"0 1 2"}
PROJECT=${PROJECT:-amt}
ENTITY=${ENTITY:-}
DEVICE=${DEVICE:-cuda}
CUDA_ID=${CUDA_ID:-}
REPORT_DIR=${REPORT_DIR:-reports/paper}
WANDB_DIR=${WANDB_DIR:-wandb}
WANDB_MODE=${WANDB_MODE:-online}
RUN_NOTE=${RUN_NOTE:-}

PY=${PY:-}
if [[ -z "${PY}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
  else
    PY="python3"
  fi
fi

if [[ "${DEVICE}" != "cuda" ]]; then
  echo "run_paper.sh is GPU-only. Set DEVICE=cuda."
  exit 2
fi
if [[ -z "${CUDA_ID}" ]]; then
  echo "Please choose a GPU index: CUDA_ID=<index> (example: CUDA_ID=0)"
  exit 2
fi
if [[ -z "${RUN_NOTE}" ]]; then
  echo "Please provide RUN_NOTE to describe the sweep intent and differentiating details."
  exit 2
fi

BASE_ARGS=(\
  --env-id CartPole-v1 \
  --num-envs 8 \
  --horizon 256 \
  --total-steps 501760 \
  --phase-len 2000 \
  --device "${DEVICE}" \
  --cuda-id "${CUDA_ID}" \
  --log-interval 10 \
  --wandb \
  --wandb-project "${PROJECT}" \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-dir "${WANDB_DIR}")

# Optional W&B entity routing
if [[ -n "${ENTITY}" ]]; then
  BASE_ARGS+=(--wandb-entity "${ENTITY}")
fi

run_config() {
  local name="$1"; shift
  local config_path="$1"; shift
  for seed in ${SEEDS}; do
    local run_name="${name}_s${seed}"
    local cmd=(
      "${PY}" amg.py "${config_path}"
      "${BASE_ARGS[@]}"
      --seed "${seed}"
      --wandb-run-name "${run_name}"
      --run-note "${RUN_NOTE} | config=${name} seed=${seed}"
      --report
      --report-dir "${REPORT_DIR}"
      --report-run-name "${run_name}"
      "$@"
    )
    printf "\n>>> %s\n" "${cmd[*]}"
    "${cmd[@]}"
  done
}

case "${CONFIG}" in
  baseline)
    run_config baseline configs/cartpole/amt.yaml \
      --amp --amp-dtype float16 \
      --reset-strategy partial --reset-long-fraction 0.5 \
      --lambda-pred 0.01 --pred-coef 0.5
    ;;
  ppo-ff)
    # Feed-forward PPO proxy: traces collapse to the current embedding (alpha=1), no resets/predictor.
    run_config ppo_ff configs/cartpole/ppo_ff.yaml \
      --alpha-base 1.0 --alpha-max 1.0 \
      --reset-strategy none \
      --lambda-pred 0.0 --pred-coef 0.0 \
      --amp --amp-dtype float16
    ;;
  ppo-fixed-trace)
    # Fixed traces (no adaptive gating/resets): alpha_base == alpha_max, no predictor, no resets.
    run_config ppo_fixed_trace configs/cartpole/ppo_fixed_trace.yaml \
      --alpha-base 0.5,0.1,0.01 --alpha-max 0.5,0.1,0.01 \
      --reset-strategy none \
      --lambda-pred 0.0 --pred-coef 0.0 \
      --amp --amp-dtype float16
    ;;
  no-amp)
    run_config no_amp configs/cartpole/no_amp.yaml \
      --reset-strategy partial --reset-long-fraction 0.5 \
      --lambda-pred 0.01 --pred-coef 0.5
    ;;
  no-pred)
    run_config no_pred configs/cartpole/no_pred.yaml \
      --amp --amp-dtype float16 \
      --reset-strategy partial --reset-long-fraction 0.5 \
      --lambda-pred 0.0 --pred-coef 0.0
    ;;
  zero-reset)
    run_config zero_reset configs/cartpole/zero_reset.yaml \
      --amp --amp-dtype float16 \
      --reset-strategy zero \
      --lambda-pred 0.01 --pred-coef 0.5
    ;;
*)
    cat <<USAGE
Usage: CUDA_ID=0 SEEDS="0 1 2" PROJECT=amt ENTITY=team bash run_paper.sh {baseline|ppo-ff|ppo-fixed-trace|no-amp|no-pred|zero-reset}
  baseline        : AMP fp16 + partial reset + predictor loss (paper default)
  ppo-ff          : Feed-forward PPO proxy (alpha=1, no resets/predictor)
  ppo-fixed-trace : Fixed multi-timescale traces (no gating/resets, no predictor)
  no-amp          : Ablation turning off mixed precision
  no-pred         : Ablation removing predictor loss
  zero-reset      : Ablation using hard zero reset strategy
USAGE
    exit 1
    ;;
esac
