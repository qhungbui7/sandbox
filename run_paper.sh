#!/usr/bin/env bash
set -euo pipefail

# Reproducible sweep launcher for AMT-PPO paper experiments.
# Usage: SEEDS="0 1 2" PROJECT=amt ENTITY=your_team bash run_paper.sh baseline

CONFIG=${1:-help}
SEEDS=${SEEDS:-"0 1 2"}
PROJECT=${PROJECT:-amt}
ENTITY=${ENTITY:-}
DEVICE=${DEVICE:-cuda}

BASE_CMD=(python3 amg.py \
  --env-id CartPole-v1 \
  --num-envs 8 \
  --horizon 256 \
  --total-steps 500000 \
  --phase-len 2000 \
  --device "${DEVICE}" \
  --log-interval 10 \
  --wandb \
  --wandb-project "${PROJECT}")

# Optional W&B entity routing
if [[ -n "${ENTITY}" ]]; then
  BASE_CMD+=(--wandb-entity "${ENTITY}")
fi

run_config() {
  local name="$1"; shift
  for seed in ${SEEDS}; do
    local cmd=("${BASE_CMD[@]}" --seed "${seed}" --wandb-run-name "${name}_s${seed}" "$@")
    printf "\n>>> %s\n" "${cmd[*]}"
    "${cmd[@]}"
  done
}

case "${CONFIG}" in
  baseline)
    run_config baseline \
      --amp --amp-dtype float16 \
      --reset-strategy partial --reset-long-fraction 0.5 \
      --lambda-pred 0.01 --pred-coef 0.5
    ;;
  ppo-ff)
    # Feed-forward PPO proxy: traces collapse to the current embedding (alpha=1), no resets/predictor.
    run_config ppo_ff \
      --alpha-base 1.0 --alpha-max 1.0 \
      --reset-strategy none \
      --lambda-pred 0.0 --pred-coef 0.0 \
      --amp --amp-dtype float16
    ;;
  ppo-fixed-trace)
    # Fixed traces (no adaptive gating/resets): alpha_base == alpha_max, no predictor, no resets.
    run_config ppo_fixed_trace \
      --alpha-base 0.5,0.1,0.01 --alpha-max 0.5,0.1,0.01 \
      --reset-strategy none \
      --lambda-pred 0.0 --pred-coef 0.0 \
      --amp --amp-dtype float16
    ;;
  no-amp)
    run_config no_amp \
      --reset-strategy partial --reset-long-fraction 0.5 \
      --lambda-pred 0.01 --pred-coef 0.5
    ;;
  no-pred)
    run_config no_pred \
      --amp --amp-dtype float16 \
      --reset-strategy partial --reset-long-fraction 0.5 \
      --lambda-pred 0.0 --pred-coef 0.0
    ;;
  zero-reset)
    run_config zero_reset \
      --amp --amp-dtype float16 \
      --reset-strategy zero \
      --lambda-pred 0.01 --pred-coef 0.5
    ;;
*)
    cat <<USAGE
Usage: SEEDS="0 1 2" PROJECT=amt ENTITY=team bash run_paper.sh {baseline|ppo-ff|ppo-fixed-trace|no-amp|no-pred|zero-reset}
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
