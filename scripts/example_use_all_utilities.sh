#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke workflow using all major utilities in this repository.
# Defaults to short GPU runs and isolated output directories.

PY="${PY:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
  else
    PY="python3"
  fi
fi

REPORT_DIR="${REPORT_DIR:-reports/example}"
PLOT_DIR="${PLOT_DIR:-${REPORT_DIR}/plots}"
GAMEPLAY_DIR="${GAMEPLAY_DIR:-${REPORT_DIR}/gameplay}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPORT_DIR}/analysis}"
GPU_ID="${GPU_ID:-}"
WANDB_PROJECT="${WANDB_PROJECT:-amt}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_DIR="${WANDB_DIR:-wandb}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_NOTE="${RUN_NOTE:-}"

if [[ -z "${GPU_ID}" ]]; then
  echo "Set GPU_ID to choose a CUDA device. Example:"
  echo "  GPU_ID=0 bash scripts/example_use_all_utilities.sh"
  exit 2
fi
if [[ -z "${RUN_NOTE}" ]]; then
  echo "Set RUN_NOTE to describe this end-to-end run. Example:"
  echo "  RUN_NOTE='utility smoke run with short steps' GPU_ID=0 bash scripts/example_use_all_utilities.sh"
  exit 2
fi

mkdir -p "${REPORT_DIR}" "${PLOT_DIR}" "${GAMEPLAY_DIR}" "${ANALYSIS_DIR}"

echo "[1/8] Run tests"
"${PY}" -m pytest -q tests/test_components.py tests/test_algorithms.py tests/test_reporting.py

echo "[2/8] Train PPO run with reports + tqdm ETA"
WAND_ARGS_PPO=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-run-name cartpole_stationary_fullobs_ppo_s0 --wandb-mode "${WANDB_MODE}" --wandb-dir "${WANDB_DIR}")
if [[ -n "${WANDB_ENTITY}" ]]; then
  WAND_ARGS_PPO+=(--wandb-entity "${WANDB_ENTITY}")
fi
"${PY}" amg.py configs/benchmarks/cartpole/stationary_fullobs.yaml \
  --algo ppo \
  --run-note "${RUN_NOTE} | ppo single run" \
  --device cuda \
  --cuda-id "${GPU_ID}" \
  --total-steps 4096 \
  --num-envs 2 \
  --horizon 32 \
  "${WAND_ARGS_PPO[@]}" \
  --report \
  --report-dir "${REPORT_DIR}" \
  --report-run-name cartpole_stationary_fullobs_ppo_s0

echo "[3/8] Train DQN run with reports"
WAND_ARGS_DQN=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-run-name cartpole_stationary_fullobs_dqn_s0 --wandb-mode "${WANDB_MODE}" --wandb-dir "${WANDB_DIR}")
if [[ -n "${WANDB_ENTITY}" ]]; then
  WAND_ARGS_DQN+=(--wandb-entity "${WANDB_ENTITY}")
fi
"${PY}" amg.py configs/benchmarks/cartpole/stationary_fullobs.yaml \
  --algo dqn \
  --run-note "${RUN_NOTE} | dqn single run" \
  --device cuda \
  --cuda-id "${GPU_ID}" \
  --total-steps 4096 \
  --num-envs 2 \
  --horizon 32 \
  --dqn-learning-starts 128 \
  --dqn-batch-size 64 \
  "${WAND_ARGS_DQN[@]}" \
  --report \
  --report-dir "${REPORT_DIR}" \
  --report-run-name cartpole_stationary_fullobs_dqn_s0

echo "[4/8] Run benchmark matrix utility"
WAND_ARGS_MATRIX=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-mode "${WANDB_MODE}" --wandb-dir "${WANDB_DIR}")
if [[ -n "${WANDB_ENTITY}" ]]; then
  WAND_ARGS_MATRIX+=(--wandb-entity "${WANDB_ENTITY}")
fi
"${PY}" scripts/run_benchmark_matrix.py \
  --algo a2c \
  --run-note "${RUN_NOTE} | matrix smoke" \
  --envs cartpole \
  --regimes stationary_fullobs,nonstationary_partialobs \
  --seeds 0 \
  --device cuda \
  --cuda-id "${GPU_ID}" \
  --no-require-full-training \
  --total-steps 2048 \
  --num-envs 2 \
  --horizon 32 \
  "${WAND_ARGS_MATRIX[@]}" \
  --report-dir "${REPORT_DIR}"

echo "[4b/8] Optional CarRacing smoke benchmark (requires gymnasium[box2d])"
if "${PY}" - <<'PY'
import gymnasium as gym
try:
    env = gym.make("CarRacing-v3")
    env.close()
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
then
  "${PY}" scripts/run_benchmark_matrix.py \
    --algo ppo \
    --run-note "${RUN_NOTE} | carracing matrix smoke" \
    --envs carracing \
    --regimes stationary_fullobs \
    --seeds 0 \
    --device cuda \
    --cuda-id "${GPU_ID}" \
    --no-require-full-training \
    --total-steps 2048 \
    --num-envs 1 \
    --horizon 64 \
    "${WAND_ARGS_MATRIX[@]}" \
    --report-dir "${REPORT_DIR}"
else
  echo "CarRacing-v3 unavailable: skipping CarRacing benchmark."
fi

echo "[5/8] Plot training curves"
"${PY}" scripts/plot_training.py \
  --report-dir "${REPORT_DIR}" \
  --output-dir "${PLOT_DIR}"

echo "[6/8] Summarize runs into CSV + Markdown"
"${PY}" scripts/summarize_benchmarks.py \
  --report-dir "${REPORT_DIR}" \
  --output-dir "${ANALYSIS_DIR}"

echo "[7/8] Generate baseline questionnaire answers from one run"
"${PY}" scripts/generate_baseline_answers.py \
  --run-dir "${REPORT_DIR}/cartpole_stationary_fullobs_ppo_s0" \
  --out "${ANALYSIS_DIR}/BASELINE_ANSWERS.md"

echo "[8/8] Record gameplay video (if moviepy is available)"
if "${PY}" - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("moviepy") else 1)
PY
then
  "${PY}" scripts/record_gameplay.py \
    --run-dir "${REPORT_DIR}/cartpole_stationary_fullobs_ppo_s0" \
    --episodes 2 \
    --device cuda \
    --cuda-id "${GPU_ID}" \
    --output-dir "${GAMEPLAY_DIR}"
else
  echo "moviepy not installed: skipping gameplay recording."
fi

echo "Done."
echo "Reports:   ${REPORT_DIR}"
echo "Plots:     ${PLOT_DIR}"
echo "Gameplay:  ${GAMEPLAY_DIR}"
echo "Analysis:  ${ANALYSIS_DIR}"
