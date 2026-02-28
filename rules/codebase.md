# Codebase Rules (AMT)

## 1) Core run contract
- Use `amg.py` as the single training entrypoint.
- Every run must provide a non-empty `--run-note`.
- CUDA runs must pass explicit `--cuda-id`.
- `--policy recurrent` is only valid with `--algo ppo`.
- `--frame-stack > 1` is only valid with `--encoder cnn`.
- `--encoder cnn` is currently invalid with `--algo dqn`.

## 2) Config system rules
- Prefer manifest configs that compose section files via `config_paths` (`env`, `model`, `training`, `other`).
- `config_paths`/`configs`, `include`/`includes`/`inherit`/`inherits`, and `overrides` are supported composition keys.
- Keep relative paths correct from the manifest’s own directory.
- `overrides` are final and must match resolved values.
- Unknown config keys are forbidden and should fail fast.
- Required keys must be explicit in YAML or explicit via CLI; do not rely on implicit defaults.

## 3) Strict parameter ownership
- Do not pass AMT-only knobs (`alpha_*`, drift/predictor/reset knobs) for `ff`/`recurrent` policies.
- Do not pass CarRacing-only knobs (`carracing_downsample`, `carracing_grayscale`) for non-CarRacing envs.
- Treat `algo`, `policy`, and `env_id` as pinned identity keys in manifests; avoid conflicting CLI overrides.
- Reject algorithm-specific knobs when using a different algorithm (PPO/TRPO/V-trace/VMPO/DQN families).
- For recurrent policy, avoid ff-only EMA-style knobs that are unused.

## 4) Section-file completeness
- `configs/env/*` must define env dynamics/observability keys (`env_id`, masking, drift scales).
- `configs/model/*` must define `algo`, `policy`, and network dimensions.
- `configs/training/*` must define rollout, optimizer, and all algorithm-family knobs (even if not used in current run) to keep manifests auditable.
- `configs/other/*` must define runtime/logging keys (`device`, `seed`, `wandb`, `report`, `report_dir`).

## 5) AMT policy rules
- If `reset_strategy != none`, `reset_long_fraction` is required.
- If `(reset_strategy == none)` and `alpha_base == alpha_max`, drift/predictor knobs are considered unused and should not be set.
- If drift is active, require and configure full drift-monitor parameter set (`rho_*`, `beta`, thresholds, cooldown/warmup).

## 6) Optimization rules
- Keep gradient clipping configurable via `max_grad_norm`; do not hardcode clip values in update paths.
- PPO value clipping should be controlled by explicit `vf_clip` and use `clip_coef`.
- Use a single optimizer construction path (`make_adam_optimizer`) so fused/foreach logic is centralized.

## 7) Benchmark runner rules
- Use `scripts/run_benchmark_matrix.py` for benchmark sweeps.
- `--config-paths` is comma-separated.
- Full training is enforced by default (`--require-full-training`): do not override `total_steps/num_envs/horizon` unless intentionally disabling this guard.
- Keep run naming deterministic: env + regime + algo + seed (+ postfix).

## 8) Logging/reporting rules
- Prefer report-enabled runs (`--report`) and unique run names.
- Keep W&B optional; when enabled, propagate run names/tags consistently.
- Preserve `run_summary.json` compatibility (used by summarizers and benchmark aggregation).

## 9) Testing expectations
- Any change to config contracts must update tests in `tests/test_config_manifests.py`.
- Any signature/algorithm-path change must update unit tests in `tests/test_components.py` and `tests/test_algorithms.py`.
- Maintain strict validation coverage for:
  - unknown config keys
  - required explicit keys
  - strange/unused parameter rejection
  - manifest override-resolve consistency

## 10) Implementation style
- Use type hints for new function signatures and keep interfaces explicit.
- Keep algorithm normalization consistent (`normalize_algo_name`) when comparing/validating algos.
- Prefer centralized validators over scattered ad-hoc checks.
- Fail early with actionable error messages that name offending keys/values.

## 11) Practical workflow
- Before adding a new knob:
  - add parser arg in `amg.py`
  - wire into update path(s)
  - add to strict required/ownership validation
  - add to training config section files
  - add to reporting active-args list if relevant
  - update tests
- Before adding a new benchmark manifest:
  - ensure `config_paths` resolve from its directory
  - pin `overrides.model.policy` and model path consistently
  - ensure carracing manifests include carracing env keys
