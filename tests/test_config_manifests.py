import sys
from pathlib import Path

import yaml

# Ensure repo root on sys.path for direct script-style import of amg.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amg import load_config_file  # noqa: E402


def _iter_manifest_paths() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted((ROOT / "configs" / "benchmarks").rglob("*.yaml")))
    paths.extend(sorted((ROOT / "configs" / "cartpole").glob("*.yaml")))
    return paths


def _iter_section_paths(section: str) -> list[Path]:
    return sorted((ROOT / "configs" / section).rglob("*.yaml"))


def test_section_configs_define_required_keys():
    requirements = {
        "env": {"env_id", "mask_indices", "phase_len", "obs_shift_scale", "reward_scale_low", "reward_scale_high"},
        "model": {"algo", "policy", "hidden_dim", "feat_dim", "act_embed_dim"},
        "training": {
            "num_envs",
            "env_workers",
            "horizon",
            "total_steps",
            "gamma",
            "gae_lam",
            "lr",
            "max_grad_norm",
            "clip_coef",
            "vf_clip",
            "target_kl",
            "vf_coef",
            "ent_coef",
            "epochs",
            "minibatch_size",
            "ema_tau",
            "log_interval",
            "trpo_max_kl",
            "trpo_backtrack_coef",
            "trpo_backtrack_iters",
            "trpo_value_epochs",
            "vtrace_rho_clip",
            "vtrace_c_clip",
            "vmpo_topk_frac",
            "vmpo_eta",
            "vmpo_kl_coef",
            "vmpo_kl_target",
            "dqn_replay_size",
            "dqn_batch_size",
            "dqn_learning_starts",
            "dqn_updates_per_iter",
            "dqn_target_update_interval",
            "dqn_double",
            "dqn_eps_start",
            "dqn_eps_end",
            "dqn_eps_decay_steps",
            "gae_ignore_resets",
        },
        "other": {"device", "seed", "wandb", "report", "report_dir"},
    }
    for section, required_keys in requirements.items():
        for cfg_path in _iter_section_paths(section):
            raw = yaml.safe_load(cfg_path.read_text()) or {}
            assert isinstance(raw, dict), f"{cfg_path} must be a YAML mapping."
            section_cfg = raw.get(section)
            assert isinstance(section_cfg, dict), f"{cfg_path} must define top-level `{section}` mapping."
            assert required_keys.issubset(section_cfg.keys()), (
                f"{cfg_path} missing keys: {sorted(required_keys - set(section_cfg.keys()))}"
            )
            if section == "model" and str(section_cfg.get("policy", "")).strip().lower() == "amt":
                amt_required = {"alpha_base", "alpha_max", "reset_strategy"}
                alpha_base = section_cfg.get("alpha_base")
                alpha_max = section_cfg.get("alpha_max")
                fixed_alpha = alpha_base == alpha_max
                reset_strategy = str(section_cfg.get("reset_strategy", "")).strip().lower()
                if reset_strategy != "none":
                    amt_required.add("reset_long_fraction")
                if not (fixed_alpha and reset_strategy == "none"):
                    amt_required.update({"lambda_pred", "pred_coef", "drift_signal"})
                    amt_required.update(
                        {
                            "rho_s",
                            "rho_l",
                            "beta",
                            "tau_soft",
                            "kappa",
                            "tau_on",
                            "tau_off",
                            "K",
                            "cooldown_steps",
                            "warmup_steps",
                        }
                    )
                assert amt_required.issubset(section_cfg.keys()), (
                    f"{cfg_path} missing AMT keys: {sorted(amt_required - set(section_cfg.keys()))}"
                )


def test_ff_and_recurrent_model_configs_do_not_define_amt_only_knobs():
    amt_only = {
        "alpha_base",
        "alpha_max",
        "lambda_pred",
        "pred_coef",
        "reset_strategy",
        "reset_long_fraction",
        "rho_s",
        "rho_l",
        "beta",
        "tau_soft",
        "kappa",
        "tau_on",
        "tau_off",
        "K",
        "cooldown_steps",
        "warmup_steps",
    }
    for cfg_path in _iter_section_paths("model"):
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        model_cfg = raw.get("model") or {}
        if not isinstance(model_cfg, dict):
            continue
        policy = str(model_cfg.get("policy", "")).strip().lower()
        if policy not in {"ff", "recurrent"}:
            continue
        overlap = sorted(amt_only & set(model_cfg.keys()))
        assert not overlap, f"{cfg_path} should not define AMT-only keys for policy={policy}: {overlap}"


def test_runnable_manifests_define_required_sections_and_keys():
    env_required = {"env_id", "mask_indices", "phase_len", "obs_shift_scale", "reward_scale_low", "reward_scale_high"}
    model_required = {"algo", "policy"}
    training_required = {"num_envs", "horizon", "total_steps"}
    other_required = {"device", "seed", "wandb"}

    for cfg_path in _iter_manifest_paths():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        assert isinstance(raw, dict), f"{cfg_path} must be a YAML mapping."

        overrides = raw.get("overrides")
        assert isinstance(overrides, dict), f"{cfg_path} must define an `overrides` mapping."

        env_cfg = overrides.get("env")
        model_cfg = overrides.get("model")
        training_cfg = overrides.get("training")
        other_cfg = overrides.get("other")
        assert isinstance(env_cfg, dict), f"{cfg_path} must define `overrides.env`."
        assert isinstance(model_cfg, dict), f"{cfg_path} must define `overrides.model`."
        assert isinstance(training_cfg, dict), f"{cfg_path} must define `overrides.training`."
        assert isinstance(other_cfg, dict), f"{cfg_path} must define `overrides.other`."

        assert env_required.issubset(env_cfg.keys()), f"{cfg_path} missing env keys: {sorted(env_required - set(env_cfg.keys()))}"
        assert model_required.issubset(model_cfg.keys()), (
            f"{cfg_path} missing model keys: {sorted(model_required - set(model_cfg.keys()))}"
        )
        assert training_required.issubset(training_cfg.keys()), (
            f"{cfg_path} missing training keys: {sorted(training_required - set(training_cfg.keys()))}"
        )
        assert other_required.issubset(other_cfg.keys()), (
            f"{cfg_path} missing other keys: {sorted(other_required - set(other_cfg.keys()))}"
        )

        if "carracing" in cfg_path.parts:
            assert "carracing_downsample" in env_cfg, f"{cfg_path} missing `carracing_downsample`."
            assert "carracing_grayscale" in env_cfg, f"{cfg_path} missing `carracing_grayscale`."
            assert "env_workers" in training_cfg, f"{cfg_path} missing `env_workers`."

        if cfg_path.name.endswith("_recurrent.yaml"):
            assert model_cfg.get("policy") == "recurrent", f"{cfg_path} must pin `policy: recurrent`."
            config_paths = raw.get("config_paths") or {}
            assert isinstance(config_paths, dict), f"{cfg_path} must define `config_paths`."
            assert str(config_paths.get("model", "")).endswith("recurrent_ppo.yaml"), (
                f"{cfg_path} should reference recurrent model config."
            )


def test_manifest_overrides_match_resolved_values():
    for cfg_path in _iter_manifest_paths():
        resolved, raw = load_config_file(cfg_path)
        overrides = (raw or {}).get("overrides") or {}
        assert isinstance(resolved, dict), f"{cfg_path} must resolve to a flat mapping."
        for section_name in ("env", "model", "training", "other"):
            section = overrides.get(section_name) or {}
            assert isinstance(section, dict), f"{cfg_path} `overrides.{section_name}` must be a mapping."
            for key, expected in section.items():
                assert key in resolved, f"{cfg_path} resolved config missing `{key}`."
                assert resolved[key] == expected, (
                    f"{cfg_path} resolved `{key}`={resolved[key]!r} does not match override {expected!r}."
                )
