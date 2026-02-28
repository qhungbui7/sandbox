import sys
from argparse import Namespace
from pathlib import Path

import torch
import numpy as np

# Ensure repo root on sys.path for direct script-style import of amg.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import amg  # noqa: E402
from amg import (  # noqa: E402
    ActorCritic,
    RecurrentActorCritic,
    FeatureEncoder,
    Predictor,
    EnvPool,
    make_env_fn,
    trace_update,
    rollout,
    rollout_recurrent,
    ppo_update,
    ppo_update_recurrent,
    set_seed,
    resolve_device,
    load_config_file,
    validate_explicit_required_keys,
    validate_no_unknown_config_keys,
    validate_no_strange_params,
)


def _build_envs(num_envs: int, seed: int = 0):
    env_fns = [
        make_env_fn(
            env_id="CartPole-v1",
            seed=seed + i * 100,
            mask_indices=[1, 3],
            phase_len=50,
            obs_shift_scale=0.05,
            reward_scale_low=0.9,
            reward_scale_high=1.1,
        )
        for i in range(num_envs)
    ]
    envs = EnvPool(env_fns)
    obs0, _ = envs.reset(seed=seed)
    return envs, obs0


def test_resolve_device_override():
    assert resolve_device("cuda", 3) == "cuda:3"
    assert resolve_device("cpu", None) == "cpu"


def test_init_wandb_includes_source_config_and_uploads_file(tmp_path, monkeypatch):
    cfg_path = tmp_path / "unit_config.yaml"
    cfg_path.write_text("seed: 7\nenv_id: CartPole-v1\n")

    class DummyRun:
        def __init__(self):
            self.saved_files = []

        def save(self, path: str, policy: str | None = None):
            self.saved_files.append((path, policy))

    class DummyWandb:
        def __init__(self):
            self.init_kwargs = None
            self._run = DummyRun()

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return self._run

    dummy = DummyWandb()
    monkeypatch.setattr(amg, "wandb", dummy)

    args = Namespace(
        wandb=True,
        device="cpu",
        cuda_id=None,
        wandb_tags="",
        wandb_project="amt-test",
        wandb_entity=None,
        wandb_run_name="unit",
        wandb_mode="offline",
        wandb_dir=None,
        _source_config={"seed": 7, "env_id": "CartPole-v1"},
        _source_config_path=str(cfg_path),
    )

    run = amg.init_wandb(args)
    assert run is not None
    assert dummy.init_kwargs is not None
    assert dummy.init_kwargs["config"]["source_config"] == {"seed": 7, "env_id": "CartPole-v1"}
    assert dummy.init_kwargs["config"]["source_config_path"] == str(cfg_path)
    assert run.saved_files == [(str(cfg_path.resolve()), "now")]


def test_load_config_file_resolves_paths_sections_and_overrides(tmp_path):
    env_cfg = tmp_path / "env.yaml"
    env_cfg.write_text(
        "env:\n"
        "  env_id: CartPole-v1\n"
        "  mask_indices: [1, 3]\n"
        "  phase_len: 2000\n"
    )
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(
        "model:\n"
        "  policy: amt\n"
        "  hidden_dim: 128\n"
        "  feat_dim: 64\n"
    )
    training_cfg = tmp_path / "training.yaml"
    training_cfg.write_text(
        "training:\n"
        "  total_steps: 501760\n"
        "  horizon: 256\n"
        "  epochs: 4\n"
    )
    other_cfg = tmp_path / "other.yaml"
    other_cfg.write_text(
        "other:\n"
        "  device: cuda\n"
        "  wandb: false\n"
    )
    run_cfg = tmp_path / "run.yaml"
    run_cfg.write_text(
        "config_paths:\n"
        "  env: env.yaml\n"
        "  model: model.yaml\n"
        "  training: training.yaml\n"
        "  other: other.yaml\n"
        "overrides:\n"
        "  training:\n"
        "    epochs: 8\n"
        "  wandb: true\n"
    )

    resolved, raw = load_config_file(run_cfg)
    assert raw["config_paths"]["env"] == "env.yaml"
    assert resolved["env_id"] == "CartPole-v1"
    assert resolved["policy"] == "amt"
    assert resolved["epochs"] == 8
    assert resolved["wandb"] is True
    assert resolved["horizon"] == 256


def _strict_args_template(**overrides):
    base = dict(
        env_id="CartPole-v1",
        num_envs=8,
        env_workers=0,
        horizon=128,
        total_steps=200704,
        seed=0,
        device="cpu",
        algo="ppo",
        policy="amt",
        hidden_dim=128,
        feat_dim=64,
        act_embed_dim=16,
        mask_indices="1,3",
        phase_len=1500,
        obs_shift_scale=0.1,
        reward_scale_low=0.8,
        reward_scale_high=1.2,
        gamma=0.99,
        lr=3e-4,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=0.01,
        epochs=4,
        log_interval=10,
        wandb=False,
        report=True,
        report_dir="reports/benchmarks",
        gae_lam=0.95,
        minibatch_size=256,
        clip_coef=0.2,
        vf_clip=True,
        target_kl=0.01,
        ema_tau=0.995,
        alpha_base="1.0",
        alpha_max="1.0",
        reset_strategy="none",
        reset_long_fraction=0.5,
        lambda_pred=0.0,
        pred_coef=0.0,
    )
    base.update(overrides)
    return Namespace(**base)


def test_validate_explicit_required_keys_raises_on_missing_critical_key():
    args = _strict_args_template()
    resolved_cfg = vars(args).copy()
    resolved_cfg.pop("lr")
    try:
        validate_explicit_required_keys(args, resolved_cfg=resolved_cfg, cli_dests=set())
    except ValueError as exc:
        assert "lr" in str(exc)
    else:
        raise AssertionError("Expected missing required key error for `lr`.")


def test_validate_explicit_required_keys_accepts_cli_override():
    args = _strict_args_template()
    resolved_cfg = vars(args).copy()
    resolved_cfg.pop("lr")
    validate_explicit_required_keys(args, resolved_cfg=resolved_cfg, cli_dests={"lr"})


def test_validate_explicit_required_keys_rejects_null_config_value():
    args = _strict_args_template()
    resolved_cfg = vars(args).copy()
    resolved_cfg["lr"] = None
    try:
        validate_explicit_required_keys(args, resolved_cfg=resolved_cfg, cli_dests=set())
    except ValueError as exc:
        assert "lr" in str(exc)
    else:
        raise AssertionError("Expected missing required key error for null `lr`.")


def test_validate_explicit_required_keys_ff_does_not_require_amt_drift_knobs():
    args = _strict_args_template(policy="ff")
    resolved_cfg = vars(args).copy()
    for key in ("alpha_base", "alpha_max", "reset_strategy", "rho_s", "rho_l", "tau_on", "tau_off"):
        resolved_cfg.pop(key, None)
    validate_explicit_required_keys(args, resolved_cfg=resolved_cfg, cli_dests=set())


def test_validate_no_unknown_config_keys_raises():
    parser = amg.argparse.ArgumentParser(add_help=False)
    parser.add_argument("--known")
    try:
        validate_no_unknown_config_keys(resolved_cfg={"known": 1, "typo_knob": 2}, parser=parser)
    except ValueError as exc:
        assert "typo_knob" in str(exc)
    else:
        raise AssertionError("Expected unknown-config-key validation to fail.")


def test_validate_no_strange_params_rejects_amt_keys_for_ff_policy():
    args = _strict_args_template(policy="ff")
    resolved_cfg = vars(args).copy()
    try:
        validate_no_strange_params(args, resolved_cfg=resolved_cfg, cli_dests=set())
    except ValueError as exc:
        assert "AMT-only" in str(exc)
    else:
        raise AssertionError("Expected strict validator to reject AMT knobs for ff policy.")


def test_validate_no_strange_params_rejects_carracing_keys_for_non_carracing_env():
    args = _strict_args_template(policy="ff", env_id="CartPole-v1")
    resolved_cfg = vars(args).copy()
    for key in (
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
    ):
        resolved_cfg.pop(key, None)
    resolved_cfg["carracing_downsample"] = 2
    resolved_cfg["carracing_grayscale"] = True
    try:
        validate_no_strange_params(args, resolved_cfg=resolved_cfg, cli_dests=set())
    except ValueError as exc:
        assert "CarRacing-only" in str(exc)
    else:
        raise AssertionError("Expected strict validator to reject CarRacing knobs on non-CarRacing env.")


def test_validate_no_strange_params_rejects_conflicting_algo_override():
    cfg_args = _strict_args_template(algo="ppo")
    resolved_cfg = vars(cfg_args).copy()
    args = _strict_args_template(algo="dqn")
    try:
        validate_no_strange_params(args, resolved_cfg=resolved_cfg, cli_dests={"algo"})
    except ValueError as exc:
        assert "conflict" in str(exc).lower()
    else:
        raise AssertionError("Expected strict validator to reject conflicting --algo override.")


def test_validate_no_strange_params_rejects_algo_specific_cli_knob():
    args = _strict_args_template(algo="ppo")
    resolved_cfg = vars(args).copy()
    try:
        validate_no_strange_params(args, resolved_cfg=resolved_cfg, cli_dests={"dqn_batch_size"})
    except ValueError as exc:
        assert "not used" in str(exc)
    else:
        raise AssertionError("Expected strict validator to reject DQN-only CLI key for PPO.")


def test_amt_rollout_and_update_cpu():
    set_seed(0)
    device = "cpu"
    envs, obs0 = _build_envs(num_envs=2, seed=0)

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(envs.single_action_space.n)
    feat_dim = 16
    hidden_dim = 32
    M = 2
    mem_dim = M * feat_dim

    ac = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=8,
        hidden_dim=hidden_dim,
        feat_dim=feat_dim,
        mem_dim=mem_dim,
    ).to(device)

    f_mem = FeatureEncoder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=8,
        hidden_dim=hidden_dim,
        feat_dim=feat_dim,
    ).to(device)
    f_mem.load_state_dict(ac.f_pol.state_dict())

    alpha_base = torch.tensor([0.5, 0.1], device=device).unsqueeze(0)
    alpha_max = torch.tensor([0.8, 0.3], device=device).unsqueeze(0)

    traces = torch.zeros((envs.num_envs, M, feat_dim), device=device)
    prev_action = torch.zeros(envs.num_envs, device=device, dtype=torch.int64)
    obs0_t = torch.as_tensor(obs0, device=device, dtype=torch.float32)
    x_mem0 = f_mem(obs0_t, prev_action)
    traces = trace_update(traces, x_mem0, alpha_base.expand(envs.num_envs, -1))

    class DummyDrift:
        def update(self, e):
            gate = torch.zeros(e.shape, device=device)
            reset = torch.zeros(e.shape, device=device, dtype=torch.bool)
            return gate, reset

        def reset_where(self, mask):
            return

    drift = DummyDrift()

    batch, obs, prev_action, traces = rollout(
        envs=envs,
        ac=ac,
        f_mem=f_mem,
        drift=drift,
        predictor=None,
        device=device,
        horizon=8,
        gamma=0.99,
        lambda_pred=0.0,
        obs_normalization="none",
        alpha_base=alpha_base,
        alpha_max=alpha_max,
        reset_strategy="none",
        reset_long_fraction=0.5,
        obs=obs0,
        prev_action=prev_action,
        traces=traces,
    )

    generator = torch.Generator(device=device).manual_seed(0)
    stats = ppo_update(
        ac=ac,
        opt=torch.optim.Adam(ac.parameters(), lr=1e-3),
        batch=batch,
        clip_coef=0.2,
        vf_clip=True,
        target_kl=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ent_coef=0.01,
        epochs=1,
        minibatch_size=4,
        lam=0.95,
        gamma=0.99,
        pred=None,
        pred_coef=0.0,
        generator=generator,
        device=device,
        use_amp=False,
        amp_dtype=torch.float16,
        grad_scaler=None,
    )
    # basic sanity: stats keys exist and finite
    for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"]:
        assert key in stats
        assert torch.isfinite(torch.tensor(stats[key]))

def test_recurrent_rollout_and_update_cpu():
    set_seed(1)
    device = "cpu"
    envs, obs0 = _build_envs(num_envs=2, seed=1)

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(envs.single_action_space.n)
    feat_dim = 16
    hidden_dim = 32

    ac = RecurrentActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=8,
        hidden_dim=hidden_dim,
        feat_dim=feat_dim,
    ).to(device)

    hidden = ac.init_hidden(envs.num_envs, device)
    prev_action = torch.zeros(envs.num_envs, device=device, dtype=torch.int64)

    batch, obs, prev_action, hidden = rollout_recurrent(
        envs=envs,
        ac=ac,
        device=device,
        horizon=6,
        gamma=0.99,
        obs_normalization="none",
        obs=obs0,
        prev_action=prev_action,
        hidden=hidden,
    )

    generator = torch.Generator(device=device).manual_seed(1)
    stats = ppo_update_recurrent(
        ac=ac,
        opt=torch.optim.Adam(ac.parameters(), lr=1e-3),
        batch=batch,
        clip_coef=0.2,
        vf_clip=True,
        target_kl=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ent_coef=0.01,
        epochs=1,
        lam=0.95,
        gamma=0.99,
        generator=generator,
        device=device,
        use_amp=False,
        amp_dtype=torch.float16,
        grad_scaler=None,
    )
    for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"]:
        assert key in stats
        assert torch.isfinite(torch.tensor(stats[key]))
    debug_stats = ppo_update_recurrent(
        ac=ac,
        opt=torch.optim.Adam(ac.parameters(), lr=1e-3),
        batch=batch,
        clip_coef=0.2,
        vf_clip=True,
        target_kl=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ent_coef=0.01,
        epochs=1,
        lam=0.95,
        gamma=0.99,
        generator=torch.Generator(device=device).manual_seed(1),
        device=device,
        use_amp=False,
        amp_dtype=torch.float16,
        grad_scaler=None,
        debug_cfg={
            "seed": 1,
            "update_idx": 1,
            "action_bins": act_dim,
            "ratio_sample_size": 128,
            "frame_delta_pairs": 16,
        },
    )
    for key in [
        "debug/action/hist_0",
        "debug/value/explained_variance",
        "debug/value/td_error_std",
        "debug/ppo/grad_norm_encoder_mean",
        "debug/ppo/grad_norm_core_mean",
    ]:
        assert key in debug_stats
    assert "terminated" in batch
    assert "truncated" in batch


def test_cnn_encoder_forward_cpu():
    device = "cpu"
    batch = 4
    obs_shape = (48, 48, 3)
    obs_dim = int(np.prod(obs_shape))
    act_dim = 5
    feat_dim = 32
    hidden_dim = 64
    M = 2
    mem_dim = M * feat_dim

    ac = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=8,
        hidden_dim=hidden_dim,
        feat_dim=feat_dim,
        mem_dim=mem_dim,
        encoder_type="cnn",
        obs_shape=obs_shape,
    ).to(device)
    prev_action = torch.zeros(batch, device=device, dtype=torch.int64)
    obs = torch.zeros((batch, *obs_shape), device=device, dtype=torch.float32)
    traces = torch.zeros((batch, mem_dim), device=device, dtype=torch.float32)
    logits, value = ac(obs, prev_action, traces)
    assert logits.shape == (batch, act_dim)
    assert value.shape == (batch,)
