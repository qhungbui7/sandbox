import sys
from argparse import Namespace
from pathlib import Path

import gymnasium as gym
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


def test_validate_no_strange_params_rejects_a2c_multiple_epochs():
    args = _strict_args_template(algo="a2c", epochs=2, minibatch_size=1024, reset_strategy="partial")
    resolved_cfg = vars(args).copy()
    try:
        validate_no_strange_params(args, resolved_cfg=resolved_cfg, cli_dests=set())
    except ValueError as exc:
        assert "epochs=1" in str(exc)
    else:
        raise AssertionError("Expected strict validator to enforce epochs=1 for A2C.")


def test_validate_no_strange_params_rejects_reinforce_minibatching():
    args = _strict_args_template(algo="reinforce", epochs=1, minibatch_size=256, reset_strategy="partial")
    resolved_cfg = vars(args).copy()
    try:
        validate_no_strange_params(args, resolved_cfg=resolved_cfg, cli_dests=set())
    except ValueError as exc:
        assert "minibatch_size" in str(exc)
    else:
        raise AssertionError("Expected strict validator to enforce full-batch REINFORCE updates.")


def test_validate_no_strange_params_accepts_a2c_single_full_batch():
    args = _strict_args_template(algo="a2c", epochs=1, minibatch_size=1024, reset_strategy="partial")
    resolved_cfg = vars(args).copy()
    validate_no_strange_params(args, resolved_cfg=resolved_cfg, cli_dests=set())


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


def test_ppo_update_recurrent_resets_hidden_on_done_boundaries():
    device = "cpu"
    T, N = 2, 1

    class HiddenRecorderPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25))
            self.hidden_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

        def forward(self, obs, prev_action, hidden):
            h, c = hidden
            self.hidden_inputs.append((h.detach().clone(), c.detach().clone()))
            batch = obs.shape[0]
            logits = torch.stack([self.scale.expand(batch), (-self.scale).expand(batch)], dim=-1)
            value = self.scale.expand(batch)
            next_h = torch.full_like(h, 3.0)
            next_c = torch.full_like(c, 5.0)
            return logits, value, (next_h, next_c)

    ac = HiddenRecorderPolicy().to(device)
    batch = {
        "obs": torch.zeros((T, N, 4), device=device),
        "prev_action": torch.zeros((T, N), device=device, dtype=torch.int64),
        "actions": torch.zeros((T, N), device=device, dtype=torch.int64),
        "logp_old": torch.zeros((T, N), device=device),
        "values_old": torch.zeros((T, N), device=device),
        "rewards": torch.tensor([[1.0], [0.0]], device=device),
        "dones": torch.tensor([[True], [False]], device=device),
        "h0": torch.ones((1, N, 2), device=device),
        "c0": 2.0 * torch.ones((1, N, 2), device=device),
        "value_T": torch.zeros((N,), device=device),
    }

    _ = ppo_update_recurrent(
        ac=ac,
        opt=torch.optim.Adam(ac.parameters(), lr=1e-3),
        batch=batch,
        clip_coef=0.2,
        vf_clip=True,
        target_kl=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ent_coef=0.01,
        epochs=1,
        lam=0.95,
        gamma=0.99,
        generator=torch.Generator(device=device).manual_seed(7),
        device=device,
        use_amp=False,
        amp_dtype=torch.float16,
        grad_scaler=None,
    )

    assert len(ac.hidden_inputs) == T
    h_t0, c_t0 = ac.hidden_inputs[0]
    h_t1, c_t1 = ac.hidden_inputs[1]
    assert torch.equal(h_t0, batch["h0"])
    assert torch.equal(c_t0, batch["c0"])
    assert torch.equal(h_t1, torch.zeros_like(h_t1))
    assert torch.equal(c_t1, torch.zeros_like(c_t1))


def test_ppo_updates_bootstrap_with_dones_not_terminated(monkeypatch):
    set_seed(3)
    device = "cpu"
    envs, obs0 = _build_envs(num_envs=2, seed=3)

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
    traces = trace_update(traces, f_mem(obs0_t, prev_action), alpha_base.expand(envs.num_envs, -1))

    class DummyDrift:
        def update(self, e):
            gate = torch.zeros(e.shape, device=device)
            reset = torch.zeros(e.shape, device=device, dtype=torch.bool)
            return gate, reset

        def reset_where(self, mask):
            return

    batch_ff, _, _, _ = rollout(
        envs=envs,
        ac=ac,
        f_mem=f_mem,
        drift=DummyDrift(),
        predictor=None,
        device=device,
        horizon=6,
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
    batch_ff["terminated"] = torch.zeros_like(batch_ff["dones"])
    batch_ff["dones"][0, 0] = True
    assert torch.any(batch_ff["dones"] != batch_ff["terminated"])

    import src.ppo as ppo_module

    def _fake_gae_ff(rewards, dones, resets, values, last_value, gamma, lam):
        assert torch.equal(dones, batch_ff["dones"])
        return torch.zeros_like(rewards), torch.zeros_like(values)

    monkeypatch.setattr(ppo_module, "compute_gae", _fake_gae_ff)
    _ = ppo_update(
        ac=ac,
        opt=torch.optim.Adam(ac.parameters(), lr=1e-3),
        batch=batch_ff,
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
        generator=torch.Generator(device=device).manual_seed(3),
        device=device,
        use_amp=False,
        amp_dtype=torch.float16,
        grad_scaler=None,
    )

    ac_rec = RecurrentActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=8,
        hidden_dim=hidden_dim,
        feat_dim=feat_dim,
    ).to(device)
    hidden = ac_rec.init_hidden(envs.num_envs, device)
    batch_rec, _, _, _ = rollout_recurrent(
        envs=envs,
        ac=ac_rec,
        device=device,
        horizon=6,
        gamma=0.99,
        obs_normalization="none",
        obs=obs0,
        prev_action=torch.zeros(envs.num_envs, device=device, dtype=torch.int64),
        hidden=hidden,
    )
    batch_rec["terminated"] = torch.zeros_like(batch_rec["dones"])
    batch_rec["dones"][0, 0] = True
    assert torch.any(batch_rec["dones"] != batch_rec["terminated"])

    def _fake_gae_rec(rewards, dones, resets, values, last_value, gamma, lam):
        assert torch.equal(dones, batch_rec["dones"])
        return torch.zeros_like(rewards), torch.zeros_like(values)

    monkeypatch.setattr(ppo_module, "compute_gae", _fake_gae_rec)
    _ = ppo_update_recurrent(
        ac=ac_rec,
        opt=torch.optim.Adam(ac_rec.parameters(), lr=1e-3),
        batch=batch_rec,
        clip_coef=0.2,
        vf_clip=True,
        target_kl=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ent_coef=0.01,
        epochs=1,
        lam=0.95,
        gamma=0.99,
        generator=torch.Generator(device=device).manual_seed(3),
        device=device,
        use_amp=False,
        amp_dtype=torch.float16,
        grad_scaler=None,
    )


def test_rollout_recurrent_resets_prev_action_on_done():
    device = "cpu"

    class AlwaysTruncatedEnv:
        def __init__(self):
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            )

        def reset(self, seed=None, options=None):
            return np.zeros((4,), dtype=np.float32), {}

        def step(self, action):
            obs = np.zeros((4,), dtype=np.float32)
            return obs, 0.0, False, True, {}

    class DeterministicRecurrentPolicy(torch.nn.Module):
        def __init__(self, act_dim: int):
            super().__init__()
            self.act_dim = act_dim

        def forward(self, obs, prev_action, hidden):
            batch = obs.shape[0]
            logits = torch.full((batch, self.act_dim), -1000.0, device=obs.device)
            logits[:, 1] = 1000.0
            value = torch.zeros(batch, device=obs.device)
            return logits, value, hidden

    envs = EnvPool([AlwaysTruncatedEnv])
    obs0, _ = envs.reset(seed=0)
    ac = DeterministicRecurrentPolicy(act_dim=2).to(device)
    hidden = (torch.zeros((1, 1, 1), device=device), torch.zeros((1, 1, 1), device=device))
    prev_action = torch.zeros(1, device=device, dtype=torch.int64)

    batch, _obs, prev_action_last, _hidden = rollout_recurrent(
        envs=envs,
        ac=ac,
        device=device,
        horizon=4,
        gamma=0.99,
        obs_normalization="none",
        obs=obs0,
        prev_action=prev_action,
        hidden=hidden,
    )

    assert torch.all(batch["dones"])
    assert torch.equal(batch["prev_action"], torch.zeros_like(batch["prev_action"]))
    assert torch.equal(prev_action_last, torch.zeros_like(prev_action_last))


def test_rollout_resets_prev_action_and_next_mem_on_done():
    device = "cpu"
    feat_dim = 3
    M = 2

    class AlwaysTruncatedEnv:
        def __init__(self):
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            )

        def reset(self, seed=None, options=None):
            return np.zeros((4,), dtype=np.float32), {}

        def step(self, action):
            obs = np.zeros((4,), dtype=np.float32)
            return obs, 0.0, False, True, {}

    class DeterministicPolicy(torch.nn.Module):
        def forward(self, obs, prev_action, traces):
            batch = obs.shape[0]
            logits = torch.full((batch, 2), -1000.0, device=obs.device)
            logits[:, 1] = 1000.0
            value = torch.zeros(batch, device=obs.device)
            return logits, value

    class PrevActionFeature(torch.nn.Module):
        def forward(self, obs, prev_action):
            return prev_action.float().unsqueeze(-1).expand(-1, feat_dim)

    envs = EnvPool([AlwaysTruncatedEnv])
    obs0, _ = envs.reset(seed=0)
    ac = DeterministicPolicy().to(device)
    f_mem = PrevActionFeature().to(device)
    prev_action = torch.zeros(1, device=device, dtype=torch.int64)
    alpha_base = torch.tensor([[0.5, 0.1]], device=device)
    traces = torch.zeros((1, M, feat_dim), device=device)
    obs0_t = torch.as_tensor(obs0, device=device, dtype=torch.float32)
    traces = trace_update(traces, f_mem(obs0_t, prev_action), alpha_base.expand(1, -1))

    batch, _obs, prev_action_last, _traces = rollout(
        envs=envs,
        ac=ac,
        f_mem=f_mem,
        drift=None,
        predictor=None,
        device=device,
        horizon=4,
        gamma=0.99,
        lambda_pred=0.0,
        obs_normalization="none",
        alpha_base=alpha_base,
        alpha_max=alpha_base.clone(),
        reset_strategy="none",
        reset_long_fraction=0.5,
        obs=obs0,
        prev_action=prev_action,
        traces=traces,
    )

    assert torch.all(batch["dones"])
    assert torch.equal(batch["prev_action"], torch.zeros_like(batch["prev_action"]))
    assert torch.equal(prev_action_last, torch.zeros_like(prev_action_last))
    assert torch.equal(batch["x_mem_next"], torch.zeros_like(batch["x_mem_next"]))


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
