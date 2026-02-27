import sys
from pathlib import Path

import torch
import numpy as np

# Ensure repo root on sys.path for direct script-style import of amg.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        vf_coef=0.5,
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
        vf_coef=0.5,
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
