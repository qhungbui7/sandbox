import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amg import ActorCritic, EnvPool, FeatureEncoder, make_env_fn, rollout, set_seed, trace_update  # noqa: E402
import src.algorithms as algorithms_module  # noqa: E402
from src.algorithms import (  # noqa: E402
    DQNReplayBuffer,
    dqn_collect_rollout,
    dqn_update,
    hard_update_,
    update_on_policy,
)


class DummyDrift:
    def __init__(self, device: str):
        self.device = device

    def update(self, e):
        gate = torch.zeros(e.shape, device=self.device)
        reset = torch.zeros(e.shape, device=self.device, dtype=torch.bool)
        return gate, reset

    def reset_where(self, mask):
        return


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


def _build_actor_critic(device: str, obs_dim: int, act_dim: int, feat_dim: int, hidden_dim: int, M: int):
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
    return ac, f_mem, mem_dim


def test_on_policy_algorithm_updates_cpu():
    set_seed(0)
    device = "cpu"
    envs, obs0 = _build_envs(num_envs=2, seed=0)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(envs.single_action_space.n)
    feat_dim = 16
    hidden_dim = 32
    M = 2

    ac, f_mem, _ = _build_actor_critic(device, obs_dim, act_dim, feat_dim, hidden_dim, M)
    alpha_base = torch.tensor([0.5, 0.1], device=device).unsqueeze(0)
    alpha_max = torch.tensor([0.8, 0.3], device=device).unsqueeze(0)
    traces = torch.zeros((envs.num_envs, M, feat_dim), device=device)
    prev_action = torch.zeros(envs.num_envs, device=device, dtype=torch.int64)
    obs0_t = torch.as_tensor(obs0, device=device, dtype=torch.float32)
    traces = trace_update(traces, f_mem(obs0_t, prev_action), alpha_base.expand(envs.num_envs, -1))
    drift = DummyDrift(device=device)

    batch, _, _, _ = rollout(
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

    algorithms = ["ppo", "a2c", "trpo", "reinforce", "v-trace", "v-mpo"]
    for algo in algorithms:
        model = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_embed_dim=8,
            hidden_dim=hidden_dim,
            feat_dim=feat_dim,
            mem_dim=M * feat_dim,
        ).to(device)
        model.load_state_dict(ac.state_dict())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        generator = torch.Generator(device=device).manual_seed(0)
        stats = update_on_policy(
            algo=algo,
            ac=model,
            opt=opt,
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
            trpo_max_kl=0.01,
            trpo_backtrack_coef=0.5,
            trpo_backtrack_iters=5,
            trpo_value_epochs=1,
            vtrace_rho_clip=1.0,
            vtrace_c_clip=1.0,
            vmpo_topk_frac=0.5,
            vmpo_eta=1.0,
            vmpo_kl_coef=1.0,
            vmpo_kl_target=0.01,
        )
        for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"]:
            assert key in stats
            assert torch.isfinite(torch.tensor(stats[key]))


def test_dqn_collect_and_update_cpu():
    set_seed(1)
    device = "cpu"
    envs, obs0 = _build_envs(num_envs=2, seed=1)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(envs.single_action_space.n)
    feat_dim = 16
    hidden_dim = 32
    M = 2

    ac, f_mem, mem_dim = _build_actor_critic(device, obs_dim, act_dim, feat_dim, hidden_dim, M)
    target_ac, _, _ = _build_actor_critic(device, obs_dim, act_dim, feat_dim, hidden_dim, M)
    hard_update_(target_ac, ac)

    alpha_base = torch.tensor([0.5, 0.1], device=device).unsqueeze(0)
    alpha_max = torch.tensor([0.8, 0.3], device=device).unsqueeze(0)
    traces = torch.zeros((envs.num_envs, M, feat_dim), device=device)
    prev_action = torch.zeros(envs.num_envs, device=device, dtype=torch.int64)
    obs0_t = torch.as_tensor(obs0, device=device, dtype=torch.float32)
    traces = trace_update(traces, f_mem(obs0_t, prev_action), alpha_base.expand(envs.num_envs, -1))

    replay = DQNReplayBuffer(capacity=256, obs_dim=obs_dim, trace_dim=mem_dim)
    drift = DummyDrift(device=device)
    action_generator = torch.Generator(device=device).manual_seed(2)

    obs, prev_action, traces, collect_stats = dqn_collect_rollout(
        envs=envs,
        ac=ac,
        f_mem=f_mem,
        drift=drift,
        predictor=None,
        replay=replay,
        device=device,
        horizon=12,
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
        epsilon=0.2,
        action_generator=action_generator,
    )
    assert obs is not None
    assert prev_action is not None
    assert traces is not None
    assert collect_stats["q_mean"] == collect_stats["q_mean"]
    assert len(replay) > 0

    stats = dqn_update(
        ac=ac,
        target_ac=target_ac,
        opt=torch.optim.Adam(ac.parameters(), lr=1e-3),
        replay=replay,
        batch_size=8,
        gamma=0.99,
        double_dqn=True,
        generator=torch.Generator(device=device).manual_seed(3),
        device=device,
        use_amp=False,
        amp_dtype=torch.float16,
        grad_scaler=None,
        max_grad_norm=0.5,
    )
    assert stats is not None
    for key in ["q_loss", "q_mean", "target_q_mean", "td_abs"]:
        assert key in stats
        assert torch.isfinite(torch.tensor(stats[key]))


def test_on_policy_gae_uses_dones_not_terminated(monkeypatch):
    set_seed(4)
    device = "cpu"
    envs, obs0 = _build_envs(num_envs=2, seed=4)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(envs.single_action_space.n)
    feat_dim = 16
    hidden_dim = 32
    M = 2

    ac, f_mem, _ = _build_actor_critic(device, obs_dim, act_dim, feat_dim, hidden_dim, M)
    alpha_base = torch.tensor([0.5, 0.1], device=device).unsqueeze(0)
    alpha_max = torch.tensor([0.8, 0.3], device=device).unsqueeze(0)
    traces = torch.zeros((envs.num_envs, M, feat_dim), device=device)
    prev_action = torch.zeros(envs.num_envs, device=device, dtype=torch.int64)
    obs0_t = torch.as_tensor(obs0, device=device, dtype=torch.float32)
    traces = trace_update(traces, f_mem(obs0_t, prev_action), alpha_base.expand(envs.num_envs, -1))
    drift = DummyDrift(device=device)

    batch, _, _, _ = rollout(
        envs=envs,
        ac=ac,
        f_mem=f_mem,
        drift=drift,
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
    batch["terminated"] = torch.zeros_like(batch["dones"])
    batch["dones"][0, 0] = True
    assert torch.any(batch["dones"] != batch["terminated"])

    calls = {"count": 0}

    def _fake_gae(rewards, dones, resets, values, last_value, gamma, lam):
        calls["count"] += 1
        assert torch.equal(dones, batch["dones"])
        return torch.zeros_like(rewards), torch.zeros_like(values)

    monkeypatch.setattr(algorithms_module, "compute_gae", _fake_gae)

    for algo in ["ppo", "a2c", "trpo", "v-mpo"]:
        model = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_embed_dim=8,
            hidden_dim=hidden_dim,
            feat_dim=feat_dim,
            mem_dim=M * feat_dim,
        ).to(device)
        model.load_state_dict(ac.state_dict())
        _ = update_on_policy(
            algo=algo,
            ac=model,
            opt=torch.optim.Adam(model.parameters(), lr=1e-3),
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
            generator=torch.Generator(device=device).manual_seed(4),
            device=device,
            use_amp=False,
            amp_dtype=torch.float16,
            grad_scaler=None,
            trpo_max_kl=0.01,
            trpo_backtrack_coef=0.5,
            trpo_backtrack_iters=5,
            trpo_value_epochs=1,
            vtrace_rho_clip=1.0,
            vtrace_c_clip=1.0,
            vmpo_topk_frac=0.5,
            vmpo_eta=1.0,
            vmpo_kl_coef=1.0,
            vmpo_kl_target=0.01,
        )

    assert calls["count"] == 4
