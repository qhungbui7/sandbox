#!/usr/bin/env python3
import argparse
import contextlib
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.distributions.categorical import Categorical

import gymnasium as gym


import wandb



def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot(x: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(x.long(), n).float()


class PartialObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, mask_indices: list[int]):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.mask_indices = np.array(mask_indices, dtype=np.int64)
        low = env.observation_space.low.copy()
        high = env.observation_space.high.copy()
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def observation(self, observation):
        obs = np.array(observation, copy=True)
        obs[..., self.mask_indices] = 0.0
        return obs


class PiecewiseDriftWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        seed: int,
        phase_len: int,
        obs_shift_scale: float,
        reward_scale_low: float,
        reward_scale_high: float,
    ):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.rng = np.random.RandomState(seed)
        self.phase_len = int(phase_len)
        self.obs_shift_scale = float(obs_shift_scale)
        self.reward_scale_low = float(reward_scale_low)
        self.reward_scale_high = float(reward_scale_high)

        self.t = 0
        self.phase = 0
        self.shift = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.r_scale = 1.0

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.t = 0
        self.phase = 0
        self.shift = self.rng.randn(*obs.shape).astype(np.float32) * self.obs_shift_scale
        self.r_scale = self.rng.uniform(self.reward_scale_low, self.reward_scale_high)
        return (obs + self.shift), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += 1
        if self.phase_len > 0 and (self.t % self.phase_len) == 0:
            self.phase += 1
            self.shift = self.rng.randn(*obs.shape).astype(np.float32) * self.obs_shift_scale
            self.r_scale = self.rng.uniform(self.reward_scale_low, self.reward_scale_high)
        return (obs + self.shift), (reward * self.r_scale), terminated, truncated, info


class EnvPool:
    def __init__(self, env_fns: list):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

        self._ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_returns = []

    def reset(self, seed: int):
        obs = []
        infos = []
        for i, env in enumerate(self.envs):
            o, info = env.reset(seed=seed + i)
            obs.append(o)
            infos.append(info)
            self._ep_returns[i] = 0.0
            self._ep_lengths[i] = 0
        return np.stack(obs, axis=0), infos

    def step(self, actions: np.ndarray):
        obs, rew, term, trunc, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, t, tr, info = env.step(int(actions[i]))
            self._ep_returns[i] += float(r)
            self._ep_lengths[i] += 1
            if t or tr:
                self.episode_returns.append((float(self._ep_returns[i]), int(self._ep_lengths[i])))
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0
                o, info_reset = env.reset()
                info["reset_info"] = info_reset
            obs.append(o)
            rew.append(r)
            term.append(t)
            trunc.append(tr)
            infos.append(info)
        return (
            np.stack(obs, axis=0),
            np.asarray(rew, dtype=np.float32),
            np.asarray(term, dtype=np.bool_),
            np.asarray(trunc, dtype=np.bool_),
            infos,
        )


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, activation=nn.Tanh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureEncoder(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_embed_dim: int, hidden_dim: int, feat_dim: int):
        super().__init__()
        self.act_emb = nn.Embedding(act_dim, act_embed_dim)
        self.mlp = MLP(obs_dim + act_embed_dim, hidden_dim, feat_dim, activation=nn.Tanh)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        a = self.act_emb(prev_action.long())
        x = torch.cat([obs, a], dim=-1)
        return self.mlp(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_embed_dim: int, hidden_dim: int, feat_dim: int, mem_dim: int):
        super().__init__()
        self.f_pol = FeatureEncoder(obs_dim, act_dim, act_embed_dim, hidden_dim, feat_dim)
        self.core = MLP(feat_dim + mem_dim, hidden_dim, hidden_dim, activation=nn.Tanh)
        self.pi = nn.Linear(hidden_dim, act_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, prev_action: torch.Tensor, traces_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_pol = self.f_pol(obs, prev_action)
        h = self.core(torch.cat([x_pol, traces_flat], dim=-1))
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


class Predictor(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.act_emb = nn.Embedding(act_dim, hidden_dim)
        self.net = MLP(feat_dim + hidden_dim, hidden_dim, feat_dim, activation=nn.Tanh)

    def forward(self, x_mem: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a = self.act_emb(action.long())
        return self.net(torch.cat([x_mem, a], dim=-1))


@dataclass
class DriftMonitorState:
    e_s: torch.Tensor
    e_l: torch.Tensor
    mu: torch.Tensor
    var: torch.Tensor
    gate: torch.Tensor
    pers: torch.Tensor
    cooldown: torch.Tensor
    rearm: torch.Tensor


class DriftMonitor:
    def __init__(
        self,
        num_envs: int,
        rho_s: float,
        rho_l: float,
        beta: float,
        tau_soft: float,
        kappa: float,
        tau_on: float,
        tau_off: float,
        K: int,
        cooldown_steps: int,
        warmup_steps: int,
        eps: float = 1e-8,
        device: str = "cpu",
    ):
        self.num_envs = num_envs
        self.rho_s = float(rho_s)
        self.rho_l = float(rho_l)
        self.beta = float(beta)
        self.tau_soft = float(tau_soft)
        self.kappa = float(kappa)
        self.tau_on = float(tau_on)
        self.tau_off = float(tau_off)
        self.K = int(K)
        self.cooldown_steps = int(cooldown_steps)
        self.warmup_steps = int(warmup_steps)
        self.eps = float(eps)
        self.device = device

        self.state = DriftMonitorState(
            e_s=torch.zeros(num_envs, device=device),
            e_l=torch.zeros(num_envs, device=device),
            mu=torch.zeros(num_envs, device=device),
            var=torch.ones(num_envs, device=device),
            gate=torch.zeros(num_envs, device=device),
            pers=torch.zeros(num_envs, device=device, dtype=torch.int32),
            cooldown=torch.zeros(num_envs, device=device, dtype=torch.int32),
            rearm=torch.ones(num_envs, device=device, dtype=torch.bool),
        )
        self.step_idx = 0

    def reset_where(self, mask: torch.Tensor) -> None:
        s = self.state
        s.e_s[mask] = 0.0
        s.e_l[mask] = 0.0
        s.mu[mask] = 0.0
        s.var[mask] = 1.0
        s.gate[mask] = 0.0
        s.pers[mask] = 0
        s.cooldown[mask] = 0
        s.rearm[mask] = True

    def update(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          gate: (num_envs,)
          reset_event: (num_envs,) bool
        """
        s = self.state

        s.e_s = (1.0 - self.rho_s) * s.e_s + self.rho_s * e
        s.e_l = (1.0 - self.rho_l) * s.e_l + self.rho_l * e

        d = (s.e_s - s.e_l) / (s.e_l + self.eps)

        mu_next = (1.0 - self.beta) * s.mu + self.beta * d
        var_next = (1.0 - self.beta) * s.var + self.beta * (d - mu_next).pow(2)
        s.mu = mu_next
        s.var = var_next

        z = (d - s.mu) / torch.sqrt(s.var + self.eps)

        s.gate = torch.sigmoid((z - self.tau_soft) / self.kappa)

        if self.step_idx < self.warmup_steps:
            s.pers[:] = 0
            s.cooldown[:] = 0
            s.rearm[:] = True
            self.step_idx += 1
            return s.gate, torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        s.cooldown = torch.clamp(s.cooldown - 1, min=0)

        rearm_mask = z < self.tau_off
        s.rearm[rearm_mask] = True

        active = s.rearm & (z > self.tau_on)
        s.pers[active] += 1
        s.pers[~active] = 0

        trigger = (s.pers >= self.K) & (s.cooldown == 0) & s.rearm
        s.pers[trigger] = 0
        s.rearm[trigger] = False
        s.cooldown[trigger] = self.cooldown_steps

        self.step_idx += 1
        return s.gate, trigger


def ema_update_(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
        tp.data.mul_(tau).add_(sp.data, alpha=(1.0 - tau))


def trace_update(z: torch.Tensor, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    z: (n_envs, M, d)
    x: (n_envs, d)
    alpha: (n_envs, M)
    """
    a = alpha.unsqueeze(-1)
    return (1.0 - a) * z + a * x.unsqueeze(1)


def apply_reset(z: torch.Tensor, x_next: torch.Tensor, strategy: str, long_mask: torch.Tensor) -> torch.Tensor:
    if strategy == "zero":
        return torch.zeros_like(z)
    if strategy == "obs":
        return x_next.unsqueeze(1).expand_as(z).clone()
    if strategy == "partial":
        out = z.clone()
        out[:, long_mask] = 0.0
        return out
    raise ValueError(f"Unknown reset strategy: {strategy}")


def parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def autocast_context(device: str, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return contextlib.nullcontext()
    if torch.device(device).type != "cuda":
        return contextlib.nullcontext()
    return amp.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def rollout(
    envs: EnvPool,
    ac: ActorCritic,
    f_mem: FeatureEncoder,
    drift: DriftMonitor,
    predictor: Predictor | None,
    device: str,
    horizon: int,
    gamma: float,
    lambda_pred: float,
    alpha_base: torch.Tensor,
    alpha_max: torch.Tensor,
    reset_strategy: str,
    reset_long_fraction: float,
    obs: np.ndarray,
    prev_action: torch.Tensor,
    traces: torch.Tensor,
):
    n_envs = envs.num_envs
    M = traces.shape[1]
    d = traces.shape[2]

    obs_buf = torch.zeros((horizon, n_envs, obs.shape[-1]), device=device)
    prev_a_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.int64)
    trace_buf = torch.zeros((horizon, n_envs, M, d), device=device)

    act_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.int64)
    logp_buf = torch.zeros((horizon, n_envs), device=device)
    val_buf = torch.zeros((horizon, n_envs), device=device)
    rew_buf = torch.zeros((horizon, n_envs), device=device)
    done_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.bool)
    reset_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.bool)

    xmem_buf = torch.zeros((horizon, n_envs, d), device=device)
    xmem_next_buf = torch.zeros((horizon, n_envs, d), device=device)

    long_start = int(math.floor((1.0 - reset_long_fraction) * M))
    long_mask = torch.zeros(M, device=device, dtype=torch.bool)
    if reset_strategy == "partial":
        long_mask[long_start:] = True

    for t in range(horizon):
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        obs_buf[t] = obs_t
        prev_a_buf[t] = prev_action
        trace_buf[t] = traces

        logits, value = ac(obs_t, prev_action, traces.reshape(n_envs, -1))
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        act_buf[t] = action
        logp_buf[t] = logp
        val_buf[t] = value

        next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        done_env = terminated | truncated

        rew = torch.as_tensor(reward, device=device, dtype=torch.float32)
        done = torch.as_tensor(done_env, device=device, dtype=torch.bool)
        rew_buf[t] = rew
        done_buf[t] = done

        x_mem_t = f_mem(obs_t, prev_action)
        x_mem_t = F.layer_norm(x_mem_t, (x_mem_t.shape[-1],))
        xmem_buf[t] = x_mem_t

        next_obs_t = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
        next_prev_action = action

        x_mem_next = f_mem(next_obs_t, next_prev_action)
        x_mem_next = F.layer_norm(x_mem_next, (x_mem_next.shape[-1],))
        xmem_next_buf[t] = x_mem_next

        logits_next, v_next_prov = ac(next_obs_t, next_prev_action, trace_update(
            traces, x_mem_next, alpha_base.expand(n_envs, -1)
        ).reshape(n_envs, -1))

        delta_prov = rew + gamma * (~done).float() * v_next_prov - value
        pred_err = torch.zeros(n_envs, device=device)
        if (predictor is not None) and (lambda_pred > 0.0):
            x_hat = predictor(x_mem_t, action)
            pred_err = (x_mem_next - x_hat).pow(2).mean(dim=-1)

        e = delta_prov.abs() + lambda_pred * pred_err

        gate, reset_event = drift.update(e)
        reset_event = reset_event & (~done)

        reset_buf[t] = reset_event

        alpha = alpha_base + gate.unsqueeze(-1) * (alpha_max - alpha_base)
        alpha = alpha.clamp(0.0, 1.0)

        traces_reset = traces
        if reset_strategy != "none":
            traces_reset = traces_reset.clone()
            if reset_strategy == "zero" or reset_strategy == "obs":
                traces_reset[reset_event] = apply_reset(traces_reset[reset_event], x_mem_next[reset_event], reset_strategy, long_mask)
            elif reset_strategy == "partial":
                traces_reset[reset_event] = apply_reset(traces_reset[reset_event], x_mem_next[reset_event], reset_strategy, long_mask)

        traces_next = trace_update(traces_reset, x_mem_next, alpha)

        done_mask = done
        if done_mask.any():
            drift.reset_where(done_mask)
            traces_next[done_mask] = 0.0
            traces_next[done_mask] = trace_update(
                traces_next[done_mask],
                x_mem_next[done_mask],
                alpha_base.expand(done_mask.sum(), -1),
            )

        traces = traces_next
        prev_action = next_prev_action
        obs = next_obs

    obs_T = torch.as_tensor(obs, device=device, dtype=torch.float32)
    logits_T, value_T = ac(obs_T, prev_action, traces.reshape(n_envs, -1))

    batch = {
        "obs": obs_buf,
        "prev_action": prev_a_buf,
        "traces": trace_buf,
        "actions": act_buf,
        "logp_old": logp_buf,
        "values_old": val_buf,
        "rewards": rew_buf,
        "dones": done_buf,
        "resets": reset_buf,
        "x_mem": xmem_buf,
        "x_mem_next": xmem_next_buf,
        "value_T": value_T,
        "obs_last": obs_T,
        "prev_action_last": prev_action,
        "traces_last": traces,
    }
    return batch, obs, prev_action, traces


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    resets: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    rewards, dones, resets, values: (T, N)
    last_value: (N,)
    """
    T, N = rewards.shape
    adv = torch.zeros((T, N), device=rewards.device)
    last_gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = (~dones[t]).float()
        delta = rewards[t] + gamma * nonterminal * next_value - values[t]
        trunc = (~resets[t]).float()
        last_gae = delta + gamma * lam * nonterminal * trunc * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def ppo_update(
    ac: ActorCritic,
    opt: torch.optim.Optimizer,
    batch: dict,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    epochs: int,
    minibatch_size: int,
    lam: float,
    gamma: float,
    pred: Predictor | None,
    pred_coef: float,
    generator: torch.Generator,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_scaler: amp.GradScaler | None,
) -> dict[str, float]:
    obs = batch["obs"]
    prev_a = batch["prev_action"]
    traces = batch["traces"]
    actions = batch["actions"]
    logp_old = batch["logp_old"]
    values_old = batch["values_old"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    resets = batch["resets"]
    value_T = batch["value_T"]
    x_mem = batch["x_mem"]
    x_mem_next = batch["x_mem_next"]

    adv, returns = compute_gae(rewards, dones, resets, values_old, value_T, gamma=gamma, lam=lam)

    T, N = rewards.shape
    b = T * N

    obs_f = obs.reshape(b, -1)
    prev_a_f = prev_a.reshape(b)
    traces_f = traces.reshape(b, -1)
    actions_f = actions.reshape(b)
    logp_old_f = logp_old.reshape(b)
    values_old_f = values_old.reshape(b)
    adv_f = adv.reshape(b)
    returns_f = returns.reshape(b)
    x_mem_f = x_mem.reshape(b, -1)
    x_mem_next_f = x_mem_next.reshape(b, -1)

    adv_f = (adv_f - adv_f.mean()) / (adv_f.std(unbiased=False) + 1e-8)

    idx = torch.arange(b, device=obs.device)
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "pred_loss": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
        "count": 0.0,
    }
    for _ in range(epochs):
        perm = idx[torch.randperm(b, generator=generator)]
        for start in range(0, b, minibatch_size):
            mb = perm[start : start + minibatch_size]

            with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
                logits, values = ac(obs_f[mb], prev_a_f[mb], traces_f[mb])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions_f[mb])
                entropy = dist.entropy().mean()

                ratio = (logp - logp_old_f[mb]).exp()
                pg1 = ratio * adv_f[mb]
                pg2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv_f[mb]
                policy_loss = -torch.min(pg1, pg2).mean()

                value_loss = 0.5 * (returns_f[mb] - values).pow(2).mean()

                pred_loss = torch.tensor(0.0, device=obs.device)
                if (pred is not None) and (pred_coef > 0.0):
                    x_hat = pred(x_mem_f[mb], actions_f[mb])
                    pred_loss = (x_mem_next_f[mb] - x_hat).pow(2).mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy + pred_coef * pred_loss

            opt.zero_grad(set_to_none=True)
            if grad_scaler is not None and grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(ac.parameters(), max_norm=0.5)
                grad_scaler.step(opt)
                grad_scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_norm=0.5)
                opt.step()

            with torch.no_grad():
                approx_kl = (logp_old_f[mb] - logp).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

            stats["policy_loss"] += policy_loss.detach().item()
            stats["value_loss"] += value_loss.detach().item()
            stats["entropy"] += entropy.detach().item()
            stats["pred_loss"] += pred_loss.detach().item()
            stats["approx_kl"] += approx_kl.detach().item()
            stats["clipfrac"] += clipfrac.detach().item()
            stats["count"] += 1.0

    count = max(stats["count"], 1.0)
    for k in list(stats.keys()):
        if k != "count":
            stats[k] /= count
    stats.pop("count", None)
    return stats


def make_env_fn(
    env_id: str,
    seed: int,
    mask_indices: list[int],
    phase_len: int,
    obs_shift_scale: float,
    reward_scale_low: float,
    reward_scale_high: float,
):
    def _thunk():
        env = gym.make(env_id)
        if len(mask_indices) > 0:
            env = PartialObsWrapper(env, mask_indices)
        if (phase_len > 0) and ((obs_shift_scale > 0.0) or (reward_scale_low != 1.0) or (reward_scale_high != 1.0)):
            env = PiecewiseDriftWrapper(
                env,
                seed=seed,
                phase_len=phase_len,
                obs_shift_scale=obs_shift_scale,
                reward_scale_low=reward_scale_low,
                reward_scale_high=reward_scale_high,
            )
        return env

    return _thunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--feat-dim", type=int, default=64)
    p.add_argument("--act-embed-dim", type=int, default=16)

    p.add_argument("--alpha-base", type=str, default="0.5,0.1,0.01")
    p.add_argument("--alpha-max", type=str, default="1.0,0.5,0.2")

    p.add_argument("--rho-s", type=float, default=0.1)
    p.add_argument("--rho-l", type=float, default=0.01)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--tau-soft", type=float, default=1.0)
    p.add_argument("--kappa", type=float, default=0.5)
    p.add_argument("--tau-on", type=float, default=2.5)
    p.add_argument("--tau-off", type=float, default=1.5)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--cooldown-steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=1000)

    p.add_argument("--reset-strategy", type=str, default="partial", choices=["none", "zero", "obs", "partial"])
    p.add_argument("--reset-long-fraction", type=float, default=0.34)

    p.add_argument("--lambda-pred", type=float, default=0.0)
    p.add_argument("--pred-coef", type=float, default=0.0)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lam", type=float, default=0.95)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--ema-tau", type=float, default=0.995)
    p.add_argument("--log-interval", type=int, default=10)

    p.add_argument("--mask-indices", type=str, default="1,3")
    p.add_argument("--phase-len", type=int, default=2000)
    p.add_argument("--obs-shift-scale", type=float, default=0.1)
    p.add_argument("--reward-scale-low", type=float, default=0.7)
    p.add_argument("--reward-scale-high", type=float, default=1.3)

    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", type=str, default="amt")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-dir", type=str, default=None)

    p.add_argument("--amp", action="store_true", help="Enable mixed precision training (CUDA only).")
    p.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])

    args = p.parse_args()
    set_seed(args.seed)

    device = args.device
    device_type = torch.device(device).type
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    use_amp = args.amp and device_type == "cuda"
    if args.amp and not use_amp:
        print("Warning: --amp requested but device is not CUDA; disabling mixed precision.")
    grad_scaler = amp.GradScaler(enabled=use_amp and (amp_dtype == torch.float16))
    alpha_base_list = parse_floats(args.alpha_base)
    alpha_max_list = parse_floats(args.alpha_max)
    assert len(alpha_base_list) == len(alpha_max_list)
    M = len(alpha_base_list)

    env_fns = [
        make_env_fn(
            env_id=args.env_id,
            seed=args.seed + 10_000 * i,
            mask_indices=parse_ints(args.mask_indices),
            phase_len=args.phase_len,
            obs_shift_scale=args.obs_shift_scale,
            reward_scale_low=args.reward_scale_low,
            reward_scale_high=args.reward_scale_high,
        )
        for i in range(args.num_envs)
    ]
    envs = EnvPool(env_fns)

    obs0, _ = envs.reset(seed=args.seed)
    assert isinstance(envs.single_observation_space, gym.spaces.Box)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(envs.single_action_space.n)
    feat_dim = args.feat_dim
    mem_dim = M * feat_dim

    ac = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=args.act_embed_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=feat_dim,
        mem_dim=mem_dim,
    ).to(device)

    f_mem = FeatureEncoder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_embed_dim=args.act_embed_dim,
        hidden_dim=args.hidden_dim,
        feat_dim=feat_dim,
    ).to(device)
    f_mem.load_state_dict(ac.f_pol.state_dict())

    predictor = None
    if (args.lambda_pred > 0.0) or (args.pred_coef > 0.0):
        predictor = Predictor(feat_dim=feat_dim, act_dim=act_dim, hidden_dim=args.hidden_dim).to(device)

    alpha_base = torch.tensor(alpha_base_list, device=device, dtype=torch.float32).unsqueeze(0)  # (1, M)
    alpha_max = torch.tensor(alpha_max_list, device=device, dtype=torch.float32).unsqueeze(0)

    traces = torch.zeros((args.num_envs, M, feat_dim), device=device)
    prev_action = torch.zeros(args.num_envs, device=device, dtype=torch.int64)

    with torch.no_grad():
        obs0_t = torch.as_tensor(obs0, device=device, dtype=torch.float32)
        x_mem0 = f_mem(obs0_t, prev_action)
        x_mem0 = F.layer_norm(x_mem0, (x_mem0.shape[-1],))
        traces = trace_update(traces, x_mem0, alpha_base.expand(args.num_envs, -1))

    drift = DriftMonitor(
        num_envs=args.num_envs,
        rho_s=args.rho_s,
        rho_l=args.rho_l,
        beta=args.beta,
        tau_soft=args.tau_soft,
        kappa=args.kappa,
        tau_on=args.tau_on,
        tau_off=args.tau_off,
        K=args.K,
        cooldown_steps=args.cooldown_steps,
        warmup_steps=args.warmup_steps,
        device=device,
    )

    params = list(ac.parameters())
    if predictor is not None:
        params += list(predictor.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    updates = args.total_steps // (args.num_envs * args.horizon)

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    obs = obs0
    wandb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed. Run `pip install wandb` or disable --wandb.")
        tags = parse_strs(args.wandb_tags)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=tags or None,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            config=vars(args),
        )

    try:
        for upd in range(updates):
            batch, obs, prev_action, traces = rollout(
                envs=envs,
                ac=ac,
                f_mem=f_mem,
                drift=drift,
                predictor=predictor,
                device=device,
                horizon=args.horizon,
                gamma=args.gamma,
                lambda_pred=args.lambda_pred,
                alpha_base=alpha_base,
                alpha_max=alpha_max,
                reset_strategy=args.reset_strategy,
                reset_long_fraction=args.reset_long_fraction,
                obs=obs,
                prev_action=prev_action,
                traces=traces,
            )

            ppo_stats = ppo_update(
                ac=ac,
                opt=opt,
                batch=batch,
                clip_coef=args.clip_coef,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                epochs=args.epochs,
                minibatch_size=args.minibatch_size,
                lam=args.gae_lam,
                gamma=args.gamma,
                pred=predictor,
                pred_coef=args.pred_coef,
                generator=rng,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                grad_scaler=grad_scaler,
            )

            ema_update_(f_mem, ac.f_pol, tau=args.ema_tau)

            metrics = {
                "loop/update": upd + 1,
                "loop/frames": (upd + 1) * args.num_envs * args.horizon,
                "diagnostics/gate_mean": float(drift.state.gate.mean().item()),
            }
            metrics.update({f"loss/{k}": v for k, v in ppo_stats.items()})

            mean_ret = mean_len = None
            if len(envs.episode_returns) > 0:
                recent = envs.episode_returns[-50:]
                mean_ret = sum(r for r, _ in recent) / len(recent)
                mean_len = sum(l for _, l in recent) / len(recent)
                metrics["train/ret50"] = mean_ret
                metrics["train/len50"] = mean_len

            if (upd + 1) % args.log_interval == 0 and mean_ret is not None:
                print(
                    f"update={upd+1:04d}  episodes={len(envs.episode_returns):06d}  "
                    f"ret50={mean_ret:8.2f}  len50={mean_len:6.1f}  gate_mean={metrics['diagnostics/gate_mean']:6.3f}  "
                    f"kl={ppo_stats['approx_kl']:7.4f}  clipfrac={ppo_stats['clipfrac']:6.3f}"
                )

            if wandb_run is not None:
                wandb_run.log(metrics, step=metrics["loop/frames"])
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
