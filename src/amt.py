from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from src.envs import EnvPool
from src.models import ActorCritic, FeatureEncoder, Predictor, RecurrentActorCritic
from src.utils import obs_to_tensor


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


def encode_mem(f_mem: FeatureEncoder, obs: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    x_mem = f_mem(obs, prev_action)
    return F.layer_norm(x_mem, (x_mem.shape[-1],))


def maybe_reset_traces(
    traces: torch.Tensor,
    reset_event: torch.Tensor,
    x_mem_next: torch.Tensor,
    reset_strategy: str,
    long_mask: torch.Tensor,
) -> torch.Tensor:
    if reset_strategy == "none" or not reset_event.any():
        return traces
    traces = traces.clone()
    traces[reset_event] = apply_reset(traces[reset_event], x_mem_next[reset_event], reset_strategy, long_mask)
    return traces


@torch.no_grad()
def rollout(
    envs: EnvPool,
    ac: ActorCritic,
    f_mem: FeatureEncoder,
    drift: DriftMonitor | None,
    predictor: Predictor | None,
    device: str,
    horizon: int,
    gamma: float,
    lambda_pred: float,
    obs_normalization: str,
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

    skip_drift = (reset_strategy == "none") and torch.allclose(alpha_base, alpha_max)
    alpha_const = alpha_base.expand(n_envs, -1).clamp(0.0, 1.0) if skip_drift else None

    obs_buf = torch.zeros((horizon, n_envs, *obs.shape[1:]), device=device)
    prev_a_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.int64)
    trace_buf = torch.zeros((horizon, n_envs, M, d), device=device)

    act_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.int64)
    logp_buf = torch.zeros((horizon, n_envs), device=device)
    val_buf = torch.zeros((horizon, n_envs), device=device)
    rew_buf = torch.zeros((horizon, n_envs), device=device)
    done_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.bool)
    term_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.bool)
    trunc_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.bool)
    reset_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.bool)

    xmem_buf = torch.zeros((horizon, n_envs, d), device=device)
    xmem_next_buf = torch.zeros((horizon, n_envs, d), device=device)

    long_start = int(math.floor((1.0 - reset_long_fraction) * M))
    long_mask = torch.zeros(M, device=device, dtype=torch.bool)
    if reset_strategy == "partial":
        long_mask[long_start:] = True

    for t in range(horizon):
        obs_t = obs_to_tensor(obs, device=device, obs_normalization=obs_normalization)
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
        term = torch.as_tensor(terminated, device=device, dtype=torch.bool)
        trunc = torch.as_tensor(truncated, device=device, dtype=torch.bool)
        done = torch.as_tensor(done_env, device=device, dtype=torch.bool)
        rew_buf[t] = rew
        term_buf[t] = term
        trunc_buf[t] = trunc
        done_buf[t] = done

        x_mem_t = encode_mem(f_mem, obs_t, prev_action)
        xmem_buf[t] = x_mem_t

        next_obs_t = obs_to_tensor(next_obs, device=device, obs_normalization=obs_normalization)
        next_prev_action = action.clone()
        if done.any():
            next_prev_action[done] = 0

        x_mem_next = encode_mem(f_mem, next_obs_t, next_prev_action)
        xmem_next_buf[t] = x_mem_next

        if skip_drift:
            reset_event = torch.zeros(n_envs, device=device, dtype=torch.bool)
            reset_buf[t] = reset_event
            traces_next = trace_update(traces, x_mem_next, alpha_const)
        else:
            if drift is None:
                raise RuntimeError("Drift monitor is required when adaptive drift is enabled.")
            _, v_next_prov = ac(
                next_obs_t,
                next_prev_action,
                trace_update(traces, x_mem_next, alpha_base.expand(n_envs, -1)).reshape(n_envs, -1),
            )

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

            traces_reset = maybe_reset_traces(traces, reset_event, x_mem_next, reset_strategy, long_mask)
            traces_next = trace_update(traces_reset, x_mem_next, alpha)

        done_mask = done
        if done_mask.any():
            if drift is not None:
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

    obs_T = obs_to_tensor(obs, device=device, obs_normalization=obs_normalization)
    logits_T, value_T = ac(obs_T, prev_action, traces.reshape(n_envs, -1))

    batch = {
        "obs": obs_buf,
        "prev_action": prev_a_buf,
        "traces": trace_buf,
        "actions": act_buf,
        "logp_old": logp_buf,
        "values_old": val_buf,
        "rewards": rew_buf,
        "terminated": term_buf,
        "truncated": trunc_buf,
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


@torch.no_grad()
def rollout_recurrent(
    envs: EnvPool,
    ac: RecurrentActorCritic,
    device: str,
    horizon: int,
    gamma: float,
    obs_normalization: str,
    obs: np.ndarray,
    prev_action: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor],
):
    n_envs = envs.num_envs
    obs_buf = torch.zeros((horizon, n_envs, *obs.shape[1:]), device=device)
    prev_a_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.int64)
    act_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.int64)
    logp_buf = torch.zeros((horizon, n_envs), device=device)
    val_buf = torch.zeros((horizon, n_envs), device=device)
    rew_buf = torch.zeros((horizon, n_envs), device=device)
    done_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.bool)
    term_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.bool)
    trunc_buf = torch.zeros((horizon, n_envs), device=device, dtype=torch.bool)

    h, c = hidden
    h0, c0 = h.clone(), c.clone()

    for t in range(horizon):
        obs_t = obs_to_tensor(obs, device=device, obs_normalization=obs_normalization)
        obs_buf[t] = obs_t
        prev_a_buf[t] = prev_action

        logits, value, (h, c) = ac(obs_t, prev_action, (h, c))
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        act_buf[t] = action
        logp_buf[t] = logp
        val_buf[t] = value

        next_obs, reward, terminated, truncated, _info = envs.step(action.cpu().numpy())
        done = terminated | truncated

        rew_buf[t] = torch.as_tensor(reward, device=device, dtype=torch.float32)
        done_buf[t] = torch.as_tensor(done, device=device, dtype=torch.bool)
        term_buf[t] = torch.as_tensor(terminated, device=device, dtype=torch.bool)
        trunc_buf[t] = torch.as_tensor(truncated, device=device, dtype=torch.bool)

        next_prev_action = action.clone()
        if done.any():
            done_idx = torch.as_tensor(done, device=device)
            h[:, done_idx] = 0.0
            c[:, done_idx] = 0.0
            next_prev_action[done_idx] = 0

        prev_action = next_prev_action
        obs = next_obs

    obs_T = obs_to_tensor(obs, device=device, obs_normalization=obs_normalization)
    # Bootstrap value at obs_T without advancing the carried recurrent state.
    _, value_T, _ = ac(obs_T, prev_action, (h, c))

    batch = {
        "obs": obs_buf,
        "prev_action": prev_a_buf,
        "actions": act_buf,
        "logp_old": logp_buf,
        "values_old": val_buf,
        "rewards": rew_buf,
        "dones": done_buf,
        "terminated": term_buf,
        "truncated": trunc_buf,
        "value_T": value_T,
        "h0": h0,
        "c0": c0,
    }
    return batch, obs, prev_action, (h, c)
