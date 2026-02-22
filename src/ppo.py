import torch
import torch.nn as nn
from torch.distributions import Categorical

from .utils import autocast_context


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    resets: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
):
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        nonterminal = (~dones[t]) & (~resets[t])
        delta = rewards[t] + gamma * nonterminal * last_value - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
        last_value = values[t]
    returns = adv + values
    return adv, returns


def ppo_update(
    ac,
    opt: torch.optim.Optimizer,
    batch: dict,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    epochs: int,
    minibatch_size: int,
    lam: float,
    gamma: float,
    pred,
    pred_coef: float,
    generator: torch.Generator,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_scaler: torch.amp.GradScaler | None,
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

            stats["policy_loss"] += policy_loss.item() * len(mb)
            stats["value_loss"] += value_loss.item() * len(mb)
            stats["entropy"] += entropy.item() * len(mb)
            stats["pred_loss"] += pred_loss.item() * len(mb)
            stats["approx_kl"] += approx_kl.item() * len(mb)
            stats["clipfrac"] += clipfrac.item() * len(mb)
            stats["count"] += len(mb)

    for k in stats:
        if k != "count":
            stats[k] /= stats["count"]
    return stats


def ppo_update_recurrent(
    ac,
    opt: torch.optim.Optimizer,
    batch: dict,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    epochs: int,
    lam: float,
    gamma: float,
    generator: torch.Generator,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    obs = batch["obs"]  # [T, N, obs_dim]
    prev_a = batch["prev_action"]
    actions = batch["actions"]
    logp_old = batch["logp_old"]
    values_old = batch["values_old"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    h0 = batch["h0"]
    c0 = batch["c0"]
    value_T = batch["value_T"]

    resets = torch.zeros_like(dones)
    adv, returns = compute_gae(rewards, dones, resets, values_old, value_T, gamma=gamma, lam=lam)
    T, N = rewards.shape

    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    env_idx = torch.arange(N, device=obs.device)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0, "count": 0.0}

    for _ in range(epochs):
        perm_env = env_idx[torch.randperm(N, generator=generator)]
        for env_id in perm_env:
            h = h0[:, env_id : env_id + 1].detach()
            c = c0[:, env_id : env_id + 1].detach()

            traj_logits = []
            traj_values = []
            traj_logp = []
            traj_entropy = []

            for t in range(T):
                with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
                    logits, value, (h, c) = ac(obs[t, env_id : env_id + 1], prev_a[t, env_id : env_id + 1], (h, c))
                    dist = Categorical(logits=logits)
                    lp = dist.log_prob(actions[t, env_id : env_id + 1]).squeeze(-1)
                    traj_logits.append(logits)
                    traj_values.append(value)
                    traj_logp.append(lp)
                    traj_entropy.append(dist.entropy().mean())

            logits_all = torch.cat(traj_logits, dim=0)
            values_all = torch.cat(traj_values, dim=0).squeeze(-1)
            logp_all = torch.stack(traj_logp, dim=0)
            entropy = torch.stack(traj_entropy, dim=0).mean()

            ratio = (logp_all - logp_old[:, env_id]).exp()
            pg1 = ratio * adv[:, env_id]
            pg2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv[:, env_id]
            policy_loss = -torch.min(pg1, pg2).mean()

            value_loss = 0.5 * (returns[:, env_id] - values_all).pow(2).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

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
                approx_kl = (logp_old[:, env_id] - logp_all).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

            L = T
            stats["policy_loss"] += policy_loss.item() * L
            stats["value_loss"] += value_loss.item() * L
            stats["entropy"] += entropy.item() * L
            stats["approx_kl"] += approx_kl.item() * L
            stats["clipfrac"] += clipfrac.item() * L
            stats["count"] += L

    for k in stats:
        if k != "count":
            stats[k] /= stats["count"]
    return stats
