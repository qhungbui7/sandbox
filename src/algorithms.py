from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from src.action_utils import evaluate_policy_actions
from src.amt import DriftMonitor, encode_mem, maybe_reset_traces, trace_update
from src.envs import EnvPool
from src.models import ActorCritic, FeatureEncoder, Predictor
from src.utils import autocast_context, obs_to_tensor

_ALGO_ALIASES = {
    "vtrace": "v-trace",
    "v_trace": "v-trace",
    "vmpo": "v-mpo",
    "v_mpo": "v-mpo",
}

ON_POLICY_ALGOS = {"ppo", "a2c", "trpo", "reinforce", "v-trace", "v-mpo"}
ALL_ALGOS = ON_POLICY_ALGOS | {"dqn"}


def normalize_algo_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    return _ALGO_ALIASES.get(key, key)


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
        trunc = (~resets[t]).float()
        delta = rewards[t] + gamma * nonterminal * trunc * next_value - values[t]
        last_gae = delta + gamma * lam * nonterminal * trunc * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def _discounted_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    T, N = rewards.shape
    out = torch.zeros_like(rewards)
    ret = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        nonterminal = (~dones[t]).float()
        ret = rewards[t] + gamma * nonterminal * ret
        out[t] = ret
    return out


def _normalize_adv(adv: torch.Tensor) -> torch.Tensor:
    return (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)


def _flatten_rollout(batch: dict, adv: torch.Tensor, returns: torch.Tensor) -> dict[str, torch.Tensor | int | None]:
    T, N = batch["rewards"].shape
    b = T * N
    prev_action = batch.get("prev_action", None)
    traces = batch.get("traces", None)
    x_mem = batch.get("x_mem", None)
    x_mem_next = batch.get("x_mem_next", None)
    actions = batch["actions"]
    if prev_action is None:
        prev_a_f = None
    elif prev_action.ndim == 2:
        prev_a_f = prev_action.reshape(b)
    else:
        prev_a_f = prev_action.reshape(b, -1)
    if actions.ndim == 2:
        actions_f = actions.reshape(b)
    else:
        actions_f = actions.reshape(b, -1)
    return {
        "T": T,
        "N": N,
        "B": b,
        "obs_f": batch["obs"].reshape((b, *batch["obs"].shape[2:])),
        "prev_a_f": prev_a_f,
        "traces_f": (traces.reshape(b, -1) if traces is not None else None),
        "actions_f": actions_f,
        "logp_old_f": batch["logp_old"].reshape(b),
        "values_old_f": batch["values_old"].reshape(b),
        "adv_f": _normalize_adv(adv.reshape(b)),
        "returns_f": returns.reshape(b),
        "x_mem_f": (x_mem.reshape(b, -1) if x_mem is not None else None),
        "x_mem_next_f": (x_mem_next.reshape(b, -1) if x_mem_next is not None else None),
    }


def _step_optimizer(
    loss: torch.Tensor,
    opt: torch.optim.Optimizer,
    model: nn.Module,
    grad_scaler: torch.amp.GradScaler | None,
    max_grad_norm: float,
    grad_modules: dict[str, nn.Module] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    module_norms: dict[str, torch.Tensor] = {}
    opt.zero_grad(set_to_none=True)
    if grad_scaler is not None and grad_scaler.is_enabled():
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(opt)
        global_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        if grad_modules is not None:
            for name, module in grad_modules.items():
                sq = torch.zeros((), device=global_norm.device)
                for param in module.parameters():
                    if param.grad is not None:
                        sq = sq + param.grad.detach().float().pow(2).sum()
                module_norms[name] = torch.sqrt(sq)
        grad_scaler.step(opt)
        grad_scaler.update()
    else:
        loss.backward()
        global_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        if grad_modules is not None:
            for name, module in grad_modules.items():
                sq = torch.zeros((), device=global_norm.device)
                for param in module.parameters():
                    if param.grad is not None:
                        sq = sq + param.grad.detach().float().pow(2).sum()
                module_norms[name] = torch.sqrt(sq)
        opt.step()
    return global_norm.detach().float(), module_norms


def _update_tensor_reservoir(
    reservoir: torch.Tensor,
    filled: int,
    values: torch.Tensor,
    *,
    generator: torch.Generator,
) -> int:
    if values.numel() == 0:
        return filled

    capacity = int(reservoir.numel())
    chunk = values.reshape(-1)
    if filled < capacity:
        take = min(capacity - filled, int(chunk.numel()))
        reservoir[filled : filled + take] = chunk[:take]
        filled += take
        if take == int(chunk.numel()):
            return filled
        chunk = chunk[take:]

    merged = torch.cat([reservoir[:filled], chunk], dim=0)
    if int(merged.numel()) <= capacity:
        reservoir[: int(merged.numel())] = merged
        return int(merged.numel())

    select = torch.randperm(int(merged.numel()), generator=generator, device=merged.device)[:capacity]
    reservoir[:] = merged[select]
    return capacity


def _pred_loss(
    pred: Predictor | None,
    pred_coef: float,
    x_mem: torch.Tensor | None,
    actions: torch.Tensor,
    x_mem_next: torch.Tensor | None,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if x_mem is not None:
        out_device = x_mem.device
    elif x_mem_next is not None:
        out_device = x_mem_next.device
    elif device is not None:
        out_device = device
    else:
        out_device = "cpu"
    if pred is None or pred_coef <= 0.0:
        return torch.zeros((), device=out_device)
    if x_mem is None or x_mem_next is None:
        raise RuntimeError("Predictor loss requires trace memory features (x_mem/x_mem_next).")
    x_hat = pred(x_mem, actions)
    return (x_mem_next - x_hat).pow(2).mean()


def ppo_update(
    ac: ActorCritic,
    opt: torch.optim.Optimizer,
    batch: dict,
    clip_coef: float,
    vf_clip: bool,
    target_kl: float,
    vf_coef: float,
    max_grad_norm: float,
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
    grad_scaler: torch.amp.GradScaler | None,
    debug_cfg: dict | None = None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> dict[str, float]:
    action_mode = str(getattr(getattr(ac, "f_pol", None), "action_type", "discrete"))
    bootstrap_stops = batch["dones"]
    adv, returns = compute_gae(
        batch["rewards"],
        bootstrap_stops,
        batch["resets"],
        batch["values_old"],
        batch["value_T"],
        gamma=gamma,
        lam=lam,
    )
    flat = _flatten_rollout(batch, adv, returns)
    idx = torch.arange(flat["B"], device=batch["obs"].device)

    sum_policy = torch.zeros((), device=idx.device)
    sum_value = torch.zeros((), device=idx.device)
    sum_entropy = torch.zeros((), device=idx.device)
    sum_pred = torch.zeros((), device=idx.device)
    sum_approx_kl = torch.zeros((), device=idx.device)
    sum_clipfrac = torch.zeros((), device=idx.device)
    mb_count = 0

    debug_enabled = debug_cfg is not None
    if debug_enabled:
        action_bins = int(debug_cfg["action_bins"]) if action_mode == "discrete" else 0
        ratio_cap = int(debug_cfg["ratio_sample_size"])
        delta_pairs = int(debug_cfg["frame_delta_pairs"])
        seed = int(debug_cfg["seed"])
        update_idx = int(debug_cfg["update_idx"])
        debug_gen = torch.Generator(device=idx.device)
        debug_gen.manual_seed(seed * 1_000_003 + update_idx)

        obs_f = flat["obs_f"]
        obs_min = obs_f.amin()
        obs_max = obs_f.amax()
        obs_mean = obs_f.mean()
        obs_std = obs_f.std(unbiased=False)
        obs_zero_frac = (obs_f == 0.0).float().mean()
        obs_max_frac = (obs_f == obs_max).float().mean()

        obs_seq = batch["obs"]
        total_pairs = int((obs_seq.shape[0] - 1) * obs_seq.shape[1]) if int(obs_seq.shape[0]) > 1 else 0
        if total_pairs > 0:
            take_pairs = min(delta_pairs, total_pairs)
            pair_ids = torch.randperm(total_pairs, generator=debug_gen, device=obs_seq.device)[:take_pairs]
            n_envs = int(obs_seq.shape[1])
            t_idx = torch.div(pair_ids, n_envs, rounding_mode="floor")
            n_idx = pair_ids % n_envs
            frame_delta_mean = (obs_seq[t_idx + 1, n_idx] - obs_seq[t_idx, n_idx]).abs().mean()
        else:
            frame_delta_mean = torch.full((), float("nan"), device=obs_seq.device)

        td_error = flat["returns_f"] - flat["values_old_f"]
        var_returns = flat["returns_f"].var(unbiased=False)
        explained_var = 1.0 - (td_error.var(unbiased=False) / (var_returns + 1e-8))

        action_hist = (
            torch.bincount(flat["actions_f"].long(), minlength=action_bins).float()[:action_bins]
            if action_mode == "discrete"
            else None
        )
        ratio_reservoir = torch.zeros((ratio_cap,), device=idx.device)
        ratio_filled = 0

        sum_total = torch.zeros((), device=idx.device)
        max_approx_kl = torch.full((), float("-inf"), device=idx.device)
        sum_max_action_prob = torch.zeros((), device=idx.device)
        sum_grad_global = torch.zeros((), device=idx.device)
        max_grad_global = torch.zeros((), device=idx.device)
        sum_grad_encoder = torch.zeros((), device=idx.device)
        sum_grad_core = torch.zeros((), device=idx.device)
        sum_grad_pi = torch.zeros((), device=idx.device)
        sum_grad_v = torch.zeros((), device=idx.device)
        pi_module = ac.pi if hasattr(ac, "pi") else ac.pi_mean
        grad_modules = {
            "encoder": ac.f_pol.obs_encoder,
            "core": ac.core,
            "pi": pi_module,
            "v": ac.v,
        }
    else:
        action_hist = None
        ratio_reservoir = None
        ratio_filled = 0
        grad_modules = None

    stop_due_to_kl = False
    kl_early_stop = target_kl > 0.0
    for _ in range(epochs):
        perm = idx[torch.randperm(flat["B"], generator=generator, device=idx.device)]
        for start in range(0, flat["B"], minibatch_size):
            mb = perm[start : start + minibatch_size]
            with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
                prev_a_mb = flat["prev_a_f"][mb] if flat["prev_a_f"] is not None else None
                traces_mb = flat["traces_f"][mb] if flat["traces_f"] is not None else None
                policy_out, values = ac(flat["obs_f"][mb], prev_a_mb, traces_mb)
                logp, entropy, max_action_prob = evaluate_policy_actions(
                    policy_out=policy_out,
                    actions=flat["actions_f"][mb],
                    action_mode=action_mode,
                    action_low=action_low,
                    action_high=action_high,
                )

                log_ratio = logp - flat["logp_old_f"][mb]
                ratio = log_ratio.exp()
                pg1 = ratio * flat["adv_f"][mb]
                pg2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * flat["adv_f"][mb]
                policy_loss = -torch.min(pg1, pg2).mean()
                if vf_clip:
                    value_delta = values - flat["values_old_f"][mb]
                    values_clipped = flat["values_old_f"][mb] + torch.clamp(value_delta, -clip_coef, clip_coef)
                    value_err = (flat["returns_f"][mb] - values).pow(2)
                    value_err_clipped = (flat["returns_f"][mb] - values_clipped).pow(2)
                    value_loss = 0.5 * torch.maximum(value_err, value_err_clipped).mean()
                else:
                    value_loss = 0.5 * (flat["returns_f"][mb] - values).pow(2).mean()
                pred_loss = _pred_loss(
                    pred=pred,
                    pred_coef=pred_coef,
                    x_mem=(flat["x_mem_f"][mb] if flat["x_mem_f"] is not None else None),
                    actions=flat["actions_f"][mb],
                    x_mem_next=(flat["x_mem_next_f"][mb] if flat["x_mem_next_f"] is not None else None),
                    device=idx.device,
                )
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy + pred_coef * pred_loss

            global_grad_norm, module_grad_norms = _step_optimizer(
                loss=loss,
                opt=opt,
                model=ac,
                grad_scaler=grad_scaler,
                max_grad_norm=max_grad_norm,
                grad_modules=grad_modules,
            )
            with torch.no_grad():
                approx_kl = (flat["logp_old_f"][mb] - logp).mean()
                kl_for_stop = (ratio - 1.0 - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
                sum_policy = sum_policy + policy_loss.detach().float()
                sum_value = sum_value + value_loss.detach().float()
                sum_entropy = sum_entropy + entropy.detach().float()
                sum_pred = sum_pred + pred_loss.detach().float()
                sum_approx_kl = sum_approx_kl + approx_kl.detach().float()
                sum_clipfrac = sum_clipfrac + clipfrac.detach().float()
                mb_count += 1

                if debug_enabled:
                    sum_total = sum_total + loss.detach().float()
                    max_approx_kl = torch.maximum(max_approx_kl, approx_kl.detach().float())
                    sum_max_action_prob = sum_max_action_prob + max_action_prob
                    sum_grad_global = sum_grad_global + global_grad_norm
                    max_grad_global = torch.maximum(max_grad_global, global_grad_norm)
                    sum_grad_encoder = sum_grad_encoder + module_grad_norms["encoder"]
                    sum_grad_core = sum_grad_core + module_grad_norms["core"]
                    sum_grad_pi = sum_grad_pi + module_grad_norms["pi"]
                    sum_grad_v = sum_grad_v + module_grad_norms["v"]
                    ratio_filled = _update_tensor_reservoir(
                        ratio_reservoir,
                        ratio_filled,
                        ratio.detach().float(),
                        generator=debug_gen,
                    )

                if kl_early_stop and float(kl_for_stop.detach().item()) > target_kl:
                    stop_due_to_kl = True
                    break
        if stop_due_to_kl:
            break

    count = max(mb_count, 1)
    inv_count = 1.0 / float(count)
    stats: dict[str, float] = {
        "policy_loss": float((sum_policy * inv_count).item()),
        "value_loss": float((sum_value * inv_count).item()),
        "entropy": float((sum_entropy * inv_count).item()),
        "pred_loss": float((sum_pred * inv_count).item()),
        "approx_kl": float((sum_approx_kl * inv_count).item()),
        "clipfrac": float((sum_clipfrac * inv_count).item()),
    }

    if debug_enabled:
        ratio_sample = ratio_reservoir[:ratio_filled].detach().float().cpu()
        ratio_q = torch.quantile(ratio_sample, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float32))
        td_std = td_error.std(unbiased=False)
        stats.update(
            {
                "debug/obs_dtype": str(obs_f.dtype),
                "debug/obs_min": float(obs_min.item()),
                "debug/obs_max": float(obs_max.item()),
                "debug/obs_mean": float(obs_mean.item()),
                "debug/obs_std": float(obs_std.item()),
                "debug/obs_fraction_zero": float(obs_zero_frac.item()),
                "debug/obs_fraction_max": float(obs_max_frac.item()),
                "debug/frame_delta_mean": float(frame_delta_mean.item()),
                "debug/ppo/policy_loss_mean": float((sum_policy * inv_count).item()),
                "debug/ppo/value_loss_mean": float((sum_value * inv_count).item()),
                "debug/ppo/entropy_mean": float((sum_entropy * inv_count).item()),
                "debug/ppo/entropy_loss_mean": float((-ent_coef * sum_entropy * inv_count).item()),
                "debug/ppo/pred_loss_mean": float((sum_pred * inv_count).item()),
                "debug/ppo/total_loss_mean": float((sum_total * inv_count).item()),
                "debug/ppo/approx_kl_mean": float((sum_approx_kl * inv_count).item()),
                "debug/ppo/approx_kl_max": float(max_approx_kl.item()),
                "debug/ppo/clipfrac_mean": float((sum_clipfrac * inv_count).item()),
                "debug/ppo/ratio_p01": float(ratio_q[0].item()),
                "debug/ppo/ratio_p50": float(ratio_q[1].item()),
                "debug/ppo/ratio_p99": float(ratio_q[2].item()),
                "debug/ppo/grad_norm_global_mean": float((sum_grad_global * inv_count).item()),
                "debug/ppo/grad_norm_global_max": float(max_grad_global.item()),
                "debug/ppo/grad_norm_encoder_mean": float((sum_grad_encoder * inv_count).item()),
                "debug/ppo/grad_norm_core_mean": float((sum_grad_core * inv_count).item()),
                "debug/ppo/grad_norm_pi_mean": float((sum_grad_pi * inv_count).item()),
                "debug/ppo/grad_norm_v_mean": float((sum_grad_v * inv_count).item()),
                "debug/value/explained_variance": float(explained_var.item()),
                "debug/value/td_error_mean": float(td_error.mean().item()),
                "debug/value/td_error_std": float(td_std.item()),
                "debug/action/max_action_prob_mean": float((sum_max_action_prob * inv_count).item())
                if action_mode == "discrete"
                else float("nan"),
                "debug/ppo/mb_count": float(count),
                "debug/ppo/ratio_sample_count": float(ratio_filled),
            }
        )
        if action_mode == "discrete":
            assert action_hist is not None
            for action_idx in range(action_bins):
                stats[f"debug/action/hist_{action_idx}"] = float(action_hist[action_idx].item())

    return stats


def a2c_update(
    ac: ActorCritic,
    opt: torch.optim.Optimizer,
    batch: dict,
    vf_coef: float,
    max_grad_norm: float,
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
    grad_scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    bootstrap_stops = batch["dones"]
    adv, returns = compute_gae(
        batch["rewards"],
        bootstrap_stops,
        batch["resets"],
        batch["values_old"],
        batch["value_T"],
        gamma=gamma,
        lam=lam,
    )
    flat = _flatten_rollout(batch, adv, returns)
    idx = torch.arange(flat["B"], device=batch["obs"].device)

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
        perm = idx[torch.randperm(flat["B"], generator=generator, device=idx.device)]
        for start in range(0, flat["B"], minibatch_size):
            mb = perm[start : start + minibatch_size]
            with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
                logits, values = ac(flat["obs_f"][mb], flat["prev_a_f"][mb], flat["traces_f"][mb])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(flat["actions_f"][mb])
                entropy = dist.entropy().mean()

                policy_loss = -(logp * flat["adv_f"][mb].detach()).mean()
                value_loss = 0.5 * (flat["returns_f"][mb] - values).pow(2).mean()
                pred_loss = _pred_loss(
                    pred=pred,
                    pred_coef=pred_coef,
                    x_mem=flat["x_mem_f"][mb],
                    actions=flat["actions_f"][mb],
                    x_mem_next=flat["x_mem_next_f"][mb],
                )
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy + pred_coef * pred_loss

            _step_optimizer(loss=loss, opt=opt, model=ac, grad_scaler=grad_scaler, max_grad_norm=max_grad_norm)
            with torch.no_grad():
                approx_kl = (flat["logp_old_f"][mb] - logp).mean()

            stats["policy_loss"] += policy_loss.detach().item()
            stats["value_loss"] += value_loss.detach().item()
            stats["entropy"] += entropy.detach().item()
            stats["pred_loss"] += pred_loss.detach().item()
            stats["approx_kl"] += approx_kl.detach().item()
            stats["count"] += 1.0

    count = max(stats["count"], 1.0)
    for k in list(stats.keys()):
        if k != "count":
            stats[k] /= count
    stats.pop("count", None)
    return stats


def reinforce_update(
    ac: ActorCritic,
    opt: torch.optim.Optimizer,
    batch: dict,
    vf_coef: float,
    max_grad_norm: float,
    ent_coef: float,
    epochs: int,
    minibatch_size: int,
    gamma: float,
    pred: Predictor | None,
    pred_coef: float,
    generator: torch.Generator,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    mc_returns = _discounted_returns(batch["rewards"], batch["dones"], gamma=gamma)
    adv = mc_returns - batch["values_old"]
    flat = _flatten_rollout(batch, adv, mc_returns)
    idx = torch.arange(flat["B"], device=batch["obs"].device)

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
        perm = idx[torch.randperm(flat["B"], generator=generator, device=idx.device)]
        for start in range(0, flat["B"], minibatch_size):
            mb = perm[start : start + minibatch_size]
            with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
                logits, values = ac(flat["obs_f"][mb], flat["prev_a_f"][mb], flat["traces_f"][mb])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(flat["actions_f"][mb])
                entropy = dist.entropy().mean()

                policy_loss = -(logp * flat["adv_f"][mb].detach()).mean()
                value_loss = 0.5 * (flat["returns_f"][mb] - values).pow(2).mean()
                pred_loss = _pred_loss(
                    pred=pred,
                    pred_coef=pred_coef,
                    x_mem=flat["x_mem_f"][mb],
                    actions=flat["actions_f"][mb],
                    x_mem_next=flat["x_mem_next_f"][mb],
                )
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy + pred_coef * pred_loss

            _step_optimizer(loss=loss, opt=opt, model=ac, grad_scaler=grad_scaler, max_grad_norm=max_grad_norm)
            with torch.no_grad():
                approx_kl = (flat["logp_old_f"][mb] - logp).mean()

            stats["policy_loss"] += policy_loss.detach().item()
            stats["value_loss"] += value_loss.detach().item()
            stats["entropy"] += entropy.detach().item()
            stats["pred_loss"] += pred_loss.detach().item()
            stats["approx_kl"] += approx_kl.detach().item()
            stats["count"] += 1.0

    count = max(stats["count"], 1.0)
    for k in list(stats.keys()):
        if k != "count":
            stats[k] /= count
    stats.pop("count", None)
    return stats


def _flatten_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.reshape(-1) for p in module.parameters()])


def _set_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].view_as(p))
        offset += n


def trpo_update(
    ac: ActorCritic,
    opt: torch.optim.Optimizer,
    batch: dict,
    vf_coef: float,
    max_grad_norm: float,
    ent_coef: float,
    lam: float,
    gamma: float,
    pred: Predictor | None,
    pred_coef: float,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_scaler: torch.amp.GradScaler | None,
    max_kl: float,
    backtrack_coef: float,
    backtrack_iters: int,
    value_epochs: int,
) -> dict[str, float]:
    bootstrap_stops = batch["dones"]
    adv, returns = compute_gae(
        batch["rewards"],
        bootstrap_stops,
        batch["resets"],
        batch["values_old"],
        batch["value_T"],
        gamma=gamma,
        lam=lam,
    )
    flat = _flatten_rollout(batch, adv, returns)
    params = tuple(ac.parameters())

    with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
        logits, _ = ac(flat["obs_f"], flat["prev_a_f"], flat["traces_f"])
        dist = Categorical(logits=logits)
        logp = dist.log_prob(flat["actions_f"])
        entropy = dist.entropy().mean()
        ratio = (logp - flat["logp_old_f"]).exp()
        surrogate = (ratio * flat["adv_f"]).mean()
        objective = surrogate + ent_coef * entropy
        policy_loss = -objective

    grads = torch.autograd.grad(policy_loss, params, allow_unused=True)
    grad_flat = []
    for g, p in zip(grads, params, strict=True):
        if g is None:
            grad_flat.append(torch.zeros_like(p).reshape(-1))
        else:
            grad_flat.append(g.reshape(-1))
    grad_flat = torch.cat(grad_flat)
    grad_norm = torch.linalg.vector_norm(grad_flat)

    old_params = _flatten_params(ac)
    accepted_kl = torch.tensor(0.0, device=old_params.device)
    accepted_obj = objective.detach()
    accepted_entropy = entropy.detach()

    if grad_norm > 1e-10:
        step_dir = -grad_flat
        step_scale = math.sqrt(2.0 * max_kl / (step_dir.pow(2).sum().item() + 1e-8))
        full_step = step_scale * step_dir
        base_obj = objective.detach()
        accepted = False
        for i in range(max(backtrack_iters, 1)):
            frac = backtrack_coef**i
            candidate = old_params + frac * full_step
            with torch.no_grad():
                _set_flat_params(ac, candidate)
                logits_new, _ = ac(flat["obs_f"], flat["prev_a_f"], flat["traces_f"])
                dist_new = Categorical(logits=logits_new)
                logp_new = dist_new.log_prob(flat["actions_f"])
                entropy_new = dist_new.entropy().mean()
                ratio_new = (logp_new - flat["logp_old_f"]).exp()
                obj_new = (ratio_new * flat["adv_f"]).mean() + ent_coef * entropy_new
                kl_new = (flat["logp_old_f"] - logp_new).mean()
                if torch.isfinite(obj_new) and torch.isfinite(kl_new) and (kl_new <= max_kl) and (obj_new > base_obj):
                    accepted = True
                    accepted_kl = kl_new
                    accepted_obj = obj_new
                    accepted_entropy = entropy_new
                    break
        if not accepted:
            with torch.no_grad():
                _set_flat_params(ac, old_params)

    value_loss_acc = 0.0
    pred_loss_acc = 0.0
    n_value_steps = max(value_epochs, 1)
    for _ in range(n_value_steps):
        with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
            _, values = ac(flat["obs_f"], flat["prev_a_f"], flat["traces_f"])
            value_loss = 0.5 * (flat["returns_f"] - values).pow(2).mean()
            pred_loss = _pred_loss(
                pred=pred,
                pred_coef=pred_coef,
                x_mem=flat["x_mem_f"],
                actions=flat["actions_f"],
                x_mem_next=flat["x_mem_next_f"],
            )
            loss = vf_coef * value_loss + pred_coef * pred_loss
        _step_optimizer(loss=loss, opt=opt, model=ac, grad_scaler=grad_scaler, max_grad_norm=max_grad_norm)
        value_loss_acc += value_loss.detach().item()
        pred_loss_acc += pred_loss.detach().item()

    return {
        "policy_loss": float((-accepted_obj).item()),
        "value_loss": value_loss_acc / n_value_steps,
        "entropy": float(accepted_entropy.item()),
        "pred_loss": pred_loss_acc / n_value_steps,
        "approx_kl": float(accepted_kl.item()),
        "clipfrac": 0.0,
    }


@torch.no_grad()
def _vtrace_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    value_T: torch.Tensor,
    rho: torch.Tensor,
    c: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape
    vtrace = torch.zeros_like(values)
    pg_adv = torch.zeros_like(values)
    v_next = value_T
    for t in reversed(range(T)):
        nonterminal = (~dones[t]).float()
        v_boot = value_T if t == T - 1 else values[t + 1]
        delta = rho[t] * (rewards[t] + gamma * nonterminal * v_boot - values[t])
        v_t = values[t] + delta + gamma * nonterminal * c[t] * (v_next - v_boot)
        vtrace[t] = v_t
        v_next = v_t

    for t in range(T):
        nonterminal = (~dones[t]).float()
        v_tp1 = value_T if t == T - 1 else vtrace[t + 1]
        pg_adv[t] = rho[t] * (rewards[t] + gamma * nonterminal * v_tp1 - values[t])

    return vtrace, pg_adv


def vtrace_update(
    ac: ActorCritic,
    opt: torch.optim.Optimizer,
    batch: dict,
    vf_coef: float,
    max_grad_norm: float,
    ent_coef: float,
    epochs: int,
    gamma: float,
    pred: Predictor | None,
    pred_coef: float,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_scaler: torch.amp.GradScaler | None,
    rho_clip: float,
    c_clip: float,
) -> dict[str, float]:
    T, N = batch["rewards"].shape
    b = T * N

    obs_f = batch["obs"].reshape(b, -1)
    prev_a_f = batch["prev_action"].reshape(b)
    traces_f = batch["traces"].reshape(b, -1)
    actions_f = batch["actions"].reshape(b)
    x_mem_f = batch["x_mem"].reshape(b, -1)
    x_mem_next_f = batch["x_mem_next"].reshape(b, -1)

    logp_old = batch["logp_old"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    value_T = batch["value_T"]

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
        with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
            logits_f, values_f = ac(obs_f, prev_a_f, traces_f)
            dist_f = Categorical(logits=logits_f)
            logp_f = dist_f.log_prob(actions_f)
            entropy = dist_f.entropy().mean()

            logp = logp_f.reshape(T, N)
            values = values_f.reshape(T, N)
            rho = torch.clamp((logp - logp_old).exp(), max=rho_clip)
            c = torch.clamp((logp - logp_old).exp(), max=c_clip)
            vtarg, pg_adv = _vtrace_targets(
                rewards=rewards,
                dones=dones,
                values=values.detach(),
                value_T=value_T.detach(),
                rho=rho.detach(),
                c=c.detach(),
                gamma=gamma,
            )

            policy_loss = -(logp * pg_adv.detach()).mean()
            value_loss = 0.5 * (vtarg.detach() - values).pow(2).mean()
            pred_loss = _pred_loss(
                pred=pred,
                pred_coef=pred_coef,
                x_mem=x_mem_f,
                actions=actions_f,
                x_mem_next=x_mem_next_f,
            )
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy + pred_coef * pred_loss

        _step_optimizer(loss=loss, opt=opt, model=ac, grad_scaler=grad_scaler, max_grad_norm=max_grad_norm)
        with torch.no_grad():
            approx_kl = (logp_old.reshape(-1) - logp_f).mean()
            clipfrac = ((logp - logp_old).exp() > rho_clip).float().mean()

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


def vmpo_update(
    ac: ActorCritic,
    opt: torch.optim.Optimizer,
    batch: dict,
    vf_coef: float,
    max_grad_norm: float,
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
    grad_scaler: torch.amp.GradScaler | None,
    topk_frac: float,
    eta: float,
    kl_coef: float,
    kl_target: float,
) -> dict[str, float]:
    bootstrap_stops = batch["dones"]
    adv, returns = compute_gae(
        batch["rewards"],
        bootstrap_stops,
        batch["resets"],
        batch["values_old"],
        batch["value_T"],
        gamma=gamma,
        lam=lam,
    )
    flat = _flatten_rollout(batch, adv, returns)
    idx = torch.arange(flat["B"], device=batch["obs"].device)
    topk = max(1, int(max(min(topk_frac, 1.0), 1e-3) * flat["B"]))
    eta_safe = max(eta, 1e-6)

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
        perm = idx[torch.randperm(flat["B"], generator=generator, device=idx.device)]
        for start in range(0, flat["B"], minibatch_size):
            mb = perm[start : start + minibatch_size]
            with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
                logits, values = ac(flat["obs_f"][mb], flat["prev_a_f"][mb], flat["traces_f"][mb])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(flat["actions_f"][mb])
                entropy = dist.entropy().mean()

                adv_mb = flat["adv_f"][mb].detach()
                k = min(topk, adv_mb.shape[0])
                _, top_idx = torch.topk(adv_mb, k=k, largest=True, sorted=False)
                weights = torch.softmax(adv_mb[top_idx] / eta_safe, dim=0).detach()
                policy_loss = -(weights * logp[top_idx]).sum()
                value_loss = 0.5 * (flat["returns_f"][mb] - values).pow(2).mean()
                pred_loss = _pred_loss(
                    pred=pred,
                    pred_coef=pred_coef,
                    x_mem=flat["x_mem_f"][mb],
                    actions=flat["actions_f"][mb],
                    x_mem_next=flat["x_mem_next_f"][mb],
                )
                approx_kl = (flat["logp_old_f"][mb] - logp).mean()
                kl_penalty = F.relu(approx_kl - kl_target)
                loss = (
                    policy_loss
                    + vf_coef * value_loss
                    - ent_coef * entropy
                    + pred_coef * pred_loss
                    + kl_coef * kl_penalty
                )

            _step_optimizer(loss=loss, opt=opt, model=ac, grad_scaler=grad_scaler, max_grad_norm=max_grad_norm)

            stats["policy_loss"] += policy_loss.detach().item()
            stats["value_loss"] += value_loss.detach().item()
            stats["entropy"] += entropy.detach().item()
            stats["pred_loss"] += pred_loss.detach().item()
            stats["approx_kl"] += approx_kl.detach().item()
            stats["count"] += 1.0

    count = max(stats["count"], 1.0)
    for k in list(stats.keys()):
        if k != "count":
            stats[k] /= count
    stats.pop("count", None)
    return stats


def update_on_policy(
    algo: str,
    ac: ActorCritic,
    opt: torch.optim.Optimizer,
    batch: dict,
    clip_coef: float,
    vf_clip: bool,
    target_kl: float,
    vf_coef: float,
    max_grad_norm: float,
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
    grad_scaler: torch.amp.GradScaler | None,
    trpo_max_kl: float,
    trpo_backtrack_coef: float,
    trpo_backtrack_iters: int,
    trpo_value_epochs: int,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
    vmpo_topk_frac: float,
    vmpo_eta: float,
    vmpo_kl_coef: float,
    vmpo_kl_target: float,
    debug_cfg: dict | None = None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> dict[str, float]:
    algo_name = normalize_algo_name(algo)
    if algo_name == "ppo":
        return ppo_update(
            ac=ac,
            opt=opt,
            batch=batch,
            clip_coef=clip_coef,
            vf_clip=vf_clip,
            target_kl=target_kl,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            ent_coef=ent_coef,
            epochs=epochs,
            minibatch_size=minibatch_size,
            lam=lam,
            gamma=gamma,
            pred=pred,
            pred_coef=pred_coef,
            generator=generator,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            debug_cfg=debug_cfg,
            action_low=action_low,
            action_high=action_high,
        )
    if algo_name == "a2c":
        return a2c_update(
            ac=ac,
            opt=opt,
            batch=batch,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            ent_coef=ent_coef,
            epochs=epochs,
            minibatch_size=minibatch_size,
            lam=lam,
            gamma=gamma,
            pred=pred,
            pred_coef=pred_coef,
            generator=generator,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
        )
    if algo_name == "reinforce":
        return reinforce_update(
            ac=ac,
            opt=opt,
            batch=batch,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            ent_coef=ent_coef,
            epochs=epochs,
            minibatch_size=minibatch_size,
            gamma=gamma,
            pred=pred,
            pred_coef=pred_coef,
            generator=generator,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
        )
    if algo_name == "trpo":
        return trpo_update(
            ac=ac,
            opt=opt,
            batch=batch,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            ent_coef=ent_coef,
            lam=lam,
            gamma=gamma,
            pred=pred,
            pred_coef=pred_coef,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            max_kl=trpo_max_kl,
            backtrack_coef=trpo_backtrack_coef,
            backtrack_iters=trpo_backtrack_iters,
            value_epochs=trpo_value_epochs,
        )
    if algo_name == "v-trace":
        return vtrace_update(
            ac=ac,
            opt=opt,
            batch=batch,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            ent_coef=ent_coef,
            epochs=epochs,
            gamma=gamma,
            pred=pred,
            pred_coef=pred_coef,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            rho_clip=vtrace_rho_clip,
            c_clip=vtrace_c_clip,
        )
    if algo_name == "v-mpo":
        return vmpo_update(
            ac=ac,
            opt=opt,
            batch=batch,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            ent_coef=ent_coef,
            epochs=epochs,
            minibatch_size=minibatch_size,
            lam=lam,
            gamma=gamma,
            pred=pred,
            pred_coef=pred_coef,
            generator=generator,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
            topk_frac=vmpo_topk_frac,
            eta=vmpo_eta,
            kl_coef=vmpo_kl_coef,
            kl_target=vmpo_kl_target,
        )
    raise ValueError(f"Unsupported on-policy algorithm: {algo}")


class DQNReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, trace_dim: int, pin_memory: bool = False):
        self.capacity = int(capacity)
        self.pin_memory = bool(pin_memory) and torch.cuda.is_available()
        self.obs = self._alloc((self.capacity, obs_dim), dtype=torch.float32)
        self.prev_action = self._alloc(self.capacity, dtype=torch.int64)
        self.traces = self._alloc((self.capacity, trace_dim), dtype=torch.float32)
        self.actions = self._alloc(self.capacity, dtype=torch.int64)
        self.rewards = self._alloc(self.capacity, dtype=torch.float32)
        self.dones = self._alloc(self.capacity, dtype=torch.bool)
        self.next_obs = self._alloc((self.capacity, obs_dim), dtype=torch.float32)
        self.next_prev_action = self._alloc(self.capacity, dtype=torch.int64)
        self.next_traces = self._alloc((self.capacity, trace_dim), dtype=torch.float32)
        self.pos = 0
        self.size = 0

    def _alloc(self, shape, *, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.zeros(shape, dtype=dtype)
        if self.pin_memory:
            return tensor.pin_memory()
        return tensor

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        traces: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor,
        next_prev_action: torch.Tensor,
        next_traces: torch.Tensor,
    ) -> None:
        obs = obs.detach().cpu().float()
        prev_action = prev_action.detach().cpu().long()
        traces = traces.detach().cpu().float()
        actions = actions.detach().cpu().long()
        rewards = rewards.detach().cpu().float()
        dones = dones.detach().cpu().bool()
        next_obs = next_obs.detach().cpu().float()
        next_prev_action = next_prev_action.detach().cpu().long()
        next_traces = next_traces.detach().cpu().float()

        n = obs.shape[0]
        if n >= self.capacity:
            obs = obs[-self.capacity :]
            prev_action = prev_action[-self.capacity :]
            traces = traces[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            dones = dones[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            next_prev_action = next_prev_action[-self.capacity :]
            next_traces = next_traces[-self.capacity :]
            n = self.capacity

        first = min(self.capacity - self.pos, n)
        second = n - first

        sl = slice(self.pos, self.pos + first)
        self.obs[sl] = obs[:first]
        self.prev_action[sl] = prev_action[:first]
        self.traces[sl] = traces[:first]
        self.actions[sl] = actions[:first]
        self.rewards[sl] = rewards[:first]
        self.dones[sl] = dones[:first]
        self.next_obs[sl] = next_obs[:first]
        self.next_prev_action[sl] = next_prev_action[:first]
        self.next_traces[sl] = next_traces[:first]

        if second > 0:
            sl2 = slice(0, second)
            self.obs[sl2] = obs[first:]
            self.prev_action[sl2] = prev_action[first:]
            self.traces[sl2] = traces[first:]
            self.actions[sl2] = actions[first:]
            self.rewards[sl2] = rewards[first:]
            self.dones[sl2] = dones[first:]
            self.next_obs[sl2] = next_obs[first:]
            self.next_prev_action[sl2] = next_prev_action[first:]
            self.next_traces[sl2] = next_traces[first:]

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, generator: torch.Generator, device: str) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(f"Replay has only {self.size} transitions, needs {batch_size}")
        idx = torch.randint(0, self.size, (batch_size,), generator=generator)
        non_blocking = self.pin_memory and str(device).startswith("cuda")
        return {
            "obs": self.obs[idx].to(device, non_blocking=non_blocking),
            "prev_action": self.prev_action[idx].to(device, non_blocking=non_blocking),
            "traces": self.traces[idx].to(device, non_blocking=non_blocking),
            "actions": self.actions[idx].to(device, non_blocking=non_blocking),
            "rewards": self.rewards[idx].to(device, non_blocking=non_blocking),
            "dones": self.dones[idx].to(device, non_blocking=non_blocking),
            "next_obs": self.next_obs[idx].to(device, non_blocking=non_blocking),
            "next_prev_action": self.next_prev_action[idx].to(device, non_blocking=non_blocking),
            "next_traces": self.next_traces[idx].to(device, non_blocking=non_blocking),
        }


def linear_schedule(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return float(end)
    mix = min(max(step, 0), decay_steps) / decay_steps
    return float((1.0 - mix) * start + mix * end)


def hard_update_(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


@torch.no_grad()
def dqn_collect_rollout(
    envs: EnvPool,
    ac: ActorCritic,
    f_mem: FeatureEncoder,
    drift: DriftMonitor | None,
    predictor: Predictor | None,
    replay: DQNReplayBuffer,
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
    epsilon: float,
    action_generator: torch.Generator,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, dict[str, float]]:
    n_envs = envs.num_envs
    M = traces.shape[1]
    feat_dim = traces.shape[2]

    long_start = int(math.floor((1.0 - reset_long_fraction) * M))
    long_mask = torch.zeros(M, device=device, dtype=torch.bool)
    if reset_strategy == "partial":
        long_mask[long_start:] = True

    skip_drift = (reset_strategy == "none") and torch.allclose(alpha_base, alpha_max)
    alpha_const = alpha_base.expand(n_envs, -1).clamp(0.0, 1.0) if skip_drift else None

    q_taken_sum = 0.0
    q_count = 0
    for _ in range(horizon):
        obs_t = obs_to_tensor(obs, device=device, obs_normalization=obs_normalization)
        traces_flat = traces.reshape(n_envs, -1)
        q_values, value = ac(obs_t, prev_action, traces_flat)
        greedy = q_values.argmax(dim=-1)
        random_actions = torch.randint(0, q_values.shape[-1], (n_envs,), generator=action_generator, device=device)
        explore_mask = torch.rand(n_envs, generator=action_generator, device=device) < epsilon
        action = torch.where(explore_mask, random_actions, greedy)
        q_taken_sum += q_values.gather(1, action.unsqueeze(-1)).mean().item()
        q_count += 1

        next_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        done_env = terminated | truncated
        rew = torch.as_tensor(reward, device=device, dtype=torch.float32)
        done = torch.as_tensor(done_env, device=device, dtype=torch.bool)

        x_mem_t = encode_mem(f_mem, obs_t, prev_action)
        next_obs_t = obs_to_tensor(next_obs, device=device, obs_normalization=obs_normalization)
        next_prev_action = action.clone()
        if done.any():
            next_prev_action[done] = 0
        x_mem_next = encode_mem(f_mem, next_obs_t, next_prev_action)

        if skip_drift:
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
            if predictor is not None and lambda_pred > 0.0:
                x_hat = predictor(x_mem_t, action)
                pred_err = (x_mem_next - x_hat).pow(2).mean(dim=-1)
            e = delta_prov.abs() + lambda_pred * pred_err

            gate, reset_event = drift.update(e)
            reset_event = reset_event & (~done)
            alpha = alpha_base + gate.unsqueeze(-1) * (alpha_max - alpha_base)
            alpha = alpha.clamp(0.0, 1.0)

            traces_reset = maybe_reset_traces(traces, reset_event, x_mem_next, reset_strategy, long_mask)
            traces_next = trace_update(traces_reset, x_mem_next, alpha)
        if done.any():
            if drift is not None:
                drift.reset_where(done)
            traces_next[done] = 0.0
            traces_next[done] = trace_update(
                traces_next[done],
                x_mem_next[done],
                alpha_base.expand(done.sum(), -1),
            )

        replay.add_batch(
            obs=obs_t,
            prev_action=prev_action,
            traces=traces_flat,
            actions=action,
            rewards=rew,
            dones=done,
            next_obs=next_obs_t,
            next_prev_action=next_prev_action,
            next_traces=traces_next.reshape(n_envs, M * feat_dim),
        )

        obs = next_obs
        prev_action = next_prev_action
        traces = traces_next

    q_mean = q_taken_sum / max(q_count, 1)
    info = {
        "epsilon": float(epsilon),
        "q_mean": float(q_mean),
    }
    return obs, prev_action, traces, info


def dqn_update(
    ac: ActorCritic,
    target_ac: ActorCritic,
    opt: torch.optim.Optimizer,
    replay: DQNReplayBuffer,
    batch_size: int,
    gamma: float,
    double_dqn: bool,
    generator: torch.Generator,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_scaler: torch.amp.GradScaler | None,
    max_grad_norm: float,
) -> dict[str, float] | None:
    if len(replay) < batch_size:
        return None
    batch = replay.sample(batch_size=batch_size, generator=generator, device=device)

    with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
        q_values, _ = ac(batch["obs"], batch["prev_action"], batch["traces"])
        q_selected = q_values.gather(1, batch["actions"].unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            if double_dqn:
                q_online_next, _ = ac(batch["next_obs"], batch["next_prev_action"], batch["next_traces"])
                next_actions = q_online_next.argmax(dim=-1)
                q_target_next, _ = target_ac(batch["next_obs"], batch["next_prev_action"], batch["next_traces"])
                q_next = q_target_next.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            else:
                q_target_next, _ = target_ac(batch["next_obs"], batch["next_prev_action"], batch["next_traces"])
                q_next = q_target_next.max(dim=-1).values
            q_boot = batch["rewards"] + gamma * (~batch["dones"]).float() * q_next
        q_loss = F.smooth_l1_loss(q_selected, q_boot)

    _step_optimizer(loss=q_loss, opt=opt, model=ac, grad_scaler=grad_scaler, max_grad_norm=max_grad_norm)
    with torch.no_grad():
        td_abs = (q_selected - q_boot).abs().mean().item()
    return {
        "q_loss": q_loss.detach().item(),
        "q_mean": q_selected.detach().mean().item(),
        "target_q_mean": q_boot.detach().mean().item(),
        "td_abs": td_abs,
    }
