import torch
import torch.nn as nn

from src.action_utils import evaluate_policy_actions
from src.models import ActorCritic, Predictor, RecurrentActorCritic
from src.utils import autocast_context


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


def _step_optimizer_recurrent(
    *,
    loss: torch.Tensor,
    opt: torch.optim.Optimizer,
    model: nn.Module,
    grad_scaler: torch.amp.GradScaler | None,
    max_grad_norm: float,
    grad_modules: dict[str, nn.Module] | None,
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
) -> dict[str, float]:
    action_mode = str(getattr(getattr(ac, "f_pol", None), "action_type", "discrete"))
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

    obs_f = obs.reshape((b, *obs.shape[2:]))
    if prev_a.ndim == 2:
        prev_a_f = prev_a.reshape(b)
    else:
        prev_a_f = prev_a.reshape(b, -1)
    traces_f = traces.reshape(b, -1)
    if actions.ndim == 2:
        actions_f = actions.reshape(b)
    else:
        actions_f = actions.reshape(b, -1)
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
    stop_due_to_kl = False
    kl_early_stop = target_kl > 0.0
    for _ in range(epochs):
        perm = idx[torch.randperm(b, generator=generator, device=idx.device)]
        for start in range(0, b, minibatch_size):
            mb = perm[start : start + minibatch_size]

            with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
                policy_out, values = ac(obs_f[mb], prev_a_f[mb], traces_f[mb])
                logp, entropy, _max_action_prob = evaluate_policy_actions(
                    policy_out=policy_out,
                    actions=actions_f[mb],
                    action_mode=action_mode,
                )

                log_ratio = logp - logp_old_f[mb]
                ratio = log_ratio.exp()
                pg1 = ratio * adv_f[mb]
                pg2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv_f[mb]
                policy_loss = -torch.min(pg1, pg2).mean()
                if vf_clip:
                    value_delta = values - values_old_f[mb]
                    values_clipped = values_old_f[mb] + torch.clamp(value_delta, -clip_coef, clip_coef)
                    value_err = (returns_f[mb] - values).pow(2)
                    value_err_clipped = (returns_f[mb] - values_clipped).pow(2)
                    value_loss = 0.5 * torch.maximum(value_err, value_err_clipped).mean()
                else:
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
                nn.utils.clip_grad_norm_(ac.parameters(), max_norm=max_grad_norm)
                grad_scaler.step(opt)
                grad_scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_norm=max_grad_norm)
                opt.step()

            with torch.no_grad():
                approx_kl = (logp_old_f[mb] - logp).mean()
                kl_for_stop = (ratio - 1.0 - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

            stats["policy_loss"] += policy_loss.detach().item()
            stats["value_loss"] += value_loss.detach().item()
            stats["entropy"] += entropy.detach().item()
            stats["pred_loss"] += pred_loss.detach().item()
            stats["approx_kl"] += approx_kl.detach().item()
            stats["clipfrac"] += clipfrac.detach().item()
            stats["count"] += 1.0

            if kl_early_stop and float(kl_for_stop.detach().item()) > target_kl:
                stop_due_to_kl = True
                break
        if stop_due_to_kl:
            break

    count = max(stats["count"], 1.0)
    for k in list(stats.keys()):
        if k != "count":
            stats[k] /= count
    stats.pop("count", None)
    return stats


def ppo_update_recurrent(
    ac: RecurrentActorCritic,
    opt: torch.optim.Optimizer,
    batch: dict,
    clip_coef: float,
    vf_clip: bool,
    target_kl: float,
    vf_coef: float,
    max_grad_norm: float,
    ent_coef: float,
    epochs: int,
    lam: float,
    gamma: float,
    generator: torch.Generator,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_scaler: torch.amp.GradScaler | None,
    debug_cfg: dict | None = None,
) -> dict[str, float]:
    action_mode = str(getattr(getattr(ac, "f_pol", None), "action_type", "discrete"))
    obs = batch["obs"]
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
    sum_policy = torch.zeros((), device=obs.device)
    sum_value = torch.zeros((), device=obs.device)
    sum_entropy = torch.zeros((), device=obs.device)
    sum_approx_kl = torch.zeros((), device=obs.device)
    sum_clipfrac = torch.zeros((), device=obs.device)
    mb_count = 0

    debug_enabled = debug_cfg is not None
    if debug_enabled:
        action_bins = int(debug_cfg["action_bins"]) if action_mode == "discrete" else 0
        ratio_cap = int(debug_cfg["ratio_sample_size"])
        delta_pairs = int(debug_cfg["frame_delta_pairs"])
        seed = int(debug_cfg["seed"])
        update_idx = int(debug_cfg["update_idx"])
        debug_gen = torch.Generator(device=obs.device)
        debug_gen.manual_seed(seed * 1_000_003 + update_idx)

        obs_f = obs.reshape((T * N, *obs.shape[2:]))
        obs_min = obs_f.amin()
        obs_max = obs_f.amax()
        obs_mean = obs_f.mean()
        obs_std = obs_f.std(unbiased=False)
        obs_zero_frac = (obs_f == 0.0).float().mean()
        obs_max_frac = (obs_f == obs_max).float().mean()

        obs_seq = obs
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

        values_old_f = values_old.reshape(-1)
        returns_f = returns.reshape(-1)
        td_error = returns_f - values_old_f
        var_returns = returns_f.var(unbiased=False)
        explained_var = 1.0 - (td_error.var(unbiased=False) / (var_returns + 1e-8))

        if action_mode == "discrete":
            action_hist = torch.bincount(actions.reshape(-1).long(), minlength=action_bins).float()[:action_bins]
        else:
            action_hist = None
        ratio_reservoir = torch.zeros((ratio_cap,), device=obs.device)
        ratio_filled = 0

        sum_total = torch.zeros((), device=obs.device)
        max_approx_kl = torch.full((), float("-inf"), device=obs.device)
        sum_max_action_prob = torch.zeros((), device=obs.device)
        sum_grad_global = torch.zeros((), device=obs.device)
        max_grad_global = torch.zeros((), device=obs.device)
        sum_grad_encoder = torch.zeros((), device=obs.device)
        sum_grad_core = torch.zeros((), device=obs.device)
        sum_grad_pi = torch.zeros((), device=obs.device)
        sum_grad_v = torch.zeros((), device=obs.device)
        pi_module = ac.pi if hasattr(ac, "pi") else ac.pi_mean
        grad_modules = {
            "encoder": ac.f_pol.obs_encoder,
            "core": ac.lstm,
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
        perm_env = env_idx[torch.randperm(N, generator=generator, device=env_idx.device)]
        for env_id in perm_env:
            h = h0[:, env_id : env_id + 1].detach()
            c = c0[:, env_id : env_id + 1].detach()

            traj_logp = []
            traj_values = []
            traj_entropy = []
            traj_max_action_prob = []

            for t in range(T):
                with autocast_context(device=device, enabled=use_amp, dtype=amp_dtype):
                    policy_out, value, (h, c) = ac(obs[t, env_id : env_id + 1], prev_a[t, env_id : env_id + 1], (h, c))
                    action_t = actions[t, env_id : env_id + 1]
                    logp_t, entropy_t, max_prob_t = evaluate_policy_actions(
                        policy_out=policy_out,
                        actions=action_t,
                        action_mode=action_mode,
                    )
                    lp = logp_t.squeeze(-1)
                    traj_logp.append(lp)
                    traj_values.append(value)
                    traj_entropy.append(entropy_t)
                    if debug_enabled:
                        traj_max_action_prob.append(max_prob_t)
                done_t = dones[t, env_id].to(dtype=h.dtype)
                h = h * (1.0 - done_t)
                c = c * (1.0 - done_t)

            values_all = torch.cat(traj_values, dim=0).squeeze(-1)
            logp_all = torch.stack(traj_logp, dim=0)
            entropy = torch.stack(traj_entropy, dim=0).mean()

            log_ratio = logp_all - logp_old[:, env_id]
            ratio = log_ratio.exp()
            pg1 = ratio * adv[:, env_id]
            pg2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv[:, env_id]
            policy_loss = -torch.min(pg1, pg2).mean()
            if vf_clip:
                value_delta = values_all - values_old[:, env_id]
                values_clipped = values_old[:, env_id] + torch.clamp(value_delta, -clip_coef, clip_coef)
                value_err = (returns[:, env_id] - values_all).pow(2)
                value_err_clipped = (returns[:, env_id] - values_clipped).pow(2)
                value_loss = 0.5 * torch.maximum(value_err, value_err_clipped).mean()
            else:
                value_loss = 0.5 * (returns[:, env_id] - values_all).pow(2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            global_grad_norm, module_grad_norms = _step_optimizer_recurrent(
                loss=loss,
                opt=opt,
                model=ac,
                grad_scaler=grad_scaler,
                max_grad_norm=max_grad_norm,
                grad_modules=grad_modules,
            )

            with torch.no_grad():
                approx_kl = (logp_old[:, env_id] - logp_all).mean()
                kl_for_stop = (ratio - 1.0 - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
                sum_policy = sum_policy + policy_loss.detach().float()
                sum_value = sum_value + value_loss.detach().float()
                sum_entropy = sum_entropy + entropy.detach().float()
                sum_approx_kl = sum_approx_kl + approx_kl.detach().float()
                sum_clipfrac = sum_clipfrac + clipfrac.detach().float()
                mb_count += 1

                if debug_enabled:
                    sum_total = sum_total + loss.detach().float()
                    max_approx_kl = torch.maximum(max_approx_kl, approx_kl.detach().float())
                    sum_max_action_prob = sum_max_action_prob + torch.stack(traj_max_action_prob, dim=0).mean()
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
        "approx_kl": float((sum_approx_kl * inv_count).item()),
        "clipfrac": float((sum_clipfrac * inv_count).item()),
    }

    if debug_enabled:
        if ratio_filled > 0:
            ratio_sample = ratio_reservoir[:ratio_filled].detach().float().cpu()
            ratio_q = torch.quantile(ratio_sample, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float32))
            ratio_p01 = float(ratio_q[0].item())
            ratio_p50 = float(ratio_q[1].item())
            ratio_p99 = float(ratio_q[2].item())
        else:
            ratio_p01 = float("nan")
            ratio_p50 = float("nan")
            ratio_p99 = float("nan")
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
                "debug/ppo/pred_loss_mean": 0.0,
                "debug/ppo/total_loss_mean": float((sum_total * inv_count).item()),
                "debug/ppo/approx_kl_mean": float((sum_approx_kl * inv_count).item()),
                "debug/ppo/approx_kl_max": float(max_approx_kl.item()),
                "debug/ppo/clipfrac_mean": float((sum_clipfrac * inv_count).item()),
                "debug/ppo/ratio_p01": ratio_p01,
                "debug/ppo/ratio_p50": ratio_p50,
                "debug/ppo/ratio_p99": ratio_p99,
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
