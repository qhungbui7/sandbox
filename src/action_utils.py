from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal


_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0
_BOUND_EPS = 1e-6
_BOUNDS_ATOL = 1e-6


def init_prev_action(*, num_envs: int, action_mode: str, act_dim: int, device: str) -> torch.Tensor:
    if str(action_mode) == "continuous":
        return torch.zeros((num_envs, act_dim), device=device, dtype=torch.float32)
    return torch.zeros(num_envs, device=device, dtype=torch.int64)


def policy_dist(policy_out: torch.Tensor, *, action_mode: str):
    mode = str(action_mode)
    if mode == "discrete":
        return Categorical(logits=policy_out)
    if mode == "continuous":
        if policy_out.shape[-1] % 2 != 0:
            raise ValueError(f"Continuous policy output must have even last dimension, got {tuple(policy_out.shape)}")
        mean, log_std = torch.chunk(policy_out, 2, dim=-1)
        log_std = torch.clamp(log_std, min=_LOG_STD_MIN, max=_LOG_STD_MAX)
        std = log_std.exp()
        return Independent(Normal(mean, std), 1)
    raise ValueError(f"Unsupported action_mode: {action_mode}")


def _continuous_params(policy_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if policy_out.shape[-1] % 2 != 0:
        raise ValueError(f"Continuous policy output must have even last dimension, got {tuple(policy_out.shape)}")
    mean, log_std = torch.chunk(policy_out, 2, dim=-1)
    log_std = torch.clamp(log_std, min=_LOG_STD_MIN, max=_LOG_STD_MAX)
    std = log_std.exp()
    return mean, log_std, std


def _resolve_continuous_bounds(
    *,
    mean: torch.Tensor,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    act_dim = int(mean.shape[-1])
    if action_low is None or action_high is None:
        low_t = torch.full((act_dim,), -1.0, device=mean.device, dtype=mean.dtype)
        high_t = torch.full((act_dim,), 1.0, device=mean.device, dtype=mean.dtype)
        return low_t, high_t

    low_t = torch.as_tensor(action_low, device=mean.device, dtype=mean.dtype).reshape(-1)
    high_t = torch.as_tensor(action_high, device=mean.device, dtype=mean.dtype).reshape(-1)
    if int(low_t.numel()) != act_dim or int(high_t.numel()) != act_dim:
        raise ValueError(
            f"Continuous action bounds must match act_dim={act_dim}, got low={tuple(low_t.shape)} high={tuple(high_t.shape)}"
        )
    if not torch.isfinite(low_t).all() or not torch.isfinite(high_t).all():
        raise ValueError("Continuous action bounds must be finite.")
    if not torch.all(high_t > low_t):
        raise ValueError("Continuous action bounds must satisfy high > low elementwise.")
    return low_t, high_t


def _tanh_mask(low_t: torch.Tensor, high_t: torch.Tensor) -> torch.Tensor:
    return torch.isclose(low_t, -torch.ones_like(low_t), atol=_BOUNDS_ATOL, rtol=0.0) & torch.isclose(
        high_t, torch.ones_like(high_t), atol=_BOUNDS_ATOL, rtol=0.0
    )


def _continuous_action_from_latent(
    latent: torch.Tensor,
    *,
    low_t: torch.Tensor,
    high_t: torch.Tensor,
    tanh_mask: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(latent)
    if bool(tanh_mask.any().item()):
        out[..., tanh_mask] = torch.tanh(latent[..., tanh_mask])
    other_mask = ~tanh_mask
    if bool(other_mask.any().item()):
        scale = (high_t[other_mask] - low_t[other_mask]).to(dtype=latent.dtype)
        out[..., other_mask] = low_t[other_mask] + scale * torch.sigmoid(latent[..., other_mask])
    return out


def _continuous_log_abs_det_from_latent(
    latent: torch.Tensor,
    *,
    low_t: torch.Tensor,
    high_t: torch.Tensor,
    tanh_mask: torch.Tensor,
) -> torch.Tensor:
    log_abs_det = torch.zeros(latent.shape[:-1], device=latent.device, dtype=latent.dtype)
    if bool(tanh_mask.any().item()):
        z_tanh = latent[..., tanh_mask]
        # log|d(tanh(z))/dz|, stable for large |z|
        log_abs_det = log_abs_det + (2.0 * (math.log(2.0) - z_tanh - F.softplus(-2.0 * z_tanh))).sum(dim=-1)
    other_mask = ~tanh_mask
    if bool(other_mask.any().item()):
        z_other = latent[..., other_mask]
        scale = (high_t[other_mask] - low_t[other_mask]).to(dtype=latent.dtype).clamp_min(_BOUND_EPS)
        log_abs_det = log_abs_det + (
            torch.log(scale) + F.logsigmoid(z_other) + F.logsigmoid(-z_other)
        ).sum(dim=-1)
    return log_abs_det


def _continuous_log_prob_from_actions(
    actions: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    low_t: torch.Tensor,
    high_t: torch.Tensor,
    tanh_mask: torch.Tensor,
) -> torch.Tensor:
    actions_f = actions.float()
    if actions_f.ndim == mean.ndim - 1:
        actions_f = actions_f.unsqueeze(-1)
    if actions_f.shape[-1] != mean.shape[-1]:
        raise ValueError(
            f"Continuous action shape mismatch: expected last dim {mean.shape[-1]}, got {actions_f.shape[-1]}"
        )
    latent = torch.empty_like(actions_f)
    if bool(tanh_mask.any().item()):
        a_tanh = actions_f[..., tanh_mask].clamp(-1.0 + _BOUND_EPS, 1.0 - _BOUND_EPS)
        latent[..., tanh_mask] = torch.atanh(a_tanh)
    other_mask = ~tanh_mask
    if bool(other_mask.any().item()):
        scale = (high_t[other_mask] - low_t[other_mask]).to(dtype=actions_f.dtype).clamp_min(_BOUND_EPS)
        u = ((actions_f[..., other_mask] - low_t[other_mask]) / scale).clamp(_BOUND_EPS, 1.0 - _BOUND_EPS)
        latent[..., other_mask] = torch.logit(u)

    base = Independent(Normal(mean, std), 1)
    log_abs_det = _continuous_log_abs_det_from_latent(
        latent,
        low_t=low_t,
        high_t=high_t,
        tanh_mask=tanh_mask,
    )
    return base.log_prob(latent) - log_abs_det


def _continuous_entropy_estimate(
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    low_t: torch.Tensor,
    high_t: torch.Tensor,
    tanh_mask: torch.Tensor,
) -> torch.Tensor:
    latent = mean + std * torch.randn_like(mean)
    base = Independent(Normal(mean, std), 1)
    log_abs_det = _continuous_log_abs_det_from_latent(
        latent,
        low_t=low_t,
        high_t=high_t,
        tanh_mask=tanh_mask,
    )
    logp = base.log_prob(latent) - log_abs_det
    return (-logp).mean()


def deterministic_action(policy_out: torch.Tensor, *, action_mode: str) -> torch.Tensor:
    mode = str(action_mode)
    if mode == "discrete":
        return policy_out.argmax(dim=-1)
    if mode == "continuous":
        mean, _log_std = torch.chunk(policy_out, 2, dim=-1)
        return mean
    raise ValueError(f"Unsupported action_mode: {action_mode}")


def evaluate_policy_actions(
    *,
    policy_out: torch.Tensor,
    actions: torch.Tensor,
    action_mode: str,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if str(action_mode) == "discrete":
        dist = policy_dist(policy_out, action_mode=action_mode)
        logp = dist.log_prob(actions.long())
        max_action_stat = policy_out.float().softmax(dim=-1).max(dim=-1).values.mean()
        entropy = dist.entropy().mean()
    else:
        mean, _log_std, std = _continuous_params(policy_out)
        low_t, high_t = _resolve_continuous_bounds(mean=mean, action_low=action_low, action_high=action_high)
        tanh_mask = _tanh_mask(low_t, high_t)
        logp = _continuous_log_prob_from_actions(
            actions=actions,
            mean=mean,
            std=std,
            low_t=low_t,
            high_t=high_t,
            tanh_mask=tanh_mask,
        )
        entropy = _continuous_entropy_estimate(
            mean=mean,
            std=std,
            low_t=low_t,
            high_t=high_t,
            tanh_mask=tanh_mask,
        )
        max_action_stat = torch.full((), float("nan"), device=policy_out.device)
    return logp, entropy, max_action_stat


def sample_policy_actions(
    *,
    policy_out: torch.Tensor,
    action_mode: str,
    deterministic: bool,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if str(action_mode) == "discrete":
        dist = policy_dist(policy_out, action_mode=action_mode)
        if deterministic:
            actions = deterministic_action(policy_out, action_mode=action_mode)
        else:
            actions = dist.sample()
        logp = dist.log_prob(actions.long())
        max_action_stat = policy_out.float().softmax(dim=-1).max(dim=-1).values.mean()
        entropy = dist.entropy().mean()
    else:
        mean, _log_std, std = _continuous_params(policy_out)
        low_t, high_t = _resolve_continuous_bounds(mean=mean, action_low=action_low, action_high=action_high)
        tanh_mask = _tanh_mask(low_t, high_t)
        if deterministic:
            latent = mean
        else:
            latent = mean + std * torch.randn_like(mean)
        actions = _continuous_action_from_latent(
            latent,
            low_t=low_t,
            high_t=high_t,
            tanh_mask=tanh_mask,
        )
        base = Independent(Normal(mean, std), 1)
        log_abs_det = _continuous_log_abs_det_from_latent(
            latent,
            low_t=low_t,
            high_t=high_t,
            tanh_mask=tanh_mask,
        )
        logp = base.log_prob(latent) - log_abs_det
        entropy = _continuous_entropy_estimate(
            mean=mean,
            std=std,
            low_t=low_t,
            high_t=high_t,
            tanh_mask=tanh_mask,
        )
        max_action_stat = torch.full((), float("nan"), device=policy_out.device)
    return actions, logp, entropy, max_action_stat


def actions_to_env_numpy(
    *,
    actions: torch.Tensor,
    action_mode: str,
    action_shape: tuple[int, ...],
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
) -> np.ndarray:
    mode = str(action_mode)
    if mode == "discrete":
        return actions.detach().cpu().numpy()
    if mode != "continuous":
        raise ValueError(f"Unsupported action_mode: {action_mode}")

    out = actions.float()
    if action_low is not None and action_high is not None:
        low_t = torch.as_tensor(action_low, device=out.device, dtype=out.dtype).reshape(1, -1)
        high_t = torch.as_tensor(action_high, device=out.device, dtype=out.dtype).reshape(1, -1)
        out = torch.maximum(torch.minimum(out, high_t), low_t)
    arr = out.detach().cpu().numpy().astype(np.float32, copy=False)
    if len(action_shape) > 0:
        arr = arr.reshape((arr.shape[0], *action_shape))
    return arr
