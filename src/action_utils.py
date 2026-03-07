from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical, Independent, Normal


_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0
_SQUASH_EPS = 1e-6


def init_prev_action(*, num_envs: int, action_mode: str, act_dim: int, device: str) -> torch.Tensor:
    if str(action_mode) == "continuous":
        return torch.zeros((num_envs, act_dim), device=device, dtype=torch.float32)
    return torch.zeros(num_envs, device=device, dtype=torch.int64)


def _parse_continuous_policy_params(policy_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if policy_out.shape[-1] % 2 != 0:
        raise ValueError(f"Continuous policy output must have even last dimension, got {tuple(policy_out.shape)}")
    mean, log_std = torch.chunk(policy_out.float(), 2, dim=-1)
    log_std = torch.clamp(log_std, min=_LOG_STD_MIN, max=_LOG_STD_MAX)
    std = log_std.exp()
    return mean, std


def _continuous_action_bounds(
    *,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if action_low is None or action_high is None:
        low = torch.full((1, action_dim), -1.0, device=device, dtype=dtype)
        high = torch.full((1, action_dim), 1.0, device=device, dtype=dtype)
    else:
        low = torch.as_tensor(action_low, device=device, dtype=dtype).reshape(1, -1)
        high = torch.as_tensor(action_high, device=device, dtype=dtype).reshape(1, -1)
    if low.shape[-1] != action_dim or high.shape[-1] != action_dim:
        raise ValueError(
            f"Action bounds shape mismatch: got low={tuple(low.shape)}, high={tuple(high.shape)}, "
            f"expected last dim {action_dim}."
        )
    if torch.any(high <= low):
        raise ValueError("Continuous action bounds must satisfy high > low elementwise.")
    scale = 0.5 * (high - low)
    bias = 0.5 * (high + low)
    return low, high, scale, bias


def _atanh_stable(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _squash_from_latent(
    latent_action: torch.Tensor,
    *,
    scale: torch.Tensor,
    bias: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    z = torch.tanh(latent_action)
    action = bias + (scale * z)
    return torch.maximum(torch.minimum(action, high), low)


def _squashed_log_prob(
    *,
    base_dist: Independent,
    actions: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    z = (actions - bias) / scale
    z = torch.clamp(z, min=-1.0 + _SQUASH_EPS, max=1.0 - _SQUASH_EPS)
    latent = _atanh_stable(z)
    base_logp = base_dist.log_prob(latent)
    log_det = torch.log(torch.clamp(scale, min=_SQUASH_EPS)) + torch.log(
        torch.clamp(1.0 - z.pow(2), min=_SQUASH_EPS)
    )
    return base_logp - log_det.sum(dim=-1)


def policy_dist(policy_out: torch.Tensor, *, action_mode: str):
    mode = str(action_mode)
    if mode == "discrete":
        return Categorical(logits=policy_out)
    if mode == "continuous":
        mean, std = _parse_continuous_policy_params(policy_out)
        return Independent(Normal(mean, std), 1)
    raise ValueError(f"Unsupported action_mode: {action_mode}")


def deterministic_action(
    policy_out: torch.Tensor,
    *,
    action_mode: str,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> torch.Tensor:
    mode = str(action_mode)
    if mode == "discrete":
        return policy_out.argmax(dim=-1)
    if mode == "continuous":
        mean, _std = _parse_continuous_policy_params(policy_out)
        low, high, scale, bias = _continuous_action_bounds(
            action_dim=int(mean.shape[-1]),
            device=mean.device,
            dtype=mean.dtype,
            action_low=action_low,
            action_high=action_high,
        )
        return _squash_from_latent(mean, scale=scale, bias=bias, low=low, high=high)
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
        mean, std = _parse_continuous_policy_params(policy_out)
        base_dist = Independent(Normal(mean, std), 1)
        _low, _high, scale, bias = _continuous_action_bounds(
            action_dim=int(mean.shape[-1]),
            device=mean.device,
            dtype=mean.dtype,
            action_low=action_low,
            action_high=action_high,
        )
        logp = _squashed_log_prob(
            base_dist=base_dist,
            actions=actions.float(),
            scale=scale,
            bias=bias,
        )
        # Exact transformed entropy is intractable in closed-form; use base Gaussian entropy proxy.
        entropy = base_dist.entropy().mean()
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
        entropy = dist.entropy().mean()
        max_action_stat = policy_out.float().softmax(dim=-1).max(dim=-1).values.mean()
    else:
        mean, std = _parse_continuous_policy_params(policy_out)
        base_dist = Independent(Normal(mean, std), 1)
        low, high, scale, bias = _continuous_action_bounds(
            action_dim=int(mean.shape[-1]),
            device=mean.device,
            dtype=mean.dtype,
            action_low=action_low,
            action_high=action_high,
        )
        if deterministic:
            actions = _squash_from_latent(mean, scale=scale, bias=bias, low=low, high=high)
        else:
            latent_action = base_dist.sample()
            actions = _squash_from_latent(latent_action, scale=scale, bias=bias, low=low, high=high)
        logp = _squashed_log_prob(
            base_dist=base_dist,
            actions=actions,
            scale=scale,
            bias=bias,
        )
        entropy = base_dist.entropy().mean()
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
    arr = out.detach().cpu().numpy().astype(np.float32, copy=False)
    if len(action_shape) > 0:
        arr = arr.reshape((arr.shape[0], *action_shape))
    return arr
