from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical, Independent, Normal


_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = policy_dist(policy_out, action_mode=action_mode)
    if str(action_mode) == "discrete":
        logp = dist.log_prob(actions.long())
        max_action_stat = policy_out.float().softmax(dim=-1).max(dim=-1).values.mean()
    else:
        logp = dist.log_prob(actions.float())
        max_action_stat = torch.full((), float("nan"), device=policy_out.device)
    entropy = dist.entropy().mean()
    return logp, entropy, max_action_stat


def sample_policy_actions(
    *,
    policy_out: torch.Tensor,
    action_mode: str,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = policy_dist(policy_out, action_mode=action_mode)
    if deterministic:
        actions = deterministic_action(policy_out, action_mode=action_mode)
    else:
        actions = dist.sample()
    if str(action_mode) == "discrete":
        logp = dist.log_prob(actions.long())
        max_action_stat = policy_out.float().softmax(dim=-1).max(dim=-1).values.mean()
    else:
        logp = dist.log_prob(actions.float())
        max_action_stat = torch.full((), float("nan"), device=policy_out.device)
    entropy = dist.entropy().mean()
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
