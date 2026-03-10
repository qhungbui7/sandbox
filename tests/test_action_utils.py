import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.action_utils import evaluate_policy_actions, sample_policy_actions  # noqa: E402


def test_continuous_deterministic_mixed_bounds_mapping():
    policy_out = torch.zeros((2, 6), dtype=torch.float32)
    policy_out[:, 0] = 0.0
    policy_out[:, 1] = 0.0
    policy_out[:, 2] = 0.0

    action_low = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    action_high = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

    actions, logp, entropy, _ = sample_policy_actions(
        policy_out=policy_out,
        action_mode="continuous",
        deterministic=True,
        action_low=action_low,
        action_high=action_high,
    )
    assert actions.shape == (2, 3)
    expected = torch.tensor([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]], dtype=torch.float32)
    assert torch.allclose(actions, expected, atol=1e-6)
    assert torch.isfinite(logp).all()
    assert torch.isfinite(entropy)


def test_continuous_logprob_finite_near_action_bounds():
    policy_out = torch.zeros((3, 6), dtype=torch.float32)
    action_low = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    action_high = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    actions = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.25, 0.4, 0.9],
        ],
        dtype=torch.float32,
    )

    logp, entropy, _ = evaluate_policy_actions(
        policy_out=policy_out,
        actions=actions,
        action_mode="continuous",
        action_low=action_low,
        action_high=action_high,
    )
    assert torch.isfinite(logp).all()
    assert torch.isfinite(entropy)
