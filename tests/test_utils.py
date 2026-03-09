import numpy as np
import torch

from src.utils import obs_to_tensor


def test_obs_to_tensor_imagenet_rgb_uint8():
    obs = np.asarray([[[[0, 127, 255]]]], dtype=np.uint8)
    out = obs_to_tensor(obs, device="cpu", obs_normalization="imagenet")

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    expected = (torch.tensor([0.0, 127.0 / 255.0, 1.0], dtype=torch.float32) - mean) / std

    assert out.shape == (1, 3, 1, 1)
    assert torch.allclose(out[0, :, 0, 0], expected, atol=1e-6)


def test_obs_to_tensor_imagenet_grayscale_uses_single_channel_stats():
    obs = np.full((1, 2, 2, 1), 255, dtype=np.uint8)
    out = obs_to_tensor(obs, device="cpu", obs_normalization="imagenet")
    mean = (0.485 + 0.456 + 0.406) / 3.0
    std = (0.229 + 0.224 + 0.225) / 3.0
    expected = (1.0 - mean) / std

    assert out.shape == (1, 1, 2, 2)
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-6)


def test_obs_to_tensor_imagenet_requires_image_shape():
    obs = np.zeros((2, 4), dtype=np.uint8)
    try:
        _ = obs_to_tensor(obs, device="cpu", obs_normalization="imagenet")
        raise AssertionError("Expected ValueError for non-image observation shape.")
    except ValueError as exc:
        assert "expects image observations" in str(exc)


def test_obs_to_tensor_auto_converts_image_batch_to_channel_first():
    obs = np.zeros((2, 4, 5, 3), dtype=np.uint8)
    out = obs_to_tensor(obs, device="cpu", obs_normalization="auto")
    assert out.shape == (2, 3, 4, 5)


def test_obs_to_tensor_auto_keeps_vector_batch_shape():
    obs = np.zeros((2, 8), dtype=np.float32)
    out = obs_to_tensor(obs, device="cpu", obs_normalization="auto")
    assert out.shape == (2, 8)
