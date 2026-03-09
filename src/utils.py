import contextlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_env_file(path: str | Path | None = None) -> None:
    """Load key=value pairs from a .env file into os.environ."""
    env_path = Path(path) if path is not None else Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and ((value[0] == value[-1]) and value[0] in {'"', "'"}):
            value = value[1:-1]
        os.environ[key] = value


def resolve_device(device_str: str, cuda_id: int | None) -> str:
    """Normalize device selection and allow explicit CUDA index override."""
    if cuda_id is not None:
        return f"cuda:{cuda_id}"
    return device_str


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Avoid touching CUDA here: calling into torch.cuda can emit warnings or
    # initialize CUDA even for CPU-only runs/tests. CUDA seeding is handled by
    # the main entrypoint after device selection.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def one_hot(x: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(x.long(), n).float()


def autocast_context(device: str, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return contextlib.nullcontext()
    if torch.device(device).type != "cuda":
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def _is_uint8_dtype(dtype: object) -> bool:
    if dtype is None:
        return False
    if isinstance(dtype, torch.dtype):
        return dtype == torch.uint8
    try:
        return np.dtype(dtype) == np.uint8
    except TypeError:
        return False


def _imagenet_stats(num_channels: int, *, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if num_channels < 1:
        raise ValueError(f"Expected channel dimension >= 1, got {num_channels}")
    base_mean = like.new_tensor(IMAGENET_MEAN)
    base_std = like.new_tensor(IMAGENET_STD)
    if num_channels == 1:
        return base_mean.mean().reshape(1), base_std.mean().reshape(1)
    repeats = (num_channels + 2) // 3
    return base_mean.repeat(repeats)[:num_channels], base_std.repeat(repeats)[:num_channels]


def _is_channel_last_image_shape(shape: tuple[int, ...]) -> bool:
    if len(shape) == 3:
        h, w, c = shape
        return h > 0 and w > 0 and c > 0
    if len(shape) == 4:
        _b, h, w, c = shape
        return h > 0 and w > 0 and c > 0
    return False


def _to_channel_first_if_image(obs: torch.Tensor) -> torch.Tensor:
    if not _is_channel_last_image_shape(tuple(int(x) for x in obs.shape)):
        return obs
    if obs.ndim == 3:
        return obs.permute(2, 0, 1).contiguous()
    if obs.ndim == 4:
        return obs.permute(0, 3, 1, 2).contiguous()
    return obs


def obs_to_tensor(obs, *, device: str, obs_normalization: str = "auto") -> torch.Tensor:
    mode = str(obs_normalization).strip().lower()
    if mode not in {"auto", "none", "uint8", "imagenet"}:
        raise ValueError(f"Unsupported obs normalization mode: {obs_normalization}")
    source_dtype = getattr(obs, "dtype", None)
    out = torch.as_tensor(obs, dtype=torch.float32)
    if mode == "none":
        pass
    elif mode == "uint8":
        out = out / 255.0
    elif mode == "imagenet":
        if out.ndim < 3:
            raise ValueError(
                "`obs_normalization=imagenet` expects image observations with a channel-last axis "
                "(e.g., [B,H,W,C] or [H,W,C])."
            )
        if _is_uint8_dtype(source_dtype):
            out = out / 255.0
        channels = int(out.shape[-1])
        mean, std = _imagenet_stats(channels, like=out)
        shape = [1] * out.ndim
        shape[-1] = channels
        out = (out - mean.view(*shape)) / std.view(*shape)
    elif _is_uint8_dtype(source_dtype):
        out = out / 255.0

    out = _to_channel_first_if_image(out)

    target_device = torch.device(device)
    if out.device == target_device:
        return out
    if target_device.type == "cuda" and out.device.type == "cpu":
        if not out.is_pinned():
            out = out.pin_memory()
        return out.to(target_device, non_blocking=True)
    return out.to(target_device)
