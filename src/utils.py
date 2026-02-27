import contextlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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


def obs_to_tensor(obs, *, device: str, obs_normalization: str = "auto") -> torch.Tensor:
    mode = str(obs_normalization).strip().lower()
    if mode not in {"auto", "none", "uint8"}:
        raise ValueError(f"Unsupported obs normalization mode: {obs_normalization}")
    source_dtype = getattr(obs, "dtype", None)
    out = torch.as_tensor(obs, device=device, dtype=torch.float32)
    if mode == "none":
        return out
    if mode == "uint8":
        return out / 255.0
    if _is_uint8_dtype(source_dtype):
        return out / 255.0
    return out
