import os
import random
from pathlib import Path
import contextlib
import numpy as np
import torch


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_hot(x: torch.Tensor, n: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(x.long(), n).float()


def autocast_context(device: str, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return contextlib.nullcontext()
    if torch.device(device).type != "cuda":
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)
