"""Randomness helpers for VAMP experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_random_seed(seed: int, enable_cudnn_determinism: bool = True) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs."""

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # type: ignore
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - device specific
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if enable_cudnn_determinism and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]

