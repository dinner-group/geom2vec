"""Tensor conversion helpers for VAMP modules."""

from __future__ import annotations

from typing import Generator, Iterable

import numpy as np


def map_data_to_tensor(data: Iterable, device=None, dtype=np.float32):
    """Yield tensors from iterable trajectory data without moving to device by default."""

    import torch  # imported lazily to avoid hard dependency during package import

    with torch.no_grad():
        if not isinstance(data, (list, tuple)):
            data = [data]
        for item in data:
            if isinstance(item, torch.Tensor):
                tensor = item
            else:
                tensor = torch.from_numpy(np.asarray(item, dtype=dtype).copy())
            if device is not None:
                tensor = tensor.to(device=device)
            yield tensor

