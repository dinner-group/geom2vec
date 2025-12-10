"""Statistical utility functions for VAMP."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def empirical_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return the absolute empirical correlation between two vectors."""

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    numerator = np.mean(x_centered * y_centered)
    denominator = np.sqrt(np.mean(x_centered * x_centered)) * np.sqrt(np.mean(y_centered * y_centered))
    return float(abs(numerator / denominator))


def mean_error_bar(data: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean and confidence interval for sampled data."""

    try:
        from scipy import stats  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scipy is required for confidence intervals. Install via `pip install scipy`."
                         ) from exc

    mean = np.mean(data, axis=0)
    down, up = stats.t.interval(
        confidence=confidence,
        df=len(data) - 1,
        loc=mean,
        scale=stats.sem(data, axis=0),
    )
    return mean, down, up

