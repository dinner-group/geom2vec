"""Backward-compatible utilities for VAMP (deprecated)."""

from __future__ import annotations

import warnings

from .ops import (
    calculate_inverse,
    compute_covariance_matrix,
    compute_covariance_matrix_stop,
    eig_decomposition,
    estimate_c_tilde_matrix,
    estimate_koopman_matrix,
    rao_blackwell_ledoit_wolf,
)
from .plotting import ContourPlot2D
from .random import set_random_seed
from .stats import empirical_correlation, mean_error_bar
from .tensor import map_data_to_tensor

warnings.warn(
    "`geom2vec.models.downstream.vamp.utils` is deprecated; import from the dedicated modules "
    "(`ops`, `plotting`, `stats`, `tensor`, `random`).",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ContourPlot2D",
    "calculate_inverse",
    "compute_covariance_matrix",
    "compute_covariance_matrix_stop",
    "eig_decomposition",
    "estimate_c_tilde_matrix",
    "estimate_koopman_matrix",
    "rao_blackwell_ledoit_wolf",
    "empirical_correlation",
    "mean_error_bar",
    "set_random_seed",
    "map_data_to_tensor",
]

