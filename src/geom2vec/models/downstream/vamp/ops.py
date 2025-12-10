"""Linear algebra and Koopman operators for VAMP models."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def eig_decomposition(matrix: torch.Tensor, epsilon: float = 1e-6, mode: str = "regularize") -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigen-decomposition for (near) Hermitian positive semi-definite matrices.

    Applies either regularisation or truncation to handle rank-deficient matrices.
    """

    if mode == "regularize":
        identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        matrix = matrix + epsilon * identity
        eigval, eigvec = torch.linalg.eigh(matrix)
        eigval = torch.abs(eigval)
        eigvec = eigvec.transpose(0, 1)
    elif mode == "trunc":
        eigval, eigvec = torch.linalg.eigh(matrix)
        mask = eigval > epsilon
        eigval = eigval[mask]
        eigvec = eigvec[:, mask].transpose(0, 1)
    else:
        raise ValueError("mode must be 'regularize' or 'trunc'")

    return eigval, eigvec


def calculate_inverse(
    matrix: torch.Tensor,
    epsilon: float = 1e-6,
    return_sqrt: bool = False,
    mode: str = "regularize",
) -> torch.Tensor:
    """Return the inverse (or inverse square root) of a positive semi-definite matrix."""

    eigval, eigvec = eig_decomposition(matrix, epsilon, mode)
    diag = torch.diag(torch.sqrt(1.0 / eigval) if return_sqrt else 1.0 / eigval)
    try:
        inverse = torch.linalg.multi_dot((eigvec.t(), diag, eigvec))
    except RuntimeError:  # pragma: no cover
        inverse = torch.chain_matmul(eigvec.t(), diag, eigvec)
    return inverse


def compute_covariance_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    remove_mean: bool = True,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate instantaneous and time-lagged covariance matrices."""

    batch_size = x.shape[0]
    if weights is not None:
        weights = weights.to(device=x.device, dtype=x.dtype).view(batch_size, 1)
        total_weight = torch.sum(weights)
        if total_weight <= 0:
            raise ValueError("Weights must sum to a positive value.")
        if remove_mean:
            x_mean = torch.sum(weights * x, dim=0, keepdim=True) / total_weight
            y_mean = torch.sum(weights * y, dim=0, keepdim=True) / total_weight
            x_centered = x - x_mean
            y_centered = y - y_mean
        else:
            x_centered = x
            y_centered = y
        cov_01 = torch.matmul(x_centered.transpose(0, 1), weights * y_centered) / total_weight
        cov_00 = torch.matmul(x_centered.transpose(0, 1), weights * x_centered) / total_weight
        cov_11 = torch.matmul(y_centered.transpose(0, 1), weights * y_centered) / total_weight
        return cov_00, cov_01, cov_11

    if remove_mean:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    x_t = x.transpose(0, 1)
    y_t = y.transpose(0, 1)
    factor = 1.0 / (batch_size - 1)

    cov_01 = factor * torch.matmul(x_t, y)
    cov_00 = factor * torch.matmul(x_t, x)
    cov_11 = factor * torch.matmul(y_t, y)
    return cov_00, cov_01, cov_11


def compute_covariance_matrix_stop(
    x: torch.Tensor,
    y: torch.Tensor,
    ind_stop: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Covariance estimator respecting stopping indicators (boundary conditions)."""

    batch_size = x.shape[0]
    x_t = x.transpose(0, 1)
    y_t = y.transpose(0, 1)
    factor = 1.0 / (batch_size - 1)
    cov_01 = factor * torch.matmul(x_t, ind_stop * y)
    cov_00 = factor * torch.matmul(x_t, x)
    cov_11 = factor * torch.matmul(y_t, y)
    return cov_00, cov_01, cov_11


def estimate_koopman_matrix(
    data: torch.Tensor,
    data_lagged: torch.Tensor,
    ind_stop: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    mode: str = "regularize",
    symmetrized: bool = False,
    vampe: bool = False,
    remove_mean: bool = True,
    weights: Optional[torch.Tensor] = None,
    return_svd: bool = False,
):
    """Estimate Koopman operator using covariance factorisation."""

    if ind_stop is not None:
        if weights is not None:
            raise ValueError("Sample weights with stopping indicators are not supported.")
        cov_00, cov_01, cov_11 = compute_covariance_matrix_stop(data, data_lagged, ind_stop)
    else:
        cov_00, cov_01, cov_11 = compute_covariance_matrix(
            data,
            data_lagged,
            remove_mean=remove_mean,
            weights=weights,
        )

    if not symmetrized:
        c00_sqrt_inv = calculate_inverse(cov_00, epsilon=epsilon, return_sqrt=True, mode=mode)
        c11_sqrt_inv = calculate_inverse(cov_11, epsilon=epsilon, return_sqrt=True, mode=mode)
        try:
            koopman_matrix = torch.linalg.multi_dot((c00_sqrt_inv, cov_01, c11_sqrt_inv)).t()
        except RuntimeError:  # pragma: no cover
            koopman_matrix = torch.chain_matmul(c00_sqrt_inv, cov_01, c11_sqrt_inv).t()
    else:
        cov_0 = 0.5 * (cov_00 + cov_11)
        cov_1 = 0.5 * (cov_01 + cov_01.t())
        cov_0_sqrt_inv = calculate_inverse(cov_0, epsilon=epsilon, return_sqrt=True, mode=mode)
        try:
            koopman_matrix = torch.linalg.multi_dot((cov_0_sqrt_inv, cov_1, cov_0_sqrt_inv)).t()
        except RuntimeError:  # pragma: no cover
            koopman_matrix = torch.chain_matmul(cov_0_sqrt_inv, cov_1, cov_0_sqrt_inv).t()

    if vampe:
        if ind_stop is not None:
            cov_00, cov_01, cov_11 = compute_covariance_matrix_stop(data, data_lagged, ind_stop)
        else:
            cov_00, cov_01, cov_11 = compute_covariance_matrix(
                data,
                data_lagged,
                remove_mean=remove_mean,
                weights=weights,
            )
        c00_sqrt_inv = calculate_inverse(cov_00, epsilon=epsilon, return_sqrt=True, mode=mode)
        c11_sqrt_inv = calculate_inverse(cov_11, epsilon=epsilon, return_sqrt=True, mode=mode)
        try:
            koopman_matrix = torch.linalg.multi_dot((c00_sqrt_inv, cov_01, c11_sqrt_inv)).t()
        except RuntimeError:  # pragma: no cover
            koopman_matrix = torch.chain_matmul(c00_sqrt_inv, cov_01, c11_sqrt_inv).t()

        u, s, v = torch.svd(koopman_matrix)
        mask = s > epsilon
        u = torch.mm(c00_sqrt_inv, u[:, mask])
        v = torch.mm(c11_sqrt_inv, v[:, mask])
        s = torch.diag(s[mask])

        score = torch.trace(
            2.0 * torch.linalg.multi_dot([s, u.t(), cov_01, v])
            - torch.linalg.multi_dot([s, u.t(), cov_00, u, s, v.t(), cov_11, v])
        ) + 1
        if return_svd:
            trace_cov = torch.unsqueeze(torch.trace(cov_00), dim=0)
            return koopman_matrix, score, (u, torch.diag(s), v, trace_cov)
        return koopman_matrix, score

    if return_svd:
        singular_values = torch.linalg.svdvals(koopman_matrix)
        return koopman_matrix, singular_values

    return koopman_matrix


def estimate_c_tilde_matrix(data: torch.Tensor, data_lagged: torch.Tensor, reversible: bool = True) -> torch.Tensor:
    """Compute the symmetrised covariance matrix C~ used for scoring."""

    cov_00, cov_01, cov_11 = compute_covariance_matrix(data, data_lagged)
    _, cov_10, _ = compute_covariance_matrix(data_lagged, data)
    if reversible:
        c_0 = 0.5 * (cov_00 + cov_11)
        c_1 = 0.5 * (cov_01 + cov_10)
    else:
        c_0 = cov_00
        c_1 = cov_01

    L = torch.linalg.cholesky(c_0)
    L_inv = torch.inverse(L)
    try:
        c_tilde = torch.linalg.multi_dot((L_inv, c_1, L_inv.t()))
    except RuntimeError:  # pragma: no cover
        c_tilde = torch.chain_matmul(L_inv, c_1, L_inv.t())
    return c_tilde


def rao_blackwell_ledoit_wolf(S: np.ndarray, n: int) -> Tuple[np.ndarray, float]:
    """Rao-Blackwellised Ledoit-Wolf shrinkage estimator for covariance matrices."""

    p = len(S)
    if S.shape != (p, p):
        raise ValueError("Input covariance must be square")

    alpha = (n - 2) / (n * (n + 2))
    beta = ((p + 1) * n - 2) / (n * (n + 2))
    trace_S2 = np.sum(S * S)
    U = (p * trace_S2 / np.trace(S) ** 2) - 1
    rho = min(alpha + beta / U, 1)

    F = (np.trace(S) / p) * np.eye(p)
    return (1 - rho) * S + rho * F, rho
