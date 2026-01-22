from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _require_ckdtree():
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError as exc:  # pragma: no cover - scipy is a core dependency
        raise ImportError(
            "scipy is required for mean-field environment features. Install it via `pip install scipy`."
        ) from exc
    return cKDTree


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration for implicit (mean-field) environment descriptors."""

    enabled: bool = False

    # Selections (MDAnalysis string defaults; MDTraj handled via indices in adapters)
    water_selection_mda: str = "resname SOL and name O"
    water_selection_mdtraj: str = "water and name O"
    lipid_head_selection_mda: Optional[str] = None
    lipid_tail_selection_mda: Optional[str] = None
    lipid_head_selection_mdtraj: Optional[str] = None
    lipid_tail_selection_mdtraj: Optional[str] = None

    # RBF / cutoff (Å)
    r_max_A: float = 10.0
    n_rbf: int = 16
    rbf_width_A: float = 0.5
    rbf_centers_A: Optional[np.ndarray] = None

    # Which groups to compute
    compute_water: bool = True
    compute_water_vectors: bool = False
    compute_lipids: bool = False
    compute_lipid_vectors: bool = False
    compute_lipid_tail: bool = False
    compute_lipid_tail_vectors: bool = False
    compute_membrane_frame: bool = False

    # Periodic boundary conditions (orthorhombic only via box lengths)
    use_pbc: bool = True

    # Temporal smoothing (mean-field)
    ema_alpha: Optional[float] = None

    # Membrane depth (only used when compute_membrane_frame=True)
    normalize_membrane_depth: bool = True


@dataclass(frozen=True)
class EnvironmentFeatures:
    env_scalar: np.ndarray  # (N, C_env_s)
    env_vector: Optional[np.ndarray] = None  # (N, 3, C_env_v)
    membrane_normal: Optional[np.ndarray] = None  # (3,)
    membrane_center: Optional[np.ndarray] = None  # (3,)


def rbf_centers_A(config: EnvironmentConfig) -> np.ndarray:
    """Return RBF centers in Å as a (K,) float32 array."""

    if config.n_rbf <= 0:
        raise ValueError("EnvironmentConfig.n_rbf must be positive.")
    if config.r_max_A <= 0:
        raise ValueError("EnvironmentConfig.r_max_A must be positive.")

    centers = config.rbf_centers_A
    if centers is None:
        centers = np.linspace(0.0, float(config.r_max_A), int(config.n_rbf), dtype=np.float32)
    else:
        centers = np.asarray(centers, dtype=np.float32).reshape(-1)
    if centers.size == 0:
        raise ValueError("rbf_centers_A must be non-empty when provided.")
    return centers


def _box_lengths_A(box: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Extract orthorhombic box lengths (Å) from common box formats."""

    if box is None:
        return None
    arr = np.asarray(box, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    lengths = arr[:3]
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        return None
    return lengths


def _gaussian_rbf(distances: np.ndarray, centers: np.ndarray, width_A: float) -> np.ndarray:
    if width_A <= 0:
        raise ValueError("EnvironmentConfig.rbf_width_A must be positive.")
    d = np.asarray(distances, dtype=np.float32)
    c = np.asarray(centers, dtype=np.float32)
    return np.exp(-((d[..., None] - c[None, ...]) ** 2) / (2.0 * float(width_A) ** 2))


def compute_rbf_features(
    token_xyz_A: np.ndarray,
    source_xyz_A: np.ndarray,
    *,
    config: EnvironmentConfig,
    box_A: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scalar RBF densities and vector displacement moments around tokens.

    Returns
    -------
    (rho, moment)
        - rho: (N, K) float32
        - moment: (N, 3, K) float32
    """

    token_xyz = np.asarray(token_xyz_A, dtype=np.float32)
    if token_xyz.ndim != 2 or token_xyz.shape[1] != 3:
        raise ValueError("token_xyz_A must have shape (N, 3).")

    source_xyz = np.asarray(source_xyz_A, dtype=np.float32)
    centers = rbf_centers_A(config)
    rho = np.zeros((token_xyz.shape[0], centers.shape[0]), dtype=np.float32)
    moment = np.zeros((token_xyz.shape[0], 3, centers.shape[0]), dtype=np.float32)

    if source_xyz.size == 0:
        return rho, moment
    if source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("source_xyz_A must have shape (M, 3).")

    cutoff = float(config.r_max_A)
    box_lengths = _box_lengths_A(box_A) if config.use_pbc else None
    cKDTree = _require_ckdtree()

    if box_lengths is None:
        tree = cKDTree(source_xyz)
        neighbors = tree.query_ball_point(token_xyz, r=cutoff)
    else:
        box = box_lengths.astype(np.float32)
        token_xyz = np.mod(token_xyz, box[None, :])
        source_xyz = np.mod(source_xyz, box[None, :])
        tree = cKDTree(source_xyz, boxsize=box)
        neighbors = tree.query_ball_point(token_xyz, r=cutoff)

    if box_lengths is None:
        for i, idx in enumerate(neighbors):
            if not idx:
                continue
            delta = source_xyz[np.asarray(idx, dtype=int)] - token_xyz[i]
            dist = np.linalg.norm(delta, axis=1)
            weights = _gaussian_rbf(dist, centers, config.rbf_width_A)
            rho[i] = weights.sum(axis=0)
            moment[i] = delta.T @ weights
        return rho, moment

    box = box_lengths.astype(np.float32)
    for i, idx in enumerate(neighbors):
        if not idx:
            continue
        delta = source_xyz[np.asarray(idx, dtype=int)] - token_xyz[i]
        delta = delta - box[None, :] * np.round(delta / box[None, :])
        dist = np.linalg.norm(delta, axis=1)
        weights = _gaussian_rbf(dist, centers, config.rbf_width_A)
        rho[i] = weights.sum(axis=0)
        moment[i] = delta.T @ weights
    return rho, moment


def compute_water_rbf_density(
    token_xyz_A: np.ndarray,
    water_xyz_A: np.ndarray,
    *,
    config: EnvironmentConfig,
    box_A: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-token water density features using Gaussian RBF bins.

    Parameters
    ----------
    token_xyz_A:
        Token coordinates in Å, shape (N, 3).
    water_xyz_A:
        Water oxygen coordinates in Å, shape (M, 3).
    config:
        Environment feature configuration.
    box_A:
        Optional box information (Å). Supports (3,) lengths or (6,) lengths+angles.

    Returns
    -------
    np.ndarray
        env_scalar array of shape (N, K) where K is the number of RBF bins.
    """
    rho, _ = compute_water_rbf_features(token_xyz_A, water_xyz_A, config=config, box_A=box_A)
    return rho


def compute_water_rbf_features(
    token_xyz_A: np.ndarray,
    water_xyz_A: np.ndarray,
    *,
    config: EnvironmentConfig,
    box_A: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-token scalar + vector mean-field features from water oxygens.

    Scalars:
        RBF densities: ``rho[i, k] = Σ_s φ_k(||r_s - r_i||)``.
    Vectors:
        Displacement moments: ``m[i, :, k] = Σ_s φ_k(d_is) * (r_s - r_i)``.

    Returns
    -------
    (rho, moment)
        - rho: (N, K) float32
        - moment: (N, 3, K) float32
    """

    return compute_rbf_features(token_xyz_A, water_xyz_A, config=config, box_A=box_A)


def compute_membrane_frame(
    headgroup_xyz_A: np.ndarray,
    *,
    config: EnvironmentConfig,
    box_A: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate a membrane normal and center from lipid headgroup coordinates.

    The normal is defined up to sign by PCA; we fix the sign deterministically by
    orienting the normal so that the headgroup with maximal |projection| has a
    positive projection.

    Returns
    -------
    (normal, center, thickness)
        - normal: (3,) unit vector (float32)
        - center: (3,) membrane center (float32)
        - thickness: float (Å), non-negative
    """

    head_xyz = np.asarray(headgroup_xyz_A, dtype=np.float32)
    if head_xyz.ndim != 2 or head_xyz.shape[1] != 3:
        raise ValueError("headgroup_xyz_A must have shape (M, 3).")
    if head_xyz.shape[0] < 3:
        raise ValueError("At least 3 headgroup points are required to estimate a membrane frame.")

    box_lengths = _box_lengths_A(box_A) if config.use_pbc else None
    if box_lengths is not None:
        mean = head_xyz.mean(axis=0, keepdims=True)
        delta = head_xyz - mean
        box = box_lengths.astype(np.float32)
        delta = delta - box[None, :] * np.round(delta / box[None, :])
        head_xyz = mean + delta

    mean = head_xyz.mean(axis=0)
    centered = head_xyz - mean[None, :]
    cov = (centered.T @ centered) / float(head_xyz.shape[0])
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, int(np.argmin(evals))].astype(np.float32, copy=False)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm <= 1e-8:
        raise ValueError("Failed to estimate a stable membrane normal from headgroup coordinates.")
    normal = normal / normal_norm

    projections = centered @ normal
    ref_idx = int(np.argmax(np.abs(projections)))
    if projections[ref_idx] < 0:
        normal = -normal
        projections = -projections

    median = float(np.median(projections))
    high = projections >= median
    low = ~high
    if high.sum() == 0 or low.sum() == 0:
        high = projections >= 0.0
        low = projections < 0.0
    if high.sum() == 0 or low.sum() == 0:
        # Degenerate: treat as a single leaflet; thickness undefined.
        return normal, mean.astype(np.float32), 0.0

    c_high = head_xyz[high].mean(axis=0)
    c_low = head_xyz[low].mean(axis=0)
    diff = c_high - c_low
    if box_lengths is not None:
        box = box_lengths.astype(np.float32)
        diff = diff - box * np.round(diff / box)

    thickness = float(np.linalg.norm(diff))
    if thickness <= 1e-8 or not np.isfinite(thickness):
        return normal, ((c_high + c_low) / 2.0).astype(np.float32), 0.0

    normal2 = (diff / thickness).astype(np.float32)
    # Ensure consistency with the PCA-oriented normal.
    if float(np.dot(normal2, normal)) < 0.0:
        normal2 = -normal2

    center = ((c_high + c_low) / 2.0).astype(np.float32)
    return normal2, center, thickness


def compute_membrane_depth(
    token_xyz_A: np.ndarray,
    *,
    membrane_center_A: np.ndarray,
    membrane_normal: np.ndarray,
    membrane_thickness_A: Optional[float] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute per-token signed membrane depth scalars.

    Returns a float32 array of shape (N, 1).
    """

    token_xyz = np.asarray(token_xyz_A, dtype=np.float32)
    if token_xyz.ndim != 2 or token_xyz.shape[1] != 3:
        raise ValueError("token_xyz_A must have shape (N, 3).")

    center = np.asarray(membrane_center_A, dtype=np.float32).reshape(3)
    normal = np.asarray(membrane_normal, dtype=np.float32).reshape(3)
    n_norm = float(np.linalg.norm(normal))
    if not np.isfinite(n_norm) or n_norm <= 1e-8:
        raise ValueError("membrane_normal must be a non-zero finite vector.")
    normal = normal / n_norm

    depth = (token_xyz - center[None, :]) @ normal
    if normalize and membrane_thickness_A is not None:
        thickness = float(membrane_thickness_A)
        if np.isfinite(thickness) and thickness > 1e-8:
            depth = (2.0 * depth) / thickness
    return depth.astype(np.float32)[:, None]
