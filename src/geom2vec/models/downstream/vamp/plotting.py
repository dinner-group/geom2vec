"""Plotting helpers for VAMP analysis."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class ContourPlot2D:
    """Matplotlib-based free-energy contour plotter with KDE smoothing."""

    def __init__(
        self,
        bw_method: str = "scotts",
        num_grids: int = 120,
        cut: int = 3,
        clip: Optional[np.ndarray] = None,
        temperature: float = 310.0,
        shade: bool = True,
        alpha: float = 0.6,
        vmin: Optional[float] = 0,
        vmax: Optional[float] = 7,
        n_levels: int = 15,
    ) -> None:
        self._bw_method = bw_method
        self._num_grids = num_grids
        self._cut = cut
        self._clip = clip if clip is not None else [(-np.inf, np.inf), (-np.inf, np.inf)]
        self._temperature = temperature
        self._shade = shade
        self._alpha = alpha
        self._vmin = vmin
        self._vmax = vmax
        self._n_levels = n_levels

    def _kde_support(self, data: np.ndarray, bw: float, clip: tuple[float, float]) -> np.ndarray:
        support_min = max(float(data.min() - bw * self._cut), clip[0])
        support_max = min(float(data.max() + bw * self._cut), clip[1])
        return np.linspace(support_min, support_max, self._num_grids)

    def _scipy_bivariate_kde(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            from scipy import stats  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("scipy is required for contour plotting. Install via `pip install scipy`."
                             ) from exc

        kde = stats.gaussian_kde(data.T)
        std = data.std(axis=0, ddof=1)

        if isinstance(self._bw_method, str):
            bw_x = getattr(kde, f"{self._bw_method}_factor")() * std[0]
            bw_y = getattr(kde, f"{self._bw_method}_factor")() * std[1]
        else:
            raise ValueError("bw_method must be a string identifying a scipy KDE bandwidth method")

        x_support = self._kde_support(data[:, 0], bw_x, self._clip[0])
        y_support = self._kde_support(data[:, 1], bw_y, self._clip[1])

        xx, yy = np.meshgrid(x_support, y_support)
        z = kde([xx.ravel(), yy.ravel()]).reshape(xx.shape)
        return xx, yy, z

    def _thermo_transform(self, z: np.ndarray) -> np.ndarray:
        try:
            from scipy.constants import Avogadro, Boltzmann, calorie_th  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("scipy is required for contour plotting. Install via `pip install scipy`."
                             ) from exc

        thermo_constant = 1e-3 * Boltzmann * Avogadro / calorie_th
        return -thermo_constant * self._temperature * np.log(z)

    def plot(
        self,
        data: np.ndarray,
        ax: Optional[Any] = None,
        cbar: bool = True,
        cbar_kwargs: Optional[Dict[str, Any]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        labelsize: int = 10,
    ):
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("matplotlib is required for plotting. Install via `pip install matplotlib`."
                             ) from exc

        if ax is None:
            ax = plt.gca()

        X, Y, Z = self._scipy_bivariate_kde(data)
        Z = self._thermo_transform(Z)

        vmin = self._vmin if self._vmin is not None else -1e-12
        vmax = self._vmax if self._vmax is not None else np.percentile(Z, 50)

        if self._shade:
            cf = ax.contourf(
                X,
                Y,
                Z - Z.min(),
                levels=np.linspace(vmin, vmax, self._n_levels),
                alpha=self._alpha,
                zorder=1,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            cf = None

        cs = ax.contour(
            X,
            Y,
            Z - Z.min(),
            cmap=plt.get_cmap("bone_r"),
            levels=np.linspace(vmin, vmax, self._n_levels),
            zorder=1,
            vmin=vmin,
            vmax=vmax,
        )

        if cbar:
            cbar_kwargs = cbar_kwargs or {}
            colorbar = plt.colorbar(cf if self._shade else cs, **cbar_kwargs)
            colorbar.ax.tick_params(labelsize=labelsize)
            colorbar.set_label("Free energy (kcal/mol)", fontsize=labelsize)

        ax.grid(zorder=0)
        if xlabel:
            ax.set_xlabel(xlabel, size=labelsize)
        if ylabel:
            ax.set_ylabel(ylabel, size=labelsize)
        return ax

