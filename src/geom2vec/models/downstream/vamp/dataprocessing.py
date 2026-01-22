# geom2vec VAMPNet implementation
#
# This file includes code adapted from xuhuihuang/graphvampnets
# (https://github.com/xuhuihuang/graphvampnets), which is licensed
# under the GNU General Public License v3.0 (GPL-3.0).
#
# Accordingly, this file and the src/geom2vec/models/downstream/vamp/
# directory are distributed under the terms of GPL-3.0.
# See LICENSE-GPL-3.0 in the repository root for details.

import numpy as np
import torch
from typing import Optional

from geom2vec.data.preprocessing import Preprocessing
from .ops import rao_blackwell_ledoit_wolf


class Postprocessing_vac(Preprocessing):
    """Transform the outputs from neural networks to slow CVs.
        Note that this method force the detailed balance constraint,
        which can be used to process the simulation data with sufficient sampling.

    Parameters
    ----------
    lag_time : int
        The lag time used for transformation.

    dtype : dtype, default = np.float32

    shrinkage : boolean, default = True
        To tell whether to do the shrinkaged estimation of covariance matrix.

    n_dims : int, default = None
        The number of slow collective variables to obtain.
    """

    def __init__(self, lag_time=1, dtype=np.float32, shrinkage=True, n_dims=None, backend: Optional[str] = "none"):
        torch_dtype = dtype
        if isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        elif dtype in (np.float32, np.dtype("float32"), float):
            torch_dtype = torch.float32
        elif dtype in (np.float64, np.dtype("float64")):
            torch_dtype = torch.float64
        else:
            raise TypeError("Unsupported dtype {}. Provide torch.float32/64 or numpy equivalent.".format(dtype))

        super().__init__(dtype=torch_dtype, backend=backend)
        self._n_dims = n_dims
        self._lag_time = lag_time
        self._numpy_dtype = np.float32 if torch_dtype == torch.float32 else np.float64
        self._shrinkage = shrinkage

        self._is_fitted = False
        self._mean = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._time_scales = None

    @property
    def shrinkage(self):
        return self._shrinkage

    @shrinkage.setter
    def shrinkage(self, value: bool):
        self._shrinkage = value

    @property
    def lag_time(self):
        return self._lag_time

    @lag_time.setter
    def lag_time(self, value: int):
        self._lag_time = value

    @property
    def mean(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        return self._mean

    @property
    def eigenvalues(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        return self._eigenvalues

    @property
    def eigenvectors(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        return self._eigenvectors

    @property
    def time_scales(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        return self._time_scales

    def fit(self, data):
        """Fit the model for transformation.

        Parameters
        ----------
        data : list or ndarray
        """

        self._mean = self._cal_mean(data)
        self._eigenvalues, self._eigenvectors = self._cal_eigvals_eigvecs(data)
        self._time_scales = -self._lag_time / np.log(np.abs(self._eigenvalues))
        self._is_fitted = True

        return self

    def _cal_mean(self, data):
        dataset = self.create_time_lagged_dataset(data, self._lag_time)
        d0, d1 = map(np.array, zip(*dataset))

        mean = (d0.mean(0) + d1.mean(0)) / 2.0

        return mean

    def _cal_cov_matrices(self, data):
        num_trajs = 1 if not isinstance(data, list) else len(data)
        dataset = self.create_time_lagged_dataset(data, self._lag_time)

        batch_size = len(dataset)
        d0, d1 = map(np.array, zip(*dataset))

        mean = 0.5 * (d0.mean(0) + d1.mean(0))

        d0_rm = d0 - mean
        d1_rm = d1 - mean

        c00 = 1.0 / batch_size * np.dot(d0_rm.T, d0_rm)
        c11 = 1.0 / batch_size * np.dot(d1_rm.T, d1_rm)
        c01 = 1.0 / batch_size * np.dot(d0_rm.T, d1_rm)
        c10 = 1.0 / batch_size * np.dot(d1_rm.T, d0_rm)

        c0 = 0.5 * (c00 + c11)
        c1 = 0.5 * (c01 + c10)

        if self.shrinkage:
            n_observations_ = batch_size + self._lag_time * num_trajs
            c0, _ = rao_blackwell_ledoit_wolf(c0, n_observations_)

        return c0, c1

    def _cal_eigvals_eigvecs(self, data):
        c0, c1 = self._cal_cov_matrices(data)

        import scipy.linalg

        eigvals, eigvecs = scipy.linalg.eigh(c1, b=c0)

        idx = np.argsort(eigvals)[::-1]

        if self._n_dims is not None:
            assert self._n_dims <= len(idx)
            idx = idx[: self._n_dims]

        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        return eigvals, eigvecs

    def transform(self, data):
        """Transfrom the basis funtions (or outputs of neural networks) to the slow CVs.
            Note that the model must be fitted first before transformation.

        Parameters
        ----------
        data : list or ndarray

        Returns
        -------
        modes : list or ndarray
            Slow CVs (i.e., dynamic modes).
        """

        modes = []

        if not self._is_fitted:
            raise ValueError("Please fit the model first")

        data = self._seq_trajs(data)
        num_trajs = len(data)

        for i in range(num_trajs):
            x_rm = data[i] - self._mean
            modes.append(np.dot(x_rm, self._eigenvectors).astype(np.float32))

        return modes if num_trajs > 1 else modes[0]

    def fit_transform(self, data):
        """Fit the model and transfrom to the slow CVs.

        Parameters
        ----------
        data : list or ndarray

        Returns
        -------
        modes : list or ndarray
            Slow CVs (i.e., dynamic modes).
        """

        modes = self.fit(data).transform(data)

        return modes

    def gmrq(self, data):
        """Score the model based on generalized matrix Rayleigh quotient.
            Note that the model should be fitted before computing computing GMRQ score.

        Parameters
        ----------
        data : list or ndarray, optional, default = None

        Returns
        -------
        score : float
            Generalized matrix Rayleigh quotient. This number indicates how
            well the top ``n_timescales+1`` eigenvectors of this tICA model perform
            as slowly decorrelating collective variables for the new data in
            ``sequences``.
        """

        if not self._is_fitted:
            raise ValueError("Please fit the model first")

        A = self._eigenvectors
        ### Q: use the mean of fitted data or input data?
        S, C = self._cal_cov_matrices(data)

        A = A.numpy()
        S = S.numpy()
        C = C.numpy()

        P = A.T.dot(C).dot(A)
        Q = A.T.dot(S).dot(A)

        score = np.trace(P.dot(np.linalg.inv(Q)))

        return score

    def empirical_correlation(self, data):
        """Score the model based on empirical correlations between the instantaneous and time-lagged slowest CVs.
            Note that the model should be fitted before computing empirical correlations.
            The empirical correlations equal to eigenvalues for the fitted equilibrium dataset.

        Parameters
        ----------
        data : list or ndarray

        Returns
        -------
        corr : ndarray
            Empirical correlations between the slowest instantaneous and time-lagged CVs.
        """

        modes = self.transform(data)
        dataset = self.create_time_lagged_dataset(data, self._lag_time)

        modes0, modes1 = map(np.array, zip(*dataset))

        modes0_rm = modes0 - np.mean(modes0, axis=0)
        modes1_rm = modes1 - np.mean(modes1, axis=0)

        corr = np.mean(modes0_rm * modes1_rm, axis=0) / (
            np.std(modes0_rm, axis=0) * np.std(modes1_rm, axis=0)
        )

        return corr


class Postprocessing_vamp(Preprocessing):
    """Transform the outputs from neural networks to slow CVs.
        Note that this method doesn't force the detailed balance constraint,
        which can be used to process the simulation data with insufficient sampling.

    Parameters
    ----------
    lag_time : int
        The lag time used for transformation.

    dtype : dtype, default = np.float32

    n_dims : int, default = None
        The number of slow collective variables to obtain.
    """

    def __init__(self, lag_time=1, dtype=np.float32, n_dims=None):
        if isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        elif dtype in (np.float32, np.dtype("float32"), float):
            torch_dtype = torch.float32
        elif dtype in (np.float64, np.dtype("float64")):
            torch_dtype = torch.float64
        else:
            raise TypeError("Unsupported dtype {}. Provide torch.float32/64 or numpy equivalent.".format(dtype))

        super().__init__(dtype=torch_dtype, backend="none")
        self._n_dims = n_dims
        self._lag_time = lag_time
        self._numpy_dtype = np.float32 if torch_dtype == torch.float32 else np.float64

        self._is_fitted = False
        self._mean_0 = None
        self._mean_t = None
        self._singularvalues = None
        self._left_singularvectors = None
        self._right_singularvectors = None
        self._time_scales = None

    @property
    def lag_time(self):
        return self._lag_time

    @lag_time.setter
    def lag_time(self, value: int):
        self._lag_time = value

    @property
    def mean_0(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        return self._mean_0

    @property
    def mean_t(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        return self._mean_t

    @property
    def singularvalues(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        return self._singularvalues

    @property
    def left_singularvectors(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        return self._left_singularvectors

    @property
    def right_singularvectors(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        return self._right_singularvectors

    @property
    def time_scales(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        return self._time_scales

    def fit(self, data):
        """Fit the model for transformation.

        Parameters
        ----------
        data : list or ndarray
        """

        self._mean_0, self._mean_t = self._cal_mean(data)
        (
            self._singularvalues,
            self._left_singularvectors,
            self._right_singularvectors,
        ) = self._cal_singularvals_singularvecs(data)
        self._time_scales = -self._lag_time / np.log(np.abs(self._singularvalues))

        self._is_fitted = True

        return self

    def _inv_sqrt(self, cov_matrix):
        import numpy.linalg

        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

        eigvals, eigvecs = numpy.linalg.eigh(cov_matrix)
        sort_key = np.abs(eigvals)
        idx = np.argsort(sort_key)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        diag = np.diag(1.0 / np.maximum(np.sqrt(np.maximum(eigvals, 1e-12)), 1e-12))
        inv_sqrt = np.dot(eigvecs, diag)

        return inv_sqrt

    def _cal_mean(self, data):
        dataset = self.create_time_lagged_dataset(data, self._lag_time)
        d0, d1 = map(np.array, zip(*dataset))
        return d0.mean(0), d1.mean(0)

    def _cal_cov_matrices(self, data):
        dataset = self.create_time_lagged_dataset(data, self._lag_time)

        batch_size = len(dataset)

        d0, d1 = map(np.array, zip(*dataset))

        d0_rm = d0 - d0.mean(0)
        d1_rm = d1 - d1.mean(0)

        c00 = 1.0 / batch_size * np.dot(d0_rm.T, d0_rm)
        c11 = 1.0 / batch_size * np.dot(d1_rm.T, d1_rm)
        c01 = 1.0 / batch_size * np.dot(d0_rm.T, d1_rm)

        return c00, c01, c11

    def _cal_singularvals_singularvecs(self, data):
        c00, c01, c11 = self._cal_cov_matrices(data)

        c00_inv_sqrt = self._inv_sqrt(c00)
        c11_inv_sqrt = self._inv_sqrt(c11)

        ks = np.dot(c00_inv_sqrt.T, c01).dot(c11_inv_sqrt)

        import scipy.linalg

        U, s, Vh = scipy.linalg.svd(ks, compute_uv=True, lapack_driver="gesvd")

        left = np.dot(c00_inv_sqrt, U)
        right = np.dot(c11_inv_sqrt, Vh.T)

        idx = np.argsort(s)[::-1]

        if self._n_dims is not None:
            assert self._n_dims <= len(idx)
            idx = idx[: self._n_dims]

        s = s[idx]
        left = left[:, idx]
        right = right[:, idx]

        return s, left, right

    def transform(self, data, instantaneous=True):
        """Transfrom the basis funtions (or outputs of neural networks) to the slow CVs.
            Note that the model must be fitted first before transformation.

        Parameters
        ----------
        data : list or ndarray

        instantaneous : boolean, default = True
            If true, projected onto left singular functions of Koopman operator.
            If false, projected onto right singular functions of Koopman operator.

        Returns
        -------
        modes : list or ndarray
            Slow CVs (i.e., slow dynamic modes).
        """

        modes = []

        if not self._is_fitted:
            raise ValueError("Please fit the model first")

        data = self._seq_trajs(data)
        num_trajs = len(data)

        if instantaneous:
            for i in range(num_trajs):
                x_rm = data[i] - self._mean_0
                modes.append(
                    np.dot(x_rm, self._left_singularvectors).astype(np.float32)
                )
        else:
            for i in range(num_trajs):
                x_rm = data[i] - self._mean_t
                modes.append(
                    np.dot(x_rm, self._right_singularvectors).astype(np.float32)
                )

        return modes if num_trajs > 1 else modes[0]

    def fit_transform(self, data, instantaneous=True):
        """Fit the model and transfrom to the slow CVs.

        Parameters
        ----------
        data : list or ndarray

        Returns
        -------
        modes : list or ndarray
            Slow CVs (i.e., dynamic modes).
        """

        modes = self.fit(data).transform(data, instantaneous=instantaneous)

        return modes


class Postprocessing_stopvamp:
    """Postprocess lobe outputs into CVs for stopping-time (killed) dynamics.

    This class mirrors :class:`Postprocessing_vamp` but uses a per-pair stopping
    indicator when estimating the cross-covariance:

    ``C01_stop = E[x_t (ind_stop * x_{t+lag})^T]``.

    Notes
    -----
    Mean removal is intentionally not applied to match the current stopped
    covariance implementation in :func:`geom2vec.models.downstream.vamp.ops.compute_covariance_matrix_stop`.
    """

    def __init__(self, lag_time: int = 1, dtype=np.float32, n_dims: Optional[int] = None):
        if lag_time <= 0:
            raise ValueError("lag_time must be a positive integer")
        self._lag_time = int(lag_time)
        self._n_dims = n_dims

        if isinstance(dtype, torch.dtype):
            self._torch_dtype = dtype
            self._numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
        elif dtype in (np.float32, np.dtype("float32"), float):
            self._torch_dtype = torch.float32
            self._numpy_dtype = np.float32
        elif dtype in (np.float64, np.dtype("float64")):
            self._torch_dtype = torch.float64
            self._numpy_dtype = np.float64
        else:
            raise TypeError(f"Unsupported dtype {dtype}. Provide torch.float32/64 or numpy equivalent.")

        self._is_fitted = False
        self._singularvalues: Optional[np.ndarray] = None
        self._left_singularvectors: Optional[np.ndarray] = None
        self._right_singularvectors: Optional[np.ndarray] = None
        self._time_scales: Optional[np.ndarray] = None

    @property
    def lag_time(self) -> int:
        return self._lag_time

    @property
    def singularvalues(self) -> np.ndarray:
        if not self._is_fitted or self._singularvalues is None:
            raise ValueError("please fit the model first")
        return self._singularvalues

    @property
    def left_singularvectors(self) -> np.ndarray:
        if not self._is_fitted or self._left_singularvectors is None:
            raise ValueError("please fit the model first")
        return self._left_singularvectors

    @property
    def right_singularvectors(self) -> np.ndarray:
        if not self._is_fitted or self._right_singularvectors is None:
            raise ValueError("please fit the model first")
        return self._right_singularvectors

    @property
    def time_scales(self) -> np.ndarray:
        if not self._is_fitted or self._time_scales is None:
            raise ValueError("please fit the model first")
        return self._time_scales

    def _as_list(self, data):
        if isinstance(data, (list, tuple)):
            return list(data)
        return [data]

    def _to_numpy(self, array) -> np.ndarray:
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy().astype(self._numpy_dtype, copy=False)
        return np.asarray(array, dtype=self._numpy_dtype)

    def _inv_sqrt(self, cov_matrix: np.ndarray) -> np.ndarray:
        import numpy.linalg

        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        eigvals, eigvecs = numpy.linalg.eigh(cov_matrix)
        sort_key = np.abs(eigvals)
        idx = np.argsort(sort_key)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        diag = np.diag(1.0 / np.maximum(np.sqrt(np.maximum(eigvals, 1e-12)), 1e-12))
        return np.dot(eigvecs, diag)

    def _ensure_boundary_mask(self, mask, *, frames: int, name: str) -> np.ndarray:
        tensor = np.asarray(mask)
        if tensor.ndim == 2 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        if tensor.ndim != 1:
            raise ValueError(f"{name} must have shape (frames,) or (frames, 1).")
        if tensor.shape[0] != frames:
            raise ValueError(f"{name} must have length equal to number of frames.")
        return tensor.astype(bool)

    def _cal_cov_matrices(self, data_0, data_t, in_a, in_b):
        from geom2vec.data.features import forward_stop

        cov_00 = None
        cov_11 = None
        cov_01 = None
        total_pairs = 0

        data_0_list = self._as_list(data_0)
        data_t_list = self._as_list(data_t)
        in_a_list = self._as_list(in_a)
        in_b_list = self._as_list(in_b)

        if len(data_0_list) != len(data_t_list):
            raise ValueError("data_0 and data_t must have the same number of trajectories")
        if len(in_a_list) != len(data_0_list) or len(in_b_list) != len(data_0_list):
            raise ValueError("in_a and in_b must match number of trajectories")

        for idx, (x0_traj, xt_traj) in enumerate(zip(data_0_list, data_t_list)):
            x0 = self._to_numpy(x0_traj)
            xt = self._to_numpy(xt_traj)
            if x0.ndim != 2 or xt.ndim != 2:
                raise ValueError("Trajectory outputs must have shape (frames, features)")
            if x0.shape[0] != xt.shape[0]:
                raise ValueError("data_0 and data_t trajectories must have matching frame counts")
            if x0.shape[1] != xt.shape[1]:
                raise ValueError("data_0 and data_t must have the same feature dimension")

            frames = x0.shape[0]
            if frames <= self._lag_time:
                continue

            in_a_mask = self._ensure_boundary_mask(in_a_list[idx], frames=frames, name="in_a")
            in_b_mask = self._ensure_boundary_mask(in_b_list[idx], frames=frames, name="in_b")
            if np.any(np.logical_and(in_a_mask, in_b_mask)):
                raise ValueError("in_a and in_b masks must be disjoint")

            in_domain = np.logical_not(np.logical_or(in_a_mask, in_b_mask))
            exit_time = forward_stop(in_domain)

            t0 = np.arange(frames - self._lag_time)
            t1 = t0 + self._lag_time
            t_stop = np.minimum(t1, exit_time[t0])
            ind_stop = in_domain[t_stop].astype(self._numpy_dtype).reshape(-1, 1)

            x0_pairs = x0[t0]
            xt_pairs = xt[t1]

            if cov_00 is None:
                feature_dim = x0_pairs.shape[1]
                cov_00 = np.zeros((feature_dim, feature_dim), dtype=self._numpy_dtype)
                cov_11 = np.zeros_like(cov_00)
                cov_01 = np.zeros_like(cov_00)

            cov_00 += x0_pairs.T @ x0_pairs
            cov_11 += xt_pairs.T @ xt_pairs
            cov_01 += x0_pairs.T @ (ind_stop * xt_pairs)
            total_pairs += x0_pairs.shape[0]

        if cov_00 is None or total_pairs == 0:
            raise ValueError("At least one trajectory with frames > lag_time is required.")

        scale = 1.0 / float(total_pairs)
        cov_00 *= scale
        cov_11 *= scale
        cov_01 *= scale
        return cov_00, cov_01, cov_11

    def fit(self, data_0, data_t, in_a, in_b):
        cov_00, cov_01, cov_11 = self._cal_cov_matrices(data_0, data_t, in_a, in_b)

        c00_inv_sqrt = self._inv_sqrt(cov_00)
        c11_inv_sqrt = self._inv_sqrt(cov_11)
        ks = np.dot(c00_inv_sqrt.T, cov_01).dot(c11_inv_sqrt)

        import scipy.linalg

        u, s, vh = scipy.linalg.svd(ks, compute_uv=True, lapack_driver="gesvd")
        left = np.dot(c00_inv_sqrt, u)
        right = np.dot(c11_inv_sqrt, vh.T)

        idx = np.argsort(s)[::-1]
        if self._n_dims is not None:
            if self._n_dims <= 0:
                raise ValueError("n_dims must be positive")
            idx = idx[: self._n_dims]

        self._singularvalues = s[idx]
        self._left_singularvectors = left[:, idx]
        self._right_singularvectors = right[:, idx]
        self._time_scales = -self._lag_time / np.log(np.abs(self._singularvalues))

        self._is_fitted = True
        return self

    def transform(self, data_0, data_t, *, instantaneous: bool = True):
        if not self._is_fitted:
            raise ValueError("Please fit the model first")

        data_0_list = self._as_list(data_0)
        data_t_list = self._as_list(data_t)
        if len(data_0_list) != len(data_t_list):
            raise ValueError("data_0 and data_t must have the same number of trajectories")

        vectors = self.left_singularvectors if instantaneous else self.right_singularvectors
        outputs = []
        for x0_traj, xt_traj in zip(data_0_list, data_t_list):
            x = self._to_numpy(x0_traj if instantaneous else xt_traj)
            outputs.append(np.dot(x, vectors).astype(np.float32))

        return outputs if len(outputs) > 1 else outputs[0]

    def fit_transform(self, data_0, data_t, in_a, in_b, *, instantaneous: bool = True):
        return self.fit(data_0, data_t, in_a, in_b).transform(data_0, data_t, instantaneous=instantaneous)


class Postprocessing_stopped_time_vamp:
    """Postprocess lobe outputs into CVs for stopped-time (absorbing) dynamics.

    This variant matches the common "absorbing boundary" construction for short
    trajectories: instead of masking the cross-covariance with a stop indicator,
    we build pairs using the stopped time index

    ``t_stop = min(t + lag_time, tau_exit(t))``

    and estimate standard VAMP covariances on ``(x_t, x_{t_stop})``.
    """

    def __init__(self, lag_time: int = 1, dtype=np.float32, n_dims: Optional[int] = None):
        if lag_time <= 0:
            raise ValueError("lag_time must be a positive integer")
        self._lag_time = int(lag_time)
        self._n_dims = n_dims

        if isinstance(dtype, torch.dtype):
            self._torch_dtype = dtype
            self._numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
        elif dtype in (np.float32, np.dtype("float32"), float):
            self._torch_dtype = torch.float32
            self._numpy_dtype = np.float32
        elif dtype in (np.float64, np.dtype("float64")):
            self._torch_dtype = torch.float64
            self._numpy_dtype = np.float64
        else:
            raise TypeError(f"Unsupported dtype {dtype}. Provide torch.float32/64 or numpy equivalent.")

        self._is_fitted = False
        self._mean_0: Optional[np.ndarray] = None
        self._mean_t: Optional[np.ndarray] = None
        self._singularvalues: Optional[np.ndarray] = None
        self._left_singularvectors: Optional[np.ndarray] = None
        self._right_singularvectors: Optional[np.ndarray] = None
        self._time_scales: Optional[np.ndarray] = None

    @property
    def lag_time(self) -> int:
        return self._lag_time

    @property
    def mean_0(self) -> np.ndarray:
        if not self._is_fitted or self._mean_0 is None:
            raise ValueError("please fit the model first")
        return self._mean_0

    @property
    def mean_t(self) -> np.ndarray:
        if not self._is_fitted or self._mean_t is None:
            raise ValueError("please fit the model first")
        return self._mean_t

    @property
    def singularvalues(self) -> np.ndarray:
        if not self._is_fitted or self._singularvalues is None:
            raise ValueError("please fit the model first")
        return self._singularvalues

    @property
    def left_singularvectors(self) -> np.ndarray:
        if not self._is_fitted or self._left_singularvectors is None:
            raise ValueError("please fit the model first")
        return self._left_singularvectors

    @property
    def right_singularvectors(self) -> np.ndarray:
        if not self._is_fitted or self._right_singularvectors is None:
            raise ValueError("please fit the model first")
        return self._right_singularvectors

    @property
    def time_scales(self) -> np.ndarray:
        if not self._is_fitted or self._time_scales is None:
            raise ValueError("please fit the model first")
        return self._time_scales

    def _as_list(self, data):
        if isinstance(data, (list, tuple)):
            return list(data)
        return [data]

    def _to_numpy(self, array) -> np.ndarray:
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy().astype(self._numpy_dtype, copy=False)
        return np.asarray(array, dtype=self._numpy_dtype)

    def _inv_sqrt(self, cov_matrix: np.ndarray) -> np.ndarray:
        import numpy.linalg

        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        eigvals, eigvecs = numpy.linalg.eigh(cov_matrix)
        sort_key = np.abs(eigvals)
        idx = np.argsort(sort_key)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        diag = np.diag(1.0 / np.maximum(np.sqrt(np.maximum(eigvals, 1e-12)), 1e-12))
        return np.dot(eigvecs, diag)

    def _ensure_boundary_mask(self, mask, *, frames: int, name: str) -> np.ndarray:
        tensor = np.asarray(mask)
        if tensor.ndim == 2 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        if tensor.ndim != 1:
            raise ValueError(f"{name} must have shape (frames,) or (frames, 1).")
        if tensor.shape[0] != frames:
            raise ValueError(f"{name} must have length equal to number of frames.")
        return tensor.astype(bool)

    def _pair_indices(self, *, frames: int, in_a_mask: np.ndarray, in_b_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from geom2vec.data.features import forward_stop

        in_domain = np.logical_not(np.logical_or(in_a_mask, in_b_mask))
        exit_time = forward_stop(in_domain)

        t0 = np.arange(frames - self._lag_time)
        t1 = t0 + self._lag_time
        t_stop = np.minimum(t1, exit_time[t0])
        return t0, t_stop

    def _cal_means(self, data_0, data_t, in_a, in_b) -> tuple[np.ndarray, np.ndarray]:
        data_0_list = self._as_list(data_0)
        data_t_list = self._as_list(data_t)
        in_a_list = self._as_list(in_a)
        in_b_list = self._as_list(in_b)

        if len(data_0_list) != len(data_t_list):
            raise ValueError("data_0 and data_t must have the same number of trajectories")
        if len(in_a_list) != len(data_0_list) or len(in_b_list) != len(data_0_list):
            raise ValueError("in_a and in_b must match number of trajectories")

        sum_0: Optional[np.ndarray] = None
        sum_t: Optional[np.ndarray] = None
        total_pairs = 0

        for idx, (x0_traj, xt_traj) in enumerate(zip(data_0_list, data_t_list)):
            x0 = self._to_numpy(x0_traj)
            xt = self._to_numpy(xt_traj)
            if x0.ndim != 2 or xt.ndim != 2:
                raise ValueError("Trajectory outputs must have shape (frames, features)")
            if x0.shape[0] != xt.shape[0]:
                raise ValueError("data_0 and data_t trajectories must have matching frame counts")
            if x0.shape[1] != xt.shape[1]:
                raise ValueError("data_0 and data_t must have the same feature dimension")

            frames = x0.shape[0]
            if frames <= self._lag_time:
                continue

            in_a_mask = self._ensure_boundary_mask(in_a_list[idx], frames=frames, name="in_a")
            in_b_mask = self._ensure_boundary_mask(in_b_list[idx], frames=frames, name="in_b")
            if np.any(np.logical_and(in_a_mask, in_b_mask)):
                raise ValueError("in_a and in_b masks must be disjoint")

            t0, t_stop = self._pair_indices(frames=frames, in_a_mask=in_a_mask, in_b_mask=in_b_mask)
            x0_pairs = x0[t0]
            xt_pairs = xt[t_stop]

            if sum_0 is None:
                sum_0 = np.zeros((x0_pairs.shape[1],), dtype=self._numpy_dtype)
                sum_t = np.zeros_like(sum_0)

            sum_0 += np.sum(x0_pairs, axis=0)
            sum_t += np.sum(xt_pairs, axis=0)
            total_pairs += x0_pairs.shape[0]

        if sum_0 is None or sum_t is None or total_pairs == 0:
            raise ValueError("At least one trajectory with frames > lag_time is required.")

        mean_0 = sum_0 / float(total_pairs)
        mean_t = sum_t / float(total_pairs)
        return mean_0, mean_t

    def _cal_cov_matrices(self, data_0, data_t, in_a, in_b, *, mean_0: np.ndarray, mean_t: np.ndarray):
        cov_00 = None
        cov_11 = None
        cov_01 = None
        total_pairs = 0

        data_0_list = self._as_list(data_0)
        data_t_list = self._as_list(data_t)
        in_a_list = self._as_list(in_a)
        in_b_list = self._as_list(in_b)

        for idx, (x0_traj, xt_traj) in enumerate(zip(data_0_list, data_t_list)):
            x0 = self._to_numpy(x0_traj)
            xt = self._to_numpy(xt_traj)
            frames = x0.shape[0]
            if frames <= self._lag_time:
                continue

            in_a_mask = self._ensure_boundary_mask(in_a_list[idx], frames=frames, name="in_a")
            in_b_mask = self._ensure_boundary_mask(in_b_list[idx], frames=frames, name="in_b")
            if np.any(np.logical_and(in_a_mask, in_b_mask)):
                raise ValueError("in_a and in_b masks must be disjoint")

            t0, t_stop = self._pair_indices(frames=frames, in_a_mask=in_a_mask, in_b_mask=in_b_mask)
            x0_pairs = x0[t0] - mean_0
            xt_pairs = xt[t_stop] - mean_t

            if cov_00 is None:
                feature_dim = x0_pairs.shape[1]
                cov_00 = np.zeros((feature_dim, feature_dim), dtype=self._numpy_dtype)
                cov_11 = np.zeros_like(cov_00)
                cov_01 = np.zeros_like(cov_00)

            cov_00 += x0_pairs.T @ x0_pairs
            cov_11 += xt_pairs.T @ xt_pairs
            cov_01 += x0_pairs.T @ xt_pairs
            total_pairs += x0_pairs.shape[0]

        if cov_00 is None or cov_11 is None or cov_01 is None or total_pairs == 0:
            raise ValueError("At least one trajectory with frames > lag_time is required.")

        scale = 1.0 / float(total_pairs)
        cov_00 *= scale
        cov_11 *= scale
        cov_01 *= scale
        return cov_00, cov_01, cov_11

    def fit(self, data_0, data_t, in_a, in_b):
        mean_0, mean_t = self._cal_means(data_0, data_t, in_a, in_b)
        cov_00, cov_01, cov_11 = self._cal_cov_matrices(data_0, data_t, in_a, in_b, mean_0=mean_0, mean_t=mean_t)

        c00_inv_sqrt = self._inv_sqrt(cov_00)
        c11_inv_sqrt = self._inv_sqrt(cov_11)
        ks = np.dot(c00_inv_sqrt.T, cov_01).dot(c11_inv_sqrt)

        import scipy.linalg

        u, s, vh = scipy.linalg.svd(ks, compute_uv=True, lapack_driver="gesvd")
        left = np.dot(c00_inv_sqrt, u)
        right = np.dot(c11_inv_sqrt, vh.T)

        idx = np.argsort(s)[::-1]
        if self._n_dims is not None:
            if self._n_dims <= 0:
                raise ValueError("n_dims must be positive")
            idx = idx[: self._n_dims]

        self._mean_0 = mean_0.astype(self._numpy_dtype, copy=False)
        self._mean_t = mean_t.astype(self._numpy_dtype, copy=False)
        self._singularvalues = s[idx]
        self._left_singularvectors = left[:, idx]
        self._right_singularvectors = right[:, idx]
        self._time_scales = -self._lag_time / np.log(np.abs(self._singularvalues))

        self._is_fitted = True
        return self

    def transform(self, data_0, data_t, *, instantaneous: bool = True):
        if not self._is_fitted:
            raise ValueError("Please fit the model first")

        data_0_list = self._as_list(data_0)
        data_t_list = self._as_list(data_t)
        if len(data_0_list) != len(data_t_list):
            raise ValueError("data_0 and data_t must have the same number of trajectories")

        vectors = self.left_singularvectors if instantaneous else self.right_singularvectors
        mean = self.mean_0 if instantaneous else self.mean_t
        outputs = []
        for x0_traj, xt_traj in zip(data_0_list, data_t_list):
            x = self._to_numpy(x0_traj if instantaneous else xt_traj)
            outputs.append(np.dot(x - mean, vectors).astype(np.float32))

        return outputs if len(outputs) > 1 else outputs[0]

    def fit_transform(self, data_0, data_t, in_a, in_b, *, instantaneous: bool = True):
        return self.fit(data_0, data_t, in_a, in_b).transform(data_0, data_t, instantaneous=instantaneous)
