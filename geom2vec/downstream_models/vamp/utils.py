import os
import random

import numpy as np
import scipy
import torch
from six import string_types


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def empirical_correlation(x, y):
    x_remove_mean = x - x.mean()
    y_remove_mean = y - y.mean()
    corr = np.mean(x_remove_mean * y_remove_mean) / (
            np.sqrt(np.mean(x_remove_mean * x_remove_mean))
            * np.sqrt(np.mean(y_remove_mean * y_remove_mean))
    )
    return np.abs(corr)


def mean_error_bar(data, confidence=0.95):
    mean = np.mean(data, axis=0)
    down, up = scipy.stats.t.interval(
        confidence=confidence,
        df=len(data) - 1,
        loc=np.mean(data, axis=0),
        scale=scipy.stats.sem(data, axis=0),
    )
    return mean, down, up


def rao_blackwell_ledoit_wolf(S, n):
    """Rao-Blackwellized Ledoit-Wolf shrinkaged estimator of the covariance matrix.

    Parameters
    ----------
    S : array, shape=(n, n)
        Sample covariance matrix (e.g. estimated with np.cov(X.T))

    n : int
        Number of data points.

    Returns
    -------
    sigma : array, shape=(n, n)

    References
    ----------
    .. [1] Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. "Shrinkage
        estimation of high dimensional covariance matrices" ICASSP (2009)
    """

    p = len(S)
    assert S.shape == (p, p)

    alpha = (n - 2) / (n * (n + 2))
    beta = ((p + 1) * n - 2) / (n * (n + 2))

    trace_S2 = np.sum(S * S)  # np.trace(S.dot(S))
    U = (p * trace_S2 / np.trace(S) ** 2) - 1
    rho = min(alpha + beta / U, 1)

    F = (np.trace(S) / p) * np.eye(p)

    return (1 - rho) * S + rho * F, rho


class ContourPlot2D:
    def __init__(
            self,
            bw_method="scotts",
            num_grids=120,
            cut=3,
            clip=None,
            temperature=310.0,
            shade=True,
            alpha=0.6,
            vmin=0,
            vmax=7,
            n_levels=15,
    ):
        self._bw_method = bw_method
        self._num_grids = num_grids
        self._cut = cut
        if clip is None:
            self._clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
        self._temperature = temperature
        self._shade = shade
        self._alpha = alpha
        self._vmin = vmin
        self._vmax = vmax
        self._n_levels = n_levels

    def _kde_support(self, data, bw, num_grids, cut, clip):
        support_min = max(data.min() - bw * cut, clip[0])
        support_max = min(data.max() + bw * cut, clip[1])

        return np.linspace(support_min, support_max, num_grids)

    def _scipy_bivariate_kde(self, data, bw_method, num_grids, cut, clip):
        from scipy import stats

        kde = stats.gaussian_kde(data.T)
        std = data.std(axis=0, ddof=1)

        if isinstance(bw_method, string_types):
            bw_x = getattr(kde, "%s_factor" % bw_method)() * std[0]
            bw_y = getattr(kde, "%s_factor" % bw_method)() * std[1]
        else:
            raise ValueError("Please input the string of a valid bandwidth method.")

        x_support = self._kde_support(data[:, 0], bw_x, num_grids, cut, clip[0])
        y_support = self._kde_support(data[:, 1], bw_y, num_grids, cut, clip[1])

        xx, yy = np.meshgrid(x_support, y_support)
        z = kde([xx.ravel(), yy.ravel()]).reshape(xx.shape)

        return xx, yy, z

    def _thermo_transform(self, z, temperature):
        from scipy.constants import Avogadro, Boltzmann, calorie_th

        THERMO_CONSTANT = 10 ** -3 * Boltzmann * Avogadro / calorie_th

        return -THERMO_CONSTANT * temperature * np.log(z)

    def plot(
            self,
            data,
            ax=None,
            cbar=True,
            cbar_kwargs={},
            xlabel=None,
            ylabel=None,
            labelsize=10,
    ):
        from matplotlib import pyplot as plt

        if ax is None:
            ax = plt.gca()

        X, Y, Z = self._scipy_bivariate_kde(
            data, self._bw_method, self._num_grids, self._cut, self._clip
        )
        Z = self._thermo_transform(Z, self._temperature)

        if self._vmin is None:
            self._vmin = -1e-12
        if self._vmax is None:
            self._vmax = np.percentile(Z, 50)

        if self._shade:
            cf = ax.contourf(
                X,
                Y,
                Z - Z.min(),
                levels=np.linspace(self._vmin, self._vmax, self._n_levels),
                alpha=self._alpha,
                zorder=1,
                vmin=self._vmin,
                vmax=self._vmax,
            )

        cs = ax.contour(
            X,
            Y,
            Z - Z.min(),
            cmap=plt.get_cmap("bone_r"),
            levels=np.linspace(self._vmin, self._vmax, self._n_levels),
            alpha=1,
            zorder=1,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        if cbar:
            if self._shade:
                cbar = plt.colorbar(cf, **cbar_kwargs)
            else:
                cbar = plt.colorbar(cs, **cbar_kwargs)

            cbar.ax.tick_params(labelsize=labelsize)
            cbar.set_label("Free energy (kcal/mol)", fontsize=labelsize)

        ax.grid(zorder=0)

        if xlabel:
            ax.set_xlabel(xlabel, size=labelsize)
        if ylabel:
            ax.set_ylabel(ylabel, size=labelsize)

        return ax


def eig_decomposition(matrix, epsilon=1e-6, mode="regularize"):
    """This method can be applied to do the eig-decomposition for a rank deficient hermetian matrix,
    this method will be further used to estimate koopman matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        The hermitian matrix: specifically, the covariance matrix.
    epsilon : float, default = 1e-6
        The regularization/trunction parameters for eigenvalues.
    mode : str, default = 'regularize'
        'regularize': regularize the eigenvalues by adding epsilon.
        'trunc': truncate the eigenvalues by filtering out the eigenvalues below epsilon.

    Returns
    -------
    (eigval, eigvec) : Tuple[torch.Tensor, torch.Tensor]
        Eigenvalues and eigenvectors.
    """
    # matrix = matrix.to(torch.float64)
    if mode == "regularize":
        identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        matrix = matrix + epsilon * identity
        eigval, eigvec = torch.linalg.eigh(matrix)
        eigval = torch.abs(eigval)
        eigvec = eigvec.transpose(0, 1)  # row -> column

    elif mode == "trunc":
        eigval, eigvec = torch.linalg.eigh(matrix)
        eigvec = eigvec.transpose(0, 1)
        mask = eigval > epsilon
        eigval = eigval[mask]
        eigvec = eigvec[mask]

    else:
        raise ValueError("Mode is not included")

    return eigval, eigvec


def calculate_inverse(matrix, epsilon=1e-6, return_sqrt=False, mode="regularize"):
    """This method can be applied to compute the inverse or the square-root of the inverse of the matrix,
    this method will be further used to estimate koopman matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix to be inverted.
    epsilon : float, default = 1e-6
        The regularization/trunction parameters for eigenvalues.
    return_sqrt : boolean, optional, default = False
        If True, the square root of the inverse matrix is returned instead.
    mode : str, default = 'regularize'
        'regularize': regularize the eigenvalues by adding epsilon.
        'trunc': truncate the eigenvalues by filtering out the eigenvalues below epsilon.

    Returns
    -------
    inverse : torch.Tensor
        Inverse of the matrix.
    """
    eigval, eigvec = eig_decomposition(matrix, epsilon, mode)

    if return_sqrt:
        diag = torch.diag(torch.sqrt(1.0 / eigval))
    else:
        diag = torch.diag(1.0 / eigval)

    try:
        inverse = torch.linalg.multi_dot((eigvec.t(), diag, eigvec))
    except:
        inverse = torch.chain_matmul(eigvec.t(), diag, eigvec)

    return inverse


def compute_covariance_matrix(x: torch.Tensor, y: torch.Tensor, remove_mean=True):
    """This method can be applied to compute the covariance matrix from two batches of data.

    Parameters
    ----------
    x : torch.Tensor
        The first batch of data of shape [batch_size, num_basis].
    y : torch.Tensor
        The second batch of data of shape [batch_size, num_basis].
    remove_mean : boolean, optional, default = True
        Whether to remove mean of the data.

    Returns
    -------
    (cov_00, cov_01, cov11) : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Instantaneous covariance matrix of x, time-lagged covariance matrix of x and y,
        and instantaneous covariance matrix of y.
    """

    batch_size = x.shape[0]

    if remove_mean:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    y_t = y.transpose(0, 1)
    x_t = x.transpose(0, 1)

    cov_01 = 1 / (batch_size - 1) * torch.matmul(x_t, y)
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x_t, x)
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y_t, y)

    return cov_00, cov_01, cov_11


def compute_covariance_matrix_stop(
        x: torch.Tensor, y: torch.Tensor, ind_stop: torch.Tensor
):
    """This method can be applied to compute the covariance matrix from two batches of data.

    Parameters
    ----------
    x : torch.Tensor
        The first batch of data of shape [batch_size, num_basis].
    y : torch.Tensor
        The second batch of data of shape [batch_size, num_basis].
    ind_stop : torch.Tensor


    Returns
    -------
    (cov_00, cov_01, cov11) : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Instantaneous covariance matrix of x, time-lagged covariance matrix of x and y,
        and instantaneous covariance matrix of y.
    """

    batch_size = x.shape[0]

    y_t = y.transpose(0, 1)
    x_t = x.transpose(0, 1)

    cov_01 = 1 / (batch_size - 1) * torch.matmul(x_t, ind_stop * y)
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x_t, x)
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y_t, y)

    return cov_00, cov_01, cov_11


def estimate_koopman_matrix(
        data: torch.Tensor,
        data_lagged: torch.Tensor,
        ind_stop: torch.Tensor = None,
        epsilon=1e-6,
        mode="regularize",
        symmetrized=False,
        vampe=False
):
    """This method can be applied to compute the koopman matrix from time-instant and time-lagged data.

    Parameters
    ----------
    data : torch.Tensor
        The time-instant data of shape [batch_size, num_basis].
    data_lagged : torch.Tensor
        The time-lagged data of shape [batch_size, num_basis].
    ind_stop : torch.Tensor, default = None,
        The indicator of the data. (B.C.)
    epsilon : float, default = 1e-6
        The regularization/trunction parameters for eigenvalues.
    mode : str, default = 'regularize'
        'regularize': regularize the eigenvalues by adding epsilon.
        'trunc': truncate the eigenvalues by filtering out the eigenvalues below epsilon.
    vampe: bool, default = False
        Constructing the koopman matrix for VAMP-E score.
    Returns
    -------
    koopman_matrix : torch.Tensor
        The koopman matrix of shape [num_basis, num_basis].
    """
    if ind_stop is not None:
        cov_00, cov_01, cov_11 = compute_covariance_matrix_stop(
            data, data_lagged, ind_stop
        )
    else:
        cov_00, cov_01, cov_11 = compute_covariance_matrix(data, data_lagged)

    if not symmetrized:
        cov_00_sqrt_inverse = calculate_inverse(
            cov_00, epsilon=epsilon, return_sqrt=True, mode=mode
        )
        cov_11_sqrt_inverse = calculate_inverse(
            cov_11, epsilon=epsilon, return_sqrt=True, mode=mode
        )
        try:
            koopman_matrix = torch.linalg.multi_dot(
                (cov_00_sqrt_inverse, cov_01, cov_11_sqrt_inverse)
            ).t()
        except:
            koopman_matrix = torch.chain_matmul(
                cov_00_sqrt_inverse, cov_01, cov_11_sqrt_inverse
            ).t()
    else:
        cov_0 = 0.5 * (cov_00 + cov_11)
        cov_1 = 0.5 * (cov_01 + cov_01.t())
        cov_0_sqrt_inverse = calculate_inverse(
            cov_0, epsilon=epsilon, return_sqrt=True, mode=mode
        )
        try:
            koopman_matrix = torch.linalg.multi_dot(
                (cov_0_sqrt_inverse, cov_1, cov_0_sqrt_inverse)
            ).t()
        except:
            koopman_matrix = torch.chain_matmul(
                (cov_0_sqrt_inverse, cov_1, cov_0_sqrt_inverse)
            ).t()

    if vampe:
        if ind_stop is not None:
            cov_00, cov_01, cov_11 = compute_covariance_matrix_stop(
                data, data_lagged, ind_stop
            )
        else:
            cov_00, cov_01, cov_11 = compute_covariance_matrix(data, data_lagged)

        c00_sqrt_inv = calculate_inverse(cov_00, epsilon=epsilon, return_sqrt=True, mode=mode)
        c11_sqrt_inv = calculate_inverse(cov_11, epsilon=epsilon, return_sqrt=True, mode=mode)
        try:
            koopman_matrix = torch.linalg.multi_dot((c00_sqrt_inv, cov_01, c11_sqrt_inv)).t()
        except:
            koopman_matrix = torch.chain_matmul(c00_sqrt_inv, cov_01, c11_sqrt_inv).t()

        u, s, v = torch.svd(koopman_matrix)
        mask = s > epsilon

        u = torch.mm(c00_sqrt_inv, u[:, mask])
        v = torch.mm(c11_sqrt_inv, v[:, mask])
        s = s[mask]

        u_t = u.t()
        v_t = v.t()
        s = torch.diag(s)
        score = torch.trace(
            2. * torch.linalg.multi_dot([s, u_t, cov_01, v]) - torch.linalg.multi_dot([s, u_t, cov_00, u, s, v_t, cov_11, v])
        ) + 1
        return koopman_matrix, score

    return koopman_matrix


def estimate_c_tilde_matrix(
        data: torch.Tensor, data_lagged: torch.Tensor, reversible=True
):
    """This method can be applied to compute the C\tilde matrix from time-instant and time-lagged data.

    Parameters
    ----------
    data : torch.Tensor
        The time-instant data of shape [batch_size, num_basis].
    data_lagged : torch.Tensor
        The time-lagged data of shape [batch_size, num_basis].

    Returns
    -------
    C_tilde : torch.Tensor
        The C tilde matrix of shape [num_basis, num_basis].
    """

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
        C_tilde = torch.linalg.multi_dot((L_inv, c_1, L_inv.t()))
    except:
        C_tilde = torch.chain_matmul(L_inv, c_1, L_inv.t())

    return C_tilde


def map_data_to_tensor(data, device=None, dtype=np.float32):
    """This function is used to yield the torch.Tensor type data from multiple trajectories without to-device.

    Parameters
    ----------
    data : list or tuple or ndarray
        The trajectories of data.
    device : torch device, default = None
        The device on which the torch modules are executed.
    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.

    Returns
    -------
    x : torch.Tensor
        The mapped data.
    """

    with torch.no_grad():
        if not isinstance(data, (list, tuple)):
            data = [data]
        for x in data:
            if isinstance(x, torch.Tensor):
                x = x
            else:
                x = torch.from_numpy(np.asarray(x, dtype=dtype).copy())
            yield x
