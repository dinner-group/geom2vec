from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from .utils import estimate_koopman_matrix, map_data_to_tensor


class BaseVAMPNet_Model:
    """The base VAMPNet model from VAMPNet estimator.

    Parameters
    ----------
    lobe : nn.Module
        A neural network model which maps the input data to the basis functions.
    lobe_lagged : nn.Module, optional, default = None
        Neural network model for timelagged data, in case of None the lobes are shared (structure and weights).
    device : torch device, default = None
        The device on which the torch modules are executed.
    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.
    """

    def __init__(self, lobe, lobe_lagged=None, device=None, dtype=np.float32):
        # TODO: check the dtype, np format will be deprecated.
        self._lobe = lobe
        self._lobe_lagged = lobe_lagged
        self._dtype = dtype
        self._device = device

        self._set_dtype(self._lobe)
        if self._lobe_lagged is not None:
            self._set_dtype(self._lobe_lagged)

    def _set_dtype(self, model):
        if self._dtype == np.float32:
            model.float()
        elif self._dtype == np.float64:
            model.double()

    @property
    def lobe(self):
        return self._lobe

    @property
    def lobe_lagged(self):
        if self._lobe_lagged is None:
            raise ValueError(
                "There is only one neural network for both time-instant and time-lagged data"
            )
        return self._lobe_lagged

    def transform(
        self,
        data: Union[List, Tuple, np.ndarray],
        instantaneous: bool = True,
        return_cv: bool = False,
        lag_time: Optional[int] = None,
        batch_size: int = 200,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Transform the data through the trained networks.

        Parameters
        ----------
        data : list or tuple or ndarray
            The data to be transformed.
        instantaneous : boolean, default = True
            Whether to use the instantaneous lobe or the time-lagged lobe for transformation.
            Note that only VAMPNet method requires two lobes
        return_cv : boolean, default = False
            Whether to return the transformed data as CVs.
        lag_time : int, optional
            The lag time for transformation to CVs. Required if return_cv is True.
        batch_size : int, default = 200
            The batch size for processing large datasets.

        Returns
        -------
        output : array_like
            List of numpy array or numpy array containing transformed data.
        """
        if instantaneous or self._lobe_lagged is None:
            self._lobe.eval()
            net = self._lobe
        else:
            self._lobe_lagged.eval()
            net = self._lobe_lagged

        output = []
        for data_tensor in map_data_to_tensor(data):
            batch_list = []
            for i in tqdm(range(0, data_tensor.shape[0], batch_size)):
                batch = data_tensor[i : i + batch_size].to(device=self._device)
                batch_list.append(net(batch).detach().cpu().numpy())
            output.append(np.concatenate(batch_list, axis=0))

        if not return_cv:
            return output if len(output) > 1 else output[0]
        else:
            if lag_time is None:
                raise ValueError("Please input the lag time for transformation to CVs")
            return self._transform_to_cv(output, lag_time, instantaneous)

    def _transform_to_cv(self, output, lag_time, instantaneous):
        raise NotImplementedError


class VAMPNet_Estimator:
    def __init__(self, epsilon, mode, symmetrized, score_method='vamp-2'):
        """VAMPNet Estimator for VAMP score calculation.

        Parameters
        ----------
        epsilon : float
            The regularization parameter for matrix inversion.
        mode : str
            The mode for eigenvalue handling. Can be 'regularize' or 'trunc'.
        symmetrized : bool
            Whether to use symmetrized VAMP score calculation.

        score_method: str, default = 'vamp-2'
            The type of vamp score to use. Can be 'vamp-1' or 'vamp-2', or 'vamp-e'

        Attributes
        ----------
        _score : float
            The current VAMP score.
        _score_list : list
            List of VAMP scores during training.
        _epsilon : float
            The regularization parameter.
        _mode : str
            The eigenvalue handling mode.
        _symmetrized : bool
            Whether symmetrized calculation is used.
        _is_fitted : bool
            Whether the estimator has been fitted.
        _score_method : str
            The type of vamp score to use. Can be 'vamp-1' or 'vamp-2', or 'vamp-e'
        """
        self._score = None
        self._score_list = []
        self._score_method = score_method

        self._epsilon = epsilon
        self._mode = mode
        self._symmetrized = symmetrized
        self._is_fitted = False


    @property
    def loss(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        else:
            return -self._score

    @property
    def score(self):
        if not self._is_fitted:
            raise ValueError("please fit the model first")
        else:
            return self._score

    def fit(self, data):
        assert len(data) == 2

        koopman = estimate_koopman_matrix(
            data[0],
            data[1],
            epsilon=self._epsilon,
            mode=self._mode,
            symmetrized=self._symmetrized,
        )
        if self._score_method == 'vamp-1':
            self._score = torch.norm(koopman, p="nuc")
        elif self._score_method == 'vamp-2':
            self._score = torch.pow(torch.norm(koopman, p="fro"), 2) + 1
        elif self._score_method == 'vamp-e':
            koopman,score = estimate_koopman_matrix(
                data[0],
                data[1],
                epsilon=self._epsilon,
                mode=self._mode,
                symmetrized=self._symmetrized,
                vampe=True
            )
            self._score = score

        self._is_fitted = True

        return self

    def save(self):
        with torch.no_grad():
            self._score_list.append(self.score)

        return self

    def clear(self):
        self._score_list = []

        return self

    def output_mean_score(self):
        mean_score = torch.mean(torch.stack(self._score_list))

        return mean_score
