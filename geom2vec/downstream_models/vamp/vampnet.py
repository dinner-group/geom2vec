from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataprocessing import Postprocessing_vamp
from .model import BaseVAMPNet_Model, VAMPNet_Estimator


class VAMPNet_Model(BaseVAMPNet_Model):
    """VAMPNet model for VAMP.

    Extends BaseVAMPNet_Model to implement VAMPNet algorithm.

    Methods:
        _transform_to_cv: Transforms output to collective variables using VAMP postprocessing.
    """

    def _transform_to_cv(self, output, lag_time, instantaneous):
        post = Postprocessing_vamp(lag_time=lag_time, dtype=self._dtype)
        output_cv = post.fit_transform(output, instantaneous=instantaneous)
        return output_cv if len(output_cv) > 1 else output_cv[0]


class VAMPNet:
    """The method used to train the VAMPnets.

    Parameters
    ----------
    lobe : nn.Module
        A neural network model which maps the input data to the basis functions.
    lobe_lagged : Optional[nn.Module], default = None
        Neural network model for timelagged data, in case of None the lobes are shared (structure and weights).
    optimizer : str, default = "Adam"
        The type of optimizer used for training.
    device : torch.device, default = None
        The device on which the torch modules are executed.
    learning_rate : float, default = 5e-4
        The learning rate of the optimizer.
    epsilon : float, default = 1e-6
        The strength of the regularization/truncation under which matrices are inverted.
    method : str
        The methods to be applied for training.
    mode : str, default = "regularize"
        'regularize': regularize the eigenvalues by adding epsilon.
        'trunc': truncate the eigenvalues by filtering out the eigenvalues below epsilon.
    dtype : dtype, default = np.float32
        The data type of the input data and the parameters of the model.
    """

    def __init__(
        self,
        lobe: nn.Module,
        lobe_lagged: Optional[nn.Module] = None,
        optimizer: str = "Adam",
        device: torch.device = None,
        learning_rate: float = 5e-4,
        epsilon: float = 1e-6,
        weight_decay: float = 0,
        mode: str = "regularize",
        symmetrized: bool = False,
        dtype=torch.float32, # changed from np.float32
        save_model_interval=None,
        score_method='vamp-2'
    ):
        assert score_method in ['vamp-2', 'vamp-1', 'vamp-e']

        self._lobe = lobe
        self._lobe_lagged = lobe_lagged
        self._device = device
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._mode = mode
        self._symmetrized = symmetrized
        self._dtype = dtype
        self._save_model_interval = save_model_interval
        self._score_method = score_method

        if self._dtype == np.float32:
            self._lobe = self._lobe.float()
            if self._lobe_lagged is not None:
                self._lobe_lagged = self._lobe_lagged.float()
        elif self._dtype == np.float64:
            self._lobe = self._lobe.double()
            if self._lobe_lagged is not None:
                self._lobe_lagged = self._lobe_lagged.double()

        self._step = 0
        self._save_models = []

        self.optimizer_types = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
            "RMSprop": torch.optim.RMSprop,
        }
        if optimizer == "GrokFastAdamW":
            from grokfast_pytorch import GrokFastAdamW

            self.optimizer_types["GrokFastAdamW"] = GrokFastAdamW
        elif optimizer == "AdamAtan2":
            from adam_atan2_pytorch import AdamAtan2

            self.optimizer_types["AdamAtan2"] = AdamAtan2

        if optimizer not in self.optimizer_types.keys():
            raise ValueError(
                f"Unknown optimizer type, supported types are {self.optimizer_types.keys()}"
            )
        if self._lobe_lagged is None:
            self._optimizer = self.optimizer_types[optimizer](
                self._lobe.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self._optimizer = self.optimizer_types[optimizer](
                list(self._lobe.parameters()) + list(self._lobe_lagged.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        self._training_steps = []
        self._validation_steps = []
        self._training_scores = []
        self._validation_scores = []

        self._estimator = VAMPNet_Estimator(
            epsilon=self._epsilon, mode=self._mode, symmetrized=self._symmetrized, score_method=self._score_method
        )

    @property
    def training_steps(self):
        return np.array(self._training_steps)

    @property
    def validation_steps(self):
        return np.array(self._validation_steps)

    @property
    def training_scores(self):
        return np.array(self._training_scores)

    @property
    def validation_scores(self):
        return np.array(self._validation_scores)

    def partial_fit(self, data: Sequence[torch.Tensor]):
        """Performs a partial fit on data with gradient accumulation.

        Parameters
        ----------
        data : tuple or list of length 2, containing instantaneous and timelagged data
            The data to train the lobe(s) on.

        Returns
        -------
        self : VAMPNet
        """

        batch_0, batch_1 = data[0], data[1]

        self._lobe.train()
        if self._lobe_lagged is not None:
            self._lobe_lagged.train()

        self._optimizer.zero_grad()

        x_0 = self._lobe(batch_0)
        if self._lobe_lagged is None:
            x_1 = self._lobe(batch_1)
        else:
            x_1 = self._lobe_lagged(batch_1)

        loss = self._estimator.fit([x_0, x_1]).loss
        loss.backward()

        self._optimizer.step()
        self._training_scores.append((-loss).item())
        self._training_steps.append(self._step)
        self._step += 1

        return self, loss

    def validate(self, val_data):
        """Evaluates the currently set lobe(s) on validation data and returns the value of the configured score.

        Parameters
        ----------
        val_data : tuple or list of length 2, containing instantaneous and time-lagged validation data.

        Returns
        -------
        score : torch.Tensor
            The value of the score.
        """

        val_batch_0, val_batch_1 = val_data[0], val_data[1]

        self._lobe.eval()

        with torch.no_grad():
            val_output_0 = self._lobe(val_batch_0)
            if self._lobe_lagged is not None:
                self._lobe_lagged.eval()
                val_output_1 = self._lobe_lagged(val_batch_1)
            else:
                val_output_1 = self._lobe(val_batch_1)

            score = self._estimator.fit([val_output_0, val_output_1]).score
            self._estimator.save()

        return score

    def fit(
        self,
        train_loader: DataLoader,
        n_epochs: int = 1,
        validation_loader: Optional[DataLoader] = None,
        progress: Any = tqdm,
        train_patience: int = 1000,
        valid_patience: int = 1000,
        train_valid_interval: int = 1000,
    ):
        """Performs fit on data.

        Parameters
        ----------
        train_loader
            Yield a tuple of batches representing instantaneous and time-lagged samples for training.
        n_epochs
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader
             Yield a tuple of batches representing instantaneous and time-lagged samples for validation.
        progress

        Returns
        -------
        self
        """

        self._step = 0

        best_train_score = 0
        best_valid_score = 0
        train_patience_counter = 0
        valid_patience_counter = 0

        best_lobe_state = self._lobe.state_dict()
        if self._lobe_lagged is not None:
            best_lobe_lagged_state = self._lobe_lagged.state_dict()

        for epoch in progress(
            range(n_epochs), desc="epoch", total=n_epochs, leave=False
        ):
            for batch_0, batch_1 in tqdm(train_loader):
                self._step += 1
                _, loss = self.partial_fit(
                    (batch_0.to(device=self._device), batch_1.to(device=self._device))
                )

                if loss.item() < best_train_score:
                    best_train_score = loss.item()
                    train_patience_counter = 0
                else:
                    train_patience_counter += 1
                    if train_patience_counter > train_patience:
                        print(f"Training patience reached at epoch {epoch}")
                        # break the outer loop
                        self._lobe.load_state_dict(best_lobe_state)
                        if self._lobe_lagged is not None:
                            self._lobe_lagged.load_state_dict(best_lobe_lagged_state)
                        return self

                if (
                    validation_loader is not None
                    and self._step % train_valid_interval == 0
                ):
                    with torch.no_grad():
                        for val_batch_0, val_batch_1 in validation_loader:
                            self.validate(
                                (
                                    val_batch_0.to(device=self._device),
                                    val_batch_1.to(device=self._device),
                                )
                            )

                        mean_score = self._estimator.output_mean_score()
                        self._validation_steps.append(self._step)
                        self._validation_scores.append(mean_score.item())
                        self._estimator.clear()

                        print(epoch, mean_score.item())

                        if mean_score.item() > best_valid_score:
                            best_valid_score = mean_score.item()
                            valid_patience_counter = 0
                            best_lobe_state = self._lobe.state_dict()
                            if self._lobe_lagged is not None:
                                best_lobe_lagged_state = self._lobe_lagged.state_dict()

                        else:
                            valid_patience_counter += 1
                            if valid_patience_counter > valid_patience:
                                print(f"Validation patience reached at epoch {epoch}")
                                # break the outer loop
                                # load the best model
                                self._lobe.load_state_dict(best_lobe_state)

                                if self._lobe_lagged is not None:
                                    self._lobe_lagged.load_state_dict(
                                        best_lobe_lagged_state
                                    )
                                return self

                        if self._save_model_interval is not None:
                            if (epoch + 1) % self._save_model_interval == 0:
                                m = self.fetch_model()
                                self._save_models.append((epoch, m))

        self._lobe.load_state_dict(best_lobe_state)
        if self._lobe_lagged is not None:
            self._lobe_lagged.load_state_dict(best_lobe_lagged_state)
        return self

    def transform(
        self, data, instantaneous=True, return_cv=False, lag_time=None, batch_size=200
    ):
        """Transform the data through the trained networks.

        Parameters
        ----------
        data : list or tuple or ndarray
            The data to be transformed.
        instantaneous : boolean, default = True
            Whether to use the instantaneous lobe or the time-lagged lobe for transformation.
            Note that only VAMPNet method requires two lobes

        Returns
        -------
        output : array_like
            List of numpy array containing transformed data.
        """

        model = self.fetch_model()
        return model.transform(
            data,
            instantaneous=instantaneous,
            return_cv=return_cv,
            lag_time=lag_time,
            batch_size=batch_size,
        )

    def fetch_model(self) -> VAMPNet_Model:
        """Yields the current model.

        Returns
        -------
        VAMPNet_Model :
            The VAMPNet model from VAMPNet estimator.
        """

        from copy import deepcopy

        lobe = deepcopy(self._lobe)
        lobe_lagged = deepcopy(self._lobe_lagged)
        return VAMPNet_Model(lobe, lobe_lagged, device=self._device, dtype=self._dtype)

    def fetch_lobe(self) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        from copy import deepcopy

        if self._lobe_lagged is not None:
            return deepcopy(self._lobe), deepcopy(self._lobe_lagged)
        else:
            return deepcopy(self._lobe)

    def save_model(
        self, path, name="lobe.pt", name_lagged="lobe_lagged.pt"
    ) -> Tuple[nn.Module, nn.Module]:
        import os

        torch.save(self._lobe.state_dict(), os.path.join(path, name))
        torch.save(self, os.path.join(path, "vampnet.pt"))
        if self._lobe_lagged is not None:
            torch.save(self._lobe_lagged.state_dict(), os.path.join(path, name_lagged))

        return self._lobe, self._lobe_lagged

    def _traj_sampler(self, traj: Any, batch_size: int, lag_time: int):
        """
        Samples batches from a trajectory for training or validation.

        This method generates batches of data from a given trajectory, considering
        the specified lag time. It ensures that the sampled data points are
        properly time-lagged.

        Parameters
        ----------
        traj
            The input trajectory to sample from.
        batch_size
            The number of samples in each batch.
        lag_time
            The time lag between instantaneous and time-lagged samples.

        Yields
        ------
        tuple
            A tuple containing two arrays:
            - data_t: Instantaneous data samples (shape: (batch_size, feature_dim))
            - data_t_lagged: Time-lagged data samples (shape: (batch_size, feature_dim))
        """
        traj_len = traj.shape[0]
        if traj_len < lag_time:
            raise ValueError("Trajectory length is smaller than the lag time")

        segment_size = (len(traj) - lag_time) // batch_size
        if segment_size < 1:
            raise ValueError("Batch size is too large for the given lag time")

        offset_indices = np.random.randint(segment_size)
        start_indices = offset_indices + segment_size * np.arange(batch_size)

        data_t = traj[start_indices]
        data_t_lagged = traj[start_indices + lag_time]

        yield data_t, data_t_lagged

    def step_trainer(
        self,
        train_trajectory,
        valid_trajectory=None,
        valid_loader=None,
        num_steps=1000,
        batch_size=10000,
        lag_time=1,
        progress=tqdm,
        valid_steps=100,
        train_patience=1000,
        valid_patience=1000,
        train_valid_interval=1000,
    ):
        """Performs fit on data.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Yield a tuple of batches representing instantaneous and time-lagged samples for training.
        n_epochs : int, default=1
            The number of epochs (i.e., passes through the training data) to use for training.
        validation_loader : torch.utils.data.DataLoader, optional, default=None
             Yield a tuple of batches representing instantaneous and time-lagged samples for validation.
        progress : context manager, default=tqdm

        Returns
        -------
        self : VAMPNet
        """

        self._step = 0

        best_train_score = 0
        best_valid_score = 0
        train_patience_counter = 0
        valid_patience_counter = 0

        best_lobe_state = self._lobe.state_dict()
        if self._lobe_lagged is not None:
            best_lobe_lagged_state = self._lobe_lagged.state_dict()

        for step in progress(
            range(num_steps), desc="epoch", total=num_steps, leave=False
        ):
            for batch_0, batch_1 in self._traj_sampler(
                train_trajectory, batch_size, lag_time
            ):
                self._step += 1
                _, loss = self.partial_fit(
                    (batch_0.to(device=self._device), batch_1.to(device=self._device))
                )

                if loss.item() < best_train_score:
                    best_train_score = loss.item()
                    train_patience_counter = 0
                else:
                    train_patience_counter += 1
                    if train_patience_counter > train_patience:
                        print(f"Training patience reached at step {step}")
                        # break the outer loop
                        self._lobe.load_state_dict(best_lobe_state)
                        if self._lobe_lagged is not None:
                            self._lobe_lagged.load_state_dict(best_lobe_lagged_state)
                        return self

                if (
                    valid_trajectory is not None
                    or valid_loader is not None
                    and self._step % train_valid_interval == 0
                ):
                    with torch.no_grad():
                        if valid_trajectory is not None:
                            valid_step_counter = 0
                            for val_batch_0, val_batch_1 in self._traj_sampler(
                                valid_trajectory, batch_size, lag_time
                            ):
                                self.validate(
                                    (
                                        val_batch_0.to(device=self._device),
                                        val_batch_1.to(device=self._device),
                                    )
                                )
                                valid_step_counter += 1
                                if valid_step_counter > valid_steps:
                                    break
                        elif valid_loader is not None:
                            print(
                                "using the validation loader instead of random sampling"
                            )
                            for val_batch_0, val_batch_1 in valid_loader:
                                self.validate(
                                    (
                                        val_batch_0.to(device=self._device),
                                        val_batch_1.to(device=self._device),
                                    )
                                )

                        mean_score = self._estimator.output_mean_score()
                        self._validation_scores.append(mean_score.item())
                        self._estimator.clear()

                        print(step, mean_score.item())

                        if mean_score.item() > best_valid_score:
                            best_valid_score = mean_score.item()
                            valid_patience_counter = 0
                            best_lobe_state = self._lobe.state_dict()
                            if self._lobe_lagged is not None:
                                best_lobe_lagged_state = self._lobe_lagged.state_dict()

                        else:
                            valid_patience_counter += 1
                            if valid_patience_counter > valid_patience:
                                print(f"Validation patience reached at step {step}")
                                # break the outer loop
                                # load the best model
                                self._lobe.load_state_dict(best_lobe_state)

                                if self._lobe_lagged is not None:
                                    self._lobe_lagged.load_state_dict(
                                        best_lobe_lagged_state
                                    )
                                return self

                        if self._save_model_interval is not None:
                            if (step + 1) % self._save_model_interval == 0:
                                m = self.fetch_model()
                                self._save_models.append((step, m))

        self._lobe.load_state_dict(best_lobe_state)
        if self._lobe_lagged is not None:
            self._lobe_lagged.load_state_dict(best_lobe_lagged_state)

        return self
