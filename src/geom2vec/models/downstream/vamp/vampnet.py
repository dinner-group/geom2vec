from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from geom2vec.data.features import FlatFeatureSpec, unpacking_features
from .dataprocessing import Postprocessing_vamp
from .model import BaseVAMPNet_Model, VAMPNet_Estimator


def _default_ca_coords_from_features(graph_features: torch.Tensor, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Construct synthetic CA coordinates from token indices.

    This is a fallback used when no CA coordinates are supplied. It lays tokens out
    on a 1D line along the x-axis and fills the remaining dimensions with zeros.
    """
    if graph_features.ndim < 2:
        return None
    num_tokens = graph_features.shape[1]
    if num_tokens <= 0:
        return None
    frames = graph_features.shape[0]
    token_axis = torch.arange(num_tokens, dtype=dtype, device="cpu").view(1, num_tokens, 1)
    token_axis = token_axis.expand(frames, -1, -1)
    zeros = torch.zeros(frames, num_tokens, 2, dtype=dtype, device="cpu")
    return torch.cat([token_axis, zeros], dim=-1)


@dataclass
class VAMPNetConfig:
    """Configuration parameters for :class:`VAMPNet`.

    Most users will configure VAMPNet via this dataclass rather than the
    long argument list of :class:`VAMPNet.__init__`.
    """

    optimizer: str = "AdamAtan2"
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    epsilon: float = 1e-6
    mode: str = "regularize"
    symmetrized: bool = False
    score_method: str = "vamp-2"
    dtype: torch.dtype = torch.float32
    device: Optional[Union[str, torch.device]] = None
    save_model_interval: Optional[int] = None
    flat_input: bool = False
    feature_spec: Optional[FlatFeatureSpec] = None
    num_tokens: Optional[int] = None
    hidden_dim: Optional[int] = None
    train_patience: int = 1000
    valid_patience: int = 1000
    train_valid_interval: int = 1000
    remove_mean: bool = True


class VAMPNet_Model(BaseVAMPNet_Model):
    """Wrapper around :class:`BaseVAMPNet_Model` with VAMP postprocessing.

    Applies :class:`Postprocessing_vamp` in :meth:`_transform_to_cv` to turn
    network outputs into collective variables (CVs).
    """

    def _transform_to_cv(self, output, lag_time, instantaneous):
        post = Postprocessing_vamp(lag_time=lag_time, dtype=self._dtype)
        output_cv = post.fit_transform(output, instantaneous=instantaneous)
        return output_cv if len(output_cv) > 1 else output_cv[0]


class VAMPNet:
    """Trainer wrapper for VAMPNet lobes.

    Parameters
    ----------
    Parameters
    ----------
    lobe :
        Neural network mapping input frames to feature vectors (basis functions).
    lobe_lagged :
        Optional separate network for time-lagged frames. If ``None``, the same
        :paramref:`lobe` is used for both instantaneous and time-lagged data.
    config :
        Optional :class:`VAMPNetConfig`. If given, it overrides the scalar keyword
        arguments below.
    optimizer :
        Name of the optimiser to use (e.g. ``\"Adam\"``, ``\"AdamW\"``,
        ``\"GrokFastAdamW\"``, ``\"AdamAtan2\"``). Default: ``\"AdamAtan2\"``.
    device :
        Device on which to run the lobes and estimator. If ``None``, uses
        the current default PyTorch device.
    learning_rate :
        Optimiser learning rate.
    epsilon :
        Regularisation / truncation strength used when inverting covariance matrices.
    weight_decay :
        Weight decay (L2 penalty) for the optimiser.
    mode :
        Either ``\"regularize\"`` (add :paramref:`epsilon` to eigenvalues) or
        ``\"trunc\"`` (drop eigenvalues below :paramref:`epsilon`).
    symmetrized :
        Whether to use symmetrised covariance matrices in the estimator.
    dtype :
        Torch dtype for model parameters and data.
    save_model_interval :
        If not ``None``, save lobe checkpoints every N epochs via :meth:`save_model`.
    score_method :
        Scoring objective used by the estimator (``\"vamp-2\"``, ``\"vamp-1\"``
        or ``\"vamp-e\"``).
    """

    def __init__(
        self,
        lobe: nn.Module,
        lobe_lagged: Optional[nn.Module] = None,
        *,
        config: Optional[VAMPNetConfig] = None,
        optimizer: str = "AdamAtan2",
        device: Optional[Union[str, torch.device]] = None,
        learning_rate: float = 5e-4,
        epsilon: float = 1e-6,
        weight_decay: float = 0.0,
        mode: str = "regularize",
        symmetrized: bool = False,
        dtype: torch.dtype = torch.float32,
        save_model_interval: Optional[int] = None,
        score_method: str = "vamp-2",
    ) -> None:
        """Create a :class:`VAMPNet` training wrapper."""
        if config is None:
            config = VAMPNetConfig(
                optimizer=optimizer,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epsilon=epsilon,
                mode=mode,
                symmetrized=symmetrized,
                dtype=dtype,
                device=device,
                score_method=score_method,
                save_model_interval=save_model_interval,
            )
        self._config = config

        if self._config.score_method not in {"vamp-2", "vamp-1", "vamp-e"}:
            raise ValueError("score_method must be one of {'vamp-2', 'vamp-1', 'vamp-e'}")

        self._lobe = lobe
        self._lobe_lagged = lobe_lagged
        self._device = torch.device(self._config.device) if self._config.device is not None else None
        self._dtype = self._config.dtype
        self._save_model_interval = self._config.save_model_interval
        self._remove_mean = self._config.remove_mean

        if self._dtype == torch.float32:
            self._lobe = self._lobe.float()
            if self._lobe_lagged is not None:
                self._lobe_lagged = self._lobe_lagged.float()
        elif self._dtype == torch.float64:
            self._lobe = self._lobe.double()
            if self._lobe_lagged is not None:
                self._lobe_lagged = self._lobe_lagged.double()

        if self._config.flat_input:
            if self._config.feature_spec is not None:
                self._feature_spec = self._config.feature_spec
            else:
                if self._config.num_tokens is None or self._config.hidden_dim is None:
                    raise ValueError(
                        "For flat_input=True provide `feature_spec` or both `num_tokens` and `hidden_dim`."
                    )
                self._feature_spec = FlatFeatureSpec(
                    num_tokens=self._config.num_tokens,
                    hidden_dim=self._config.hidden_dim,
                )
        else:
            self._feature_spec = self._config.feature_spec

        self._optimizer = self._build_optimizer()

        self._training_steps: List[int] = []
        self._validation_steps: List[int] = []
        self._training_scores: List[float] = []
        self._validation_scores: List[float] = []
        self._step = 0
        self._checkpoint_states: List[Tuple[int, Any]] = []

        self._estimator = VAMPNet_Estimator(
            epsilon=self._config.epsilon,
            mode=self._config.mode,
            symmetrized=self._config.symmetrized,
            score_method=self._config.score_method,
            remove_mean=self._remove_mean,
        )

    def _build_optimizer(self):
        """Instantiate the requested optimizer, resolving optional dependencies on demand."""
        optimizer_name = self._config.optimizer
        optimizer_map = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
            "RMSprop": torch.optim.RMSprop,
        }

        if optimizer_name == "GrokFastAdamW":
            try:
                from grokfast_pytorch import GrokFastAdamW  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "grokfast_pytorch is required for GrokFastAdamW. Install via `pip install grokfast_pytorch`."
                ) from exc
            optimizer_map[optimizer_name] = GrokFastAdamW
        elif optimizer_name == "AdamAtan2":
            try:
                from adam_atan2_pytorch import AdamAtan2  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "adam-atan2-pytorch is required for AdamAtan2. Install via `pip install adam-atan2-pytorch`."
                ) from exc
            optimizer_map[optimizer_name] = AdamAtan2

        if optimizer_name not in optimizer_map:
            raise ValueError(f"Unknown optimizer '{optimizer_name}'. Supported: {sorted(optimizer_map)}")

        params = list(self._lobe.parameters())
        if self._lobe_lagged is not None:
            params.extend(self._lobe_lagged.parameters())

        optimizer_cls = optimizer_map[optimizer_name]
        return optimizer_cls(
            params,
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

    def _prepare_batch(self, batch):
        """Normalize a training/validation batch into a dict understood by the lobe.

        Supports three input conventions:

        - dict with ``\"graph_features\"`` (and optional ``\"ca_coords\"``, ``\"weights\"``)
        - tuple/list ``(graph_features, ca_coords)``
        - raw tensor, optionally flattened; if ``flat_input=True`` it is unpacked
          using :class:`FlatFeatureSpec`.
        """
        if isinstance(batch, dict):
            graph = batch["graph_features"].to(device=self._device, dtype=self._dtype)
            coords = batch.get("ca_coords")
            if coords is not None:
                coords = coords.to(device=self._device, dtype=self._dtype)
            weights = batch.get("weights")
            if weights is not None:
                weights = weights.to(device=self._device, dtype=self._dtype)
            prepared = {"graph_features": graph}
            if coords is not None:
                prepared["ca_coords"] = coords
            if weights is not None:
                prepared["weights"] = weights.view(-1)
            return prepared

        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            graph = batch[0].to(device=self._device, dtype=self._dtype)
            coords = batch[1]
            if coords is not None:
                coords = coords.to(device=self._device, dtype=self._dtype)
            return {"graph_features": graph, "ca_coords": coords}

        tensor = batch.to(device=self._device)
        if self._config.flat_input:
            if self._feature_spec is None:
                raise ValueError("Feature specification required for flat inputs.")
            unpacked = unpacking_features(
                tensor,
                self._feature_spec.num_tokens,
                self._feature_spec.hidden_dim,
            )
            graph = unpacked["graph_features"].to(device=self._device, dtype=self._dtype)
            coords = unpacked.get("ca_coords")
            if coords is not None:
                coords = coords.to(device=self._device, dtype=self._dtype)
            prepared = {"graph_features": graph}
            if coords is not None:
                prepared["ca_coords"] = coords
            return prepared

        tensor = tensor.to(dtype=self._dtype)
        if tensor.ndim >= 4:
            coords = _default_ca_coords_from_features(tensor.detach().cpu(), dtype=self._dtype)
            coords = coords.to(device=self._device, dtype=self._dtype) if coords is not None else None
            prepared = {"graph_features": tensor}
            if coords is not None:
                prepared["ca_coords"] = coords
            return prepared
        return tensor.to(dtype=self._dtype)
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
        """Perform a single optimisation step on a mini-batch pair.

        Parameters
        ----------
        data :
            Tuple/list ``(x_t, x_tlag)`` containing instantaneous and time-lagged
            mini-batches (each in any format accepted by :meth:`_prepare_batch`).

        Returns
        -------
        self : VAMPNet
        """

        batch_0, batch_1 = data[0], data[1]

        self._lobe.train()
        if self._lobe_lagged is not None:
            self._lobe_lagged.train()

        self._optimizer.zero_grad()

        batch_0 = self._prepare_batch(batch_0)
        batch_1 = self._prepare_batch(batch_1)

        weights = None
        if isinstance(batch_0, dict):
            weights = batch_0.pop("weights", None)
        if isinstance(batch_1, dict):
            batch_1.pop("weights", None)
        if weights is not None:
            weights = weights.to(device=self._device, dtype=self._dtype).view(-1)

        x_0 = self._lobe(batch_0)
        if self._lobe_lagged is None:
            x_1 = self._lobe(batch_1)
        else:
            x_1 = self._lobe_lagged(batch_1)

        loss = self._estimator.fit([x_0, x_1], sample_weights=weights).loss
        loss.backward()

        self._optimizer.step()
        loss_value = loss.item()
        self._training_scores.append(-loss_value)
        self._training_steps.append(self._step)
        self._step += 1

        return self, loss

    def validate(self, val_data):
        """Evaluate the current lobe(s) on a validation mini-batch pair.

        Parameters
        ----------
        val_data :
            Tuple/list ``(x_t, x_tlag)`` of validation mini-batches.

        Returns
        -------
        score :
            The scalar score returned by the underlying estimator.
        """

        val_batch_0, val_batch_1 = val_data[0], val_data[1]

        self._lobe.eval()

        with torch.no_grad():
            val_batch_0 = self._prepare_batch(val_batch_0)
            val_batch_1 = self._prepare_batch(val_batch_1)

            weights = None
            if isinstance(val_batch_0, dict):
                weights = val_batch_0.pop("weights", None)
            if isinstance(val_batch_1, dict):
                val_batch_1.pop("weights", None)
            if weights is not None:
                weights = weights.to(device=self._device, dtype=self._dtype).view(-1)

            val_output_0 = self._lobe(val_batch_0)
            if self._lobe_lagged is not None:
                self._lobe_lagged.eval()
                val_output_1 = self._lobe_lagged(val_batch_1)
            else:
                val_output_1 = self._lobe(val_batch_1)

            score = self._estimator.fit([val_output_0, val_output_1], sample_weights=weights).score
            self._estimator.save()

        return score

    def fit(
        self,
        train_loader: DataLoader,
        n_epochs: int = 1,
        validation_loader: Optional[DataLoader] = None,
        progress: Any = tqdm,
        train_patience: Optional[int] = None,
        valid_patience: Optional[int] = None,
        train_valid_interval: Optional[int] = None,
        score_callback: Optional[Callable[[float], None]] = None,
    ):
        """Train the VAMPNet lobes using mini-batch data loaders."""

        self._step = 0

        train_patience = train_patience or self._config.train_patience
        valid_patience = valid_patience or self._config.valid_patience
        train_valid_interval = train_valid_interval or self._config.train_valid_interval

        best_train_score = float("inf")
        best_valid_score = float("-inf")
        train_patience_counter = 0
        valid_patience_counter = 0

        best_lobe_state = self._lobe.state_dict()
        if self._lobe_lagged is not None:
            best_lobe_lagged_state = self._lobe_lagged.state_dict()

        for epoch in progress(
            range(n_epochs), desc="epoch", total=n_epochs, leave=False
        ):
            for batch_0, batch_1 in tqdm(train_loader, leave=False, desc="train"):
                self._step += 1
                _, loss = self.partial_fit((batch_0, batch_1))

                loss_value = loss.item()
                if loss_value < best_train_score:
                    best_train_score = loss_value
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
                            self.validate((val_batch_0, val_batch_1))

                        mean_score = self._estimator.output_mean_score()
                        mean_score_value = mean_score.item()
                        self._validation_steps.append(self._step)
                        self._validation_scores.append(mean_score_value)
                        self._estimator.clear()

                        if mean_score_value > best_valid_score:
                            best_valid_score = mean_score_value
                            if score_callback is not None:
                                score_callback(best_valid_score)
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

                        if self._save_model_interval is not None and (epoch + 1) % self._save_model_interval == 0:
                            self._checkpoint_states.append((epoch, self.fetch_model()))

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
        data :
            Trajectories or feature arrays; see :meth:`VAMPNet_Model.transform`
            for supported formats.
        instantaneous :
            Whether to use the instantaneous lobe or the time-lagged lobe for
            transformation. Note that only the VAMPNet method requires two lobes.

        Returns
        -------
        output :
            List/array of transformed features or CVs, depending on
            :paramref:`return_cv`.
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
        """Return a deepcopy of the current lobe (and optional time-lagged lobe)."""
        from copy import deepcopy

        if self._lobe_lagged is not None:
            return deepcopy(self._lobe), deepcopy(self._lobe_lagged)
        else:
            return deepcopy(self._lobe)

    def save_model(
        self,
        path: Union[str, Path],
        name: str = "lobe.pt",
        name_lagged: str = "lobe_lagged.pt",
    ) -> Tuple[nn.Module, Optional[nn.Module]]:
        """Persist the current lobe weights (and optionally the trainer) to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._lobe.state_dict(), path / name)
        torch.save(self, path / "vampnet.pt")
        if self._lobe_lagged is not None:
            torch.save(self._lobe_lagged.state_dict(), path / name_lagged)

        return self._lobe, self._lobe_lagged
