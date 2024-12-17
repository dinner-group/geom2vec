from copy import deepcopy
from typing import Literal, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class VCN(nn.Module):
    r"""
    Variational committor network for estimating the committor function.

    Parameters
    ----------
    lobe
        Neural network.
    score
        Score function to use. Can be 'vcn' (variational committor network) or 'svcn' (stopped variational committor network). Defaults to 'vcn'.
    optimizer
        Name of optimizer.
    device
        Device to use.
    learning_rate
        Optimizer learning rate.
    weight_decay
        Optimizer weight decay.
    lag_time
        Dataset lag time.
    save_model_interval
        Interval at which to save the model.

    """

    def __init__(
        self,
        lobe: nn.Module,
        *,
        score: Literal["vcn", "svcn"] = "vcn",
        optimizer: str = "adam",
        device: str = "cuda",
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
        lag_time: float = 1.0,
        save_model_interval: Optional[int] = None,
    ):
        super().__init__()

        self._lobe = lobe
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._device = torch.device(device)
        self._score = score
        self._lag_time = lag_time

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

        self._optimizer = self.optimizer_types[optimizer](
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self._step = 0
        self._save_model_interval = save_model_interval
        self._save_models = []
        self._training_steps = []
        self._validation_steps = []
        self._training_scores = []
        self._validation_scores = []
        self._training_bc_losses = []
        self._validation_bc_losses = []

        # early stopping
        self._best_train_score = torch.inf
        self._best_valid_score = torch.inf
        self._train_patience_counter = 0
        self._valid_patience_counter = 0

        self._best_lobe_state = deepcopy(self._lobe.state_dict())

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

    @property
    def training_bc_losses(self):
        return np.array(self._training_bc_losses)

    @property
    def validation_bc_losses(self):
        return np.array(self._validation_bc_losses)

    def _loss_fns(self, batch):
        """Compute the variational and boundary loss functions."""
        model = self._lobe
        device = self._device

        batch = [tensor.to(device) for tensor in batch]
        x0, x1, *score_data = batch

        x = torch.cat([x0, x1])
        u = model(x)
        u0, u1 = torch.unflatten(u, 0, (2, -1))

        score_types = {"vcn": self._vcn_score, "svcn": self._svcn_score}
        return score_types[self._score](u0, u1, *score_data)

    def _vcn_score(self, u0, u1, a0, a1, b0, b1):
        """Variational committor network loss function."""
        q0 = torch.where(a0, 0, torch.where(b0, 1, u0))
        q1 = torch.where(a1, 0, torch.where(b1, 1, u1))
        score = (q0 - q1) ** 2 / (2 * self._lag_time)
        bc_loss = (
            a0 * (u0 - 0) ** 2
            + b0 * (u0 - 1) ** 2
            + a1 * (u1 - 0) ** 2
            + b1 * (u1 - 1) ** 2
        ) / (2 * self._lag_time)
        return score, bc_loss

    def _svcn_score(self, u0, u1, a0, a1, b0, b1, dd, da, db, ad, bd, ab_ba):
        """Stopped variational committor network loss function."""
        q0 = torch.where(a0, 0, torch.where(b0, 1, u0))
        q1 = torch.where(a1, 0, torch.where(b1, 1, u1))
        score = (
            dd * (q1 - q0) ** 2
            + da * (q0 - 0) ** 2
            + db * (q0 - 1) ** 2
            + ad * (q1 - 0) ** 2
            + bd * (q1 - 1) ** 2
            + ab_ba
        ) / (2 * self._lag_time)
        bc_loss = (
            a0 * (u0 - 0) ** 2
            + b0 * (u0 - 1) ** 2
            + a1 * (u1 - 0) ** 2
            + b1 * (u1 - 1) ** 2
        ) / (2 * self._lag_time)
        return score, bc_loss

    def fit(
        self,
        train_loader: DataLoader,
        *,
        n_epochs: int = 1,
        validation_loader: Optional[DataLoader] = None,
        progress=tqdm,
        train_patience: int = 1000,
        valid_patience: int = 1000,
        train_valid_interval: int = 1000,
    ):
        r"""
        Fit the committor network to the training data.

        Parameters
        ----------
        train_loader
            Training data loader.
        n_epochs
            Number of epochs (passes through the data set).
        validation_loader
            Validation data loader.
        progress
            `tqdm` or similar object (progress bar).
        train_patience
            Number of steps to wait for training loss to improve
        valid_patience
            Number of steps to wait for validation loss to improve

        Returns
        -------
        self

        """
        self.partial_fit(
            train_loader,
            n_epochs=n_epochs,
            validation_loader=validation_loader,
            progress=progress,
            train_patience=train_patience,
            valid_patience=valid_patience,
            train_valid_interval=train_valid_interval,
        )
        self._lobe.load_state_dict(self._best_lobe_state)
        return self

    def partial_fit(
        self,
        train_loader: DataLoader,
        *,
        n_epochs: int = 1,
        validation_loader: Optional[DataLoader] = None,
        progress=tqdm,
        train_patience: int = 1000,
        valid_patience: int = 1000,
        train_valid_interval: int = 1000,
    ):
        r"""
        Fit the committor network to the training data.

        Unlike `fit`, `partial_fit` does not load the checkpoint with the best validation score.

        Parameters
        ----------
        train_loader
            Training data loader.
        n_epochs
            Number of epochs (passes through the data set).
        validation_loader
            Validation data loader.
        progress
            `tqdm` or similar object (progress bar).
        train_patience
            Number of steps to wait for training loss to improve
        valid_patience
            Number of steps to wait for validation loss to improve

        Returns
        -------
        self

        """
        for epoch in progress(
            range(n_epochs), desc="epoch", total=n_epochs, leave=False
        ):
            for batch in progress(
                train_loader, desc="batch", total=len(train_loader), leave=False
            ):
                self._step += 1

                score, bc_loss = self.training_step(batch)
                loss = score + bc_loss

                self._training_steps.append(self._step)
                self._training_scores.append(score)
                self._training_bc_losses.append(bc_loss)

                if self._save_model_interval is not None:
                    if self._step % self._save_model_interval == 0:
                        # save the model with the epoch, the step, and the metrics
                        self._save_models.append(
                            (epoch, self._step, score, bc_loss, self._lobe.state_dict())
                        )

                # early stopping on training loss
                self._train_patience_counter += 1
                if loss < self._best_train_score:
                    self._best_train_score = loss
                    self._train_patience_counter = 0
                if self._train_patience_counter >= train_patience:
                    print(f"Training patience reached at epoch {epoch}")
                    return self

                if (
                    validation_loader is not None
                    and self._step % train_valid_interval == 0
                ):
                    score, bc_loss = self.validate(validation_loader, progress=progress)

                    self._validation_steps.append(self._step)
                    self._validation_scores.append(score)
                    self._validation_bc_losses.append(bc_loss)

                    # early stopping on validation score
                    self._valid_patience_counter += 1
                    if score < self._best_valid_score:
                        self._best_valid_score = score
                        self._valid_patience_counter = 0
                        self._best_lobe_state = deepcopy(self._lobe.state_dict())
                    if self._valid_patience_counter >= valid_patience:
                        print(f"Validation patience reached at epoch {epoch}")
                        return self

        return self

    def training_step(self, *batches):
        """
        Perform a single optimization step.

        Parameters
        ----------
        *batches
            Minibatches of training data. Gradient accumulation is performed if more than one batch is provided.

        """
        self._lobe.train()
        self._optimizer.zero_grad()

        score = 0.0
        bc_loss = 0.0

        for batch in batches:
            score_batch, bc_loss_batch = self._loss_fns(batch)
            score_batch = torch.mean(score_batch) / len(batches)
            bc_loss_batch = torch.mean(bc_loss_batch) / len(batches)

            loss = score_batch + bc_loss_batch
            loss.backward()

            score += score_batch.item()
            bc_loss += bc_loss_batch.item()

        self._optimizer.step()

        return score, bc_loss

    def validate(self, validation_loader: DataLoader, progress=tqdm):
        """
        Evaluate the variational loss on validation data.

        Parameters
        ----------
        validation_loader
            Validation data loader.
        progress
            `tqdm` or similar object (progress bar).

        Returns
        -------
        score
            Variational loss.
        bc_loss
            Loss due to violating boundary conditions.

        """
        self._lobe.eval()
        with torch.no_grad():
            score = 0.0
            bc_loss = 0.0
            n_samples = 0
            for batch in progress(
                validation_loader,
                desc="validation",
                total=len(validation_loader),
                leave=False,
            ):
                score_batch, bc_loss_batch = self._loss_fns(batch)
                score += torch.sum(score_batch).item()
                bc_loss += torch.sum(bc_loss_batch).item()
                n_samples += len(score_batch)
            score /= n_samples
            bc_loss /= n_samples
        return score, bc_loss

    def forward(self, x, a, b):
        model = self._lobe
        device = self._device
        x = x.to(device)
        a = a.to(device)
        b = b.to(device)
        u = model(x)
        q = torch.where(a, 0, torch.where(b, 1, u))
        return q

    def transform(self, dataset, batch_size: int) -> np.ndarray:
        """
        Predict the committor on the data.

        Parameters
        ----------
        dataset
            Prediction dataset.
        batch_size
            Maximum batch size for the neural network.

        Returns
        -------
        Predicted committor.

        """
        self.eval()
        out_list = []
        with torch.no_grad():
            for x, a, b in tqdm(DataLoader(dataset, batch_size=batch_size)):
                out_list.append(self(x, a, b).cpu())
        return torch.cat(out_list).numpy()

    def brier_score(self, test_loader: DataLoader) -> float:
        """
        Evaluate the Brier score on test data.

        Parameters
        ----------
        test_loader
            Test data loader.

        Returns
        -------
        Brier score.

        """
        self.eval()
        with torch.no_grad():
            total = 0.0
            n_samples = 0
            for x, a, b, target in tqdm(test_loader):
                q = self(x, a, b)
                total += torch.sum((q - target.to(self._device)) ** 2).item()
                n_samples += len(q)
            return total / n_samples

    def fetch_model(self) -> nn.Module:
        """Return a copy of the neural network (lobe)."""
        return deepcopy(self._lobe)

    def save_model(self, path: str, name: str = "lobe.pt") -> nn.Module:
        """Save the model."""
        import os

        torch.save(self._lobe.state_dict(), os.path.join(path, name))
        torch.save(self, os.path.join(path, "vcn.pt"))

        return self._lobe
