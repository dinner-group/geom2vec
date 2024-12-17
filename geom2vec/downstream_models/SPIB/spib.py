from .vae import SPIBVAE
from .base import SPIB
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import numpy as np


class SPIBModel(SPIBVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def update_model(self, inputs, input_weights, train_data_labels, test_data_labels, batch_size, threshold=0):

        # send to device
        inputs = inputs.to(self.device)
        input_weights = input_weights.to(self.device)
        train_data_labels = train_data_labels.to(self.device)
        test_data_labels = test_data_labels.to(self.device)

        mean_rep = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(self.device)

            # pass through VAE
            x = self.encoder(batch_inputs)
            z_mean, z_logvar = self.encode(x)
            mean_rep += [z_mean]

        mean_rep = torch.cat(mean_rep, dim=0)

        state_population = train_data_labels.sum(dim=0).float() / train_data_labels.shape[0]

        # ignore states whose state_population is smaller than threshold to speed up the convergence
        # By default, the threshold is set to be zero
        train_data_labels = train_data_labels[:, state_population > threshold]
        test_data_labels = test_data_labels[:, state_population > threshold]

        # save new guess of representative-inputs
        representative_inputs = []

        for i in range(train_data_labels.shape[-1]):
            weights = input_weights[train_data_labels[:, i].bool()].reshape(-1, 1)
            center_z = ((weights * mean_rep[train_data_labels[:, i].bool()]).sum(dim=0) / weights.sum()).reshape(1, -1)

            # find the one cloest to center_z as representative-inputs
            dist = torch.square(mean_rep - center_z).sum(dim=-1)
            index = torch.argmin(dist)
            representative_inputs += [inputs[index].reshape(1, -1)]

        representative_inputs = torch.cat(representative_inputs, dim=0)

        self.reset_representative(representative_inputs)

        # record the old parameters
        w = self.decoder_output[0].weight[state_population > threshold]
        b = self.decoder_output[0].bias[state_population > threshold]

        # reset the dimension of the output
        self.decoder_output = nn.Sequential(
            nn.Linear(self.neuron_num2, self.output_dim),
            nn.LogSoftmax(dim=1))

        self.decoder_output[0].weight = nn.Parameter(w.to(self.device))
        self.decoder_output[0].bias = nn.Parameter(b.to(self.device))

        return train_data_labels, test_data_labels

    @torch.no_grad()
    def update_labels(self, inputs, batch_size):
        if self.UpdateLabel:
            labels = []

            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]

                # pass through VAE
                batch_inputs = batch_inputs.to(self.device)
                x = self.encoder(batch_inputs)
                z_mean, z_logvar = self.encode(x)
                log_prediction = self.decode(z_mean)

                # label = p/Z
                labels += [log_prediction.exp()]

            labels = torch.cat(labels, dim=0)
            max_pos = labels.argmax(1)
            labels = F.one_hot(max_pos, num_classes=self.output_dim)

            return labels

    def calculate_loss(self, data_inputs, data_targets, data_weights):

        # pass through VAE
        data_inputs = data_inputs.to(self.device)
        data_targets = data_targets.to(self.device)
        data_weights = data_weights.to(self.device)

        outputs, z_sample, z_mean, z_logvar = self.forward(data_inputs)

        # KL Divergence
        log_p = self.log_p(z_sample)
        log_q = -0.5 * torch.sum(z_logvar + torch.pow(z_sample - z_mean, 2) / torch.exp(z_logvar), dim=1)

        reconstruction_error = torch.sum(data_weights * torch.sum(-data_targets * outputs, dim=1)) / data_weights.sum()
        # KL Divergence
        kl_loss = torch.sum(data_weights * (log_q - log_p)) / data_weights.sum()

        loss = reconstruction_error + self.beta * kl_loss

        return loss, reconstruction_error.detach().cpu().data, kl_loss.detach().cpu().data

    def fit(self, train_dataset, test_dataset, batch_size=128, tolerance=0.001, patience=5, refinements=15,
            mask_threshold=0, index=0):

        self.train()

        # Data preparation
        train_dataloader, test_dataloader = self._prepare_data_loaders(train_dataset, test_dataset, batch_size)

        # Initialize pseudo-inputs
        self.init_representative_inputs(train_dataset.past_data, train_dataset.future_labels)

        # Set optimizer and scheduler
        optimizer, scheduler = self._configure_optimizer_and_scheduler()

        # Initialize variables
        start_time = time.time()
        step = 0
        update_times = 0
        unchanged_epochs = 0
        epoch = 0
        train_epoch_loss_prev = 0

        # Initial state population
        state_population_prev = self._compute_state_population(train_dataset.future_labels)

        # Training loop
        while True:
            epoch += 1

            # Train for one epoch
            step, train_losses = self._train_one_epoch(train_dataloader, optimizer, step)

            # Evaluate on training and test data
            train_time = time.time() - start_time
            test_losses = self._evaluate_model(test_dataloader)

            # Print losses
            self._print_epoch_statistics(epoch, train_time, train_losses, test_losses)

            # Check convergence
            relative_change = self._check_convergence(train_dataset, state_population_prev, batch_size, mask_threshold)
            print('relative change in state population:', relative_change)

            # Update learning rate scheduler
            scheduler.step()
            if self.lr_scheduler_gamma < 1:
                print("Update lr to %f" % optimizer.param_groups[0]['lr'])

            # Check for early stopping or refinements
            unchanged_epochs = self._update_unchanged_epochs(train_losses['loss'], train_epoch_loss_prev, tolerance,
                                                             unchanged_epochs)
            train_epoch_loss_prev = train_losses['loss']

            if unchanged_epochs > patience:
                if self.UpdateLabel and update_times < refinements:
                    update_times += 1
                    print("Update %d\n" % update_times)
                    self._update_model_and_labels(train_dataset, test_dataset, batch_size, mask_threshold)
                    optimizer, scheduler = self._configure_optimizer_and_scheduler()
                    epoch = 0
                    unchanged_epochs = 0
                    state_population_prev = self._compute_state_population(train_dataset.future_labels)
                else:
                    break

        # Output final results
        total_training_time = time.time() - start_time
        print("Total training time: %f" % total_training_time)
        self.eval()

        if self.UpdateLabel:
            self._update_model_and_labels(train_dataset, test_dataset, batch_size)

        return self

    # Helper methods
    def _prepare_data_loaders(self, train_dataset, test_dataset, batch_size):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(train_dataset), batch_size, False),
            batch_size=None)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(test_dataset), batch_size, False),
            batch_size=None)
        return train_dataloader, test_dataloader

    def _configure_optimizer_and_scheduler(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_lambda = lambda epoch: self.lr_scheduler_gamma ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, scheduler

    def _compute_state_population(self, labels):
        return (torch.sum(labels, dim=0).float() / labels.shape[0]).cpu()

    def _train_one_epoch(self, dataloader, optimizer, step):
        self.train()
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_recon_error = 0
        total_weight = 0

        for batch_inputs, batch_outputs, batch_weights in dataloader:
            batch_inputs = batch_inputs.to(self.device)
            batch_outputs = batch_outputs.to(self.device)
            batch_weights = batch_weights.to(self.device)

            step += 1
            loss, recon_error, kl_loss = self.calculate_loss(batch_inputs, batch_outputs, batch_weights)

            if torch.isnan(loss).any():
                print("Loss is nan!")
                raise ValueError

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            weight = batch_weights.sum().cpu()
            total_weight += weight
            epoch_loss += loss.detach().cpu().item() * weight
            epoch_kl_loss += kl_loss * weight
            epoch_recon_error += recon_error * weight

            self.train_loss_history.append([step, loss.detach().cpu().item()])

        # Average losses
        epoch_loss /= total_weight
        epoch_kl_loss /= total_weight
        epoch_recon_error /= total_weight

        return step, {'loss': epoch_loss, 'kl_loss': epoch_kl_loss, 'recon_error': epoch_recon_error}

    def _evaluate_model(self, dataloader):
        self.eval()
        total_loss = 0
        total_kl_loss = 0
        total_recon_error = 0
        total_weight = 0

        with torch.no_grad():
            for batch_inputs, batch_outputs, batch_weights in dataloader:
                loss, recon_error, kl_loss = self.calculate_loss(batch_inputs, batch_outputs, batch_weights)
                weight = batch_weights.sum().cpu()
                total_weight += weight
                total_loss += loss.cpu().item() * weight
                total_kl_loss += kl_loss * weight
                total_recon_error += recon_error * weight

        # Average losses
        total_loss /= total_weight
        total_kl_loss /= total_weight
        total_recon_error /= total_weight

        self.test_loss_history.append([self.train_loss_history[-1][0], total_loss])

        return {'loss': total_loss, 'kl_loss': total_kl_loss, 'recon_error': total_recon_error}

    def _print_epoch_statistics(self, epoch, train_time, train_losses, test_losses):
        print(
            f"Epoch {epoch}:\tTime {train_time:.2f} s\n"
            f"Loss (train): {train_losses['loss']:.6f}\tKL loss (train): {train_losses['kl_loss']:.6f}\n"
            f"Reconstruction loss (train): {train_losses['recon_error']:.6f}"
        )
        print(
            f"Loss (test): {test_losses['loss']:.6f}\tKL loss (test): {test_losses['kl_loss']:.6f}\n"
            f"Reconstruction loss (test): {test_losses['recon_error']:.6f}"
        )

    def _check_convergence(self, train_dataset, state_population_prev, batch_size, mask_threshold):
        new_labels = self.update_labels(train_dataset.future_data, batch_size)
        state_population = self._compute_state_population(new_labels)

        print('State population:')
        print(state_population.numpy())
        self.label_history.append(state_population.numpy())

        mask = (state_population_prev > mask_threshold)
        relative_change = torch.sqrt(
            torch.square((state_population - state_population_prev)[mask] / state_population_prev[mask]).mean()
        )

        print(f'Relative state population change={relative_change:.6f}')
        self.relative_state_population_change_history.append(
            [self.train_loss_history[-1][0], relative_change.numpy()]
        )

        return relative_change

    def _update_unchanged_epochs(self, current_loss, previous_loss, tolerance, unchanged_epochs):
        if abs(current_loss - previous_loss) < tolerance:
            unchanged_epochs += 1
        else:
            unchanged_epochs = 0
        return unchanged_epochs

    def _update_model_and_labels(self, train_dataset, test_dataset, batch_size, mask_threshold=0):
        train_labels = self.update_labels(train_dataset.future_data, batch_size)
        test_labels = self.update_labels(test_dataset.future_data, batch_size)

        train_labels, test_labels = self.update_model(
            train_dataset.past_data, train_dataset.data_weights,
            train_labels, test_labels, batch_size, mask_threshold
        )

        if self.score_model is not None:
            train_score = self.score_model.score(train_labels)
            test_score = self.score_model.score(test_labels)
            self.score_history.append([len(self.convergence_history) + 1, train_score, test_score])

        train_dataset.update_labels(train_labels)
        test_dataset.update_labels(test_labels)

    @torch.no_grad()
    def transform(self, data, batch_size=128, to_numpy=False):
        r""" Transforms data through the instantaneous or time-shifted network lobe.
        Parameters
        ----------
        data : numpy ndarray or torch tensor
            The data to transform.
        batch_size : int, default=128
        to_numpy: bool, default=True
            Whether to convert torch tensor to numpy array.
        Returns
        -------
        List of numpy array or torch tensor containing transformed data.
        """
        self.eval()

        if isinstance(data, torch.Tensor):
            inputs = data
        else:
            inputs = torch.from_numpy(data.copy()).float()

        all_prediction = []
        all_z_mean = []
        all_z_logvar = []

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(self.device)

            # pass through VAE
            x = self.encoder(batch_inputs)
            z_mean, z_logvar = self.encode(x)

            log_prediction = self.decode(z_mean)

            all_prediction += [log_prediction.exp().cpu()]
            all_z_logvar += [z_logvar.cpu()]
            all_z_mean += [z_mean.cpu()]

        all_prediction = torch.cat(all_prediction, dim=0)
        all_z_logvar = torch.cat(all_z_logvar, dim=0)
        all_z_mean = torch.cat(all_z_mean, dim=0)

        labels = all_prediction.argmax(1)

        if to_numpy:
            return labels.numpy().astype(np.int32), all_prediction.numpy().astype(np.double), \
                all_z_mean.numpy().astype(np.double), all_z_logvar.numpy().astype(np.double)
        else:
            return labels, all_prediction, all_z_mean, all_z_logvar


