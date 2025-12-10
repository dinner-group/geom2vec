from .base import SPIB
import torch
import torch.nn as nn
import numpy as np


class SPIBVAE(SPIB):
    """SPIB variant with a variational autoencoder encoder/decoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('representative_inputs',
                             torch.eye(self.output_dim, np.prod(self.data_shape), device=self.device,
                                       requires_grad=False))
        self.register_buffer('idle_input',
                             torch.eye(self.output_dim, self.output_dim, device=self.device, requires_grad=False))

        self.representative_weights = nn.Sequential(
            nn.Linear(self.output_dim, 1, bias=False),
            nn.Softmax(dim=0))

        if self.encoder is None:
            self.encoder = nn.Sequential(
                nn.Linear(np.prod(self.data_shape), self.neuron_num1),
                nn.ReLU(),
                nn.Linear(self.neuron_num1, self.neuron_num1),
                nn.ReLU()
            )
        self.encoder_mean = nn.Linear(self.neuron_num1, self.z_dim)
        self.encoder_logvar = nn.Parameter(torch.tensor([0.0]))

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.neuron_num2),
            nn.ReLU(),
            nn.Linear(self.neuron_num2, self.neuron_num2),
            nn.ReLU()
        )
        self.decoder_output = nn.Sequential(
            nn.Linear(self.neuron_num2, self.output_dim),
            nn.LogSoftmax(dim=1))

    def log_p(self, z, sum_up=True):
        """Evaluate the prior log-density at latent samples.

        The prior is a mixture of Gaussians whose means/log-variances are
        given by :meth:`get_representative_z` and whose mixture weights are
        parameterised by :attr:`representative_weights`.
        """
        representative_z_mean, representative_z_logvar = self.get_representative_z()
        w = self.representative_weights(self.idle_input)
        z_expand = z.unsqueeze(1)
        representative_mean = representative_z_mean.unsqueeze(0)
        representative_logvar = representative_z_logvar.unsqueeze(0)

        # Calculate log probability for each representative
        log_prob = -0.5 * torch.sum(
            representative_logvar +
            torch.pow(z_expand - representative_mean, 2) / torch.exp(representative_logvar),
            dim=2
        )

        if sum_up:
            log_p = torch.sum(torch.log(torch.exp(log_prob) @ w + self.eps), dim=1)
        else:
            log_p = torch.log(torch.exp(log_prob) * w.T + self.eps)

        return log_p

    def get_representative_z(self):
        """Encode representative inputs to obtain the prior mixture components."""
        X = self.representative_inputs.to(self.device)
        x = self.encoder(X)
        representative_z_mean, representative_z_logvar = self.encode(x)  # C x M
        return representative_z_mean, representative_z_logvar

    def reset_representative(self, representative_inputs):

        self.output_dim = representative_inputs.shape[0]
        self.idle_input = torch.eye(self.output_dim, self.output_dim, device=self.device, requires_grad=False)

        self.representative_weights = nn.Sequential(
            nn.Linear(self.output_dim, 1, bias=False),
            nn.Softmax(dim=0))
        self.representative_weights[0].weight = nn.Parameter(torch.ones([1, self.output_dim], device=self.device))
        self.representative_inputs = representative_inputs.clone().detach()

    @torch.no_grad()
    def init_representative_inputs(self, inputs, labels):
        state_population = labels.sum(dim=0).cpu()

        # randomly pick up one sample from each initlal state as the initial guess of representative-inputs
        representative_inputs = []

        for i in range(state_population.shape[-1]):
            if state_population[i] > 0:
                index = np.random.randint(0, state_population[i])
                representative_inputs += [inputs[labels[:, i].bool()][index].reshape(1, -1)]
            else:
                # randomly select one sample as the representative input
                index = np.random.randint(0, inputs.shape[0])
                representative_inputs += [inputs[index].reshape(1, -1)]

        representative_inputs = torch.cat(representative_inputs, dim=0)

        self.reset_representative(representative_inputs.to(self.device))

        return representative_inputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu

    def encode(self, x):
        z_mean = self.encoder_mean(x)
        z_logvar = self.encoder_logvar
        return z_mean, z_logvar

    def decode(self, z):
        dec = self.decoder(z)
        outputs = self.decoder_output(dec)
        return outputs

    def forward(self, data):
        # print(data.shape)
        x = self.encoder(data.to(self.device))
        z_mean, z_logvar = self.encode(x)
        z_sample = self.reparameterize(z_mean, z_logvar)
        outputs = self.decode(z_sample)

        return outputs, z_sample, z_mean, z_logvar
