import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class SPIB(nn.Module):
    """Base SPIB model combining dimensionality reduction and MSM objectives.

    Parameters
    ----------
    output_dim :
        Number of discrete states in the output.
    data_shape :
        Shape of the flattened input data (e.g. ``(num_tokens, 4, hidden_dim)``).
    z_dim :
        Dimension of the bottleneck (latent space).
    lagtime :
        Time delay :math:`\\Delta t` in frames used for SPIB training.
    beta :
        Trade-off between predictive capacity and model complexity.
    learning_rate :
        Learning rate for the optimizer.
    lr_scheduler_gamma :
        Multiplicative factor for learning rate decay. ``1.0`` disables decay.
    device :
        Torch device on which to run the model.
    UpdateLabel :
        If ``True``, iteratively refine labels during training.
    neuron_num1 :
        Hidden width of the encoder network.
    neuron_num2 :
        Hidden width of the decoder network.
    score_model :
        Optional scoring model used to track SPIB refinement quality.
    encoder :
        Optional custom encoder module. If ``None``, subclasses are expected
        to construct one.
    """

    def __init__(self, output_dim, data_shape, z_dim=2, lagtime=1, beta=1e-3,
                 learning_rate=1e-3, lr_scheduler_gamma=1, device=torch.device("cpu"),
                 UpdateLabel=True, neuron_num1=64, neuron_num2=64, score_model=None, encoder=None):
        super(SPIB, self).__init__()

        self.z_dim = z_dim
        self.lagtime = lagtime
        self.beta = beta

        self.learning_rate = learning_rate
        self.lr_scheduler_gamma = lr_scheduler_gamma

        self.output_dim = output_dim
        self.neuron_num1 = neuron_num1
        self.neuron_num2 = neuron_num2
        self.data_shape = data_shape
        self.UpdateLabel = UpdateLabel

        self.eps = 1e-10
        self.device = device
        self.score_model = score_model
        self.encoder = encoder

        if score_model is not None:
            self.score_history = []

        self.relative_state_population_change_history = []
        self.train_loss_history = []
        self.test_loss_history = []
        self.convergence_history = []
        self.label_history = []
