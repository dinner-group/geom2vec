import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class SPIB(nn.Module):
    """
    A SPIB model which can be fit to data optimizing for dimension reduction and MSM.
    Parameters
    ----------
    output_dim : int
        Number of initial states.
    data_shape: int...
        A sequence of integers defining the shape of the input data.
    encoder_type: str, default='Nonlinear'
        Encoder type (Linear or Nonlinear)
    z_dim: int, default=2
        Dimension of bottleneck
    lagime : int, default=1
        Time delay delta t in terms of # of minimal time resolution of the trajectory data
    beta: float, default=1e-3
        Hyper-parameter beta makes a trade-off between the predictive capacity and model complexity.
    learning_rate : float, default=1e-3
        The learning rate of the Adam optimizer.
    lr_scheduler_gamma: float, default=1.0
        Multiplicative factor of learning rate decay. lr_scheduler_gamma=1 means no learning rate decay.
    device : torch device, default=torch.device("cpu")
        The device on which the torch modules are executed.
    path : str, default='./SPIB'
        Path to save the training files.
    UpdateLabel : bool, default=True
        Whether to refine the labels during the training process.
    neuron_num1 : int, default=64
        Number of nodes in each hidden layer of the encoder.
    neuron_num2 : int, default=64
        Number of nodes in each hidden layer of the decoder.
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

