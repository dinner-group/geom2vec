from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    r"""A multi-layer perceptron (MLP) with SiLU activation functions.

    Args:
        input_channels (int): The number of input features.
        hidden_channels (int): The number of hidden features.
        out_channels (int): The number of output features.
        num_layers (int): The number of MLP layers.
        out_activation (nn.Module, optional): The activation function of the
            output layer. (default: :obj:`None`)

    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_channels, hidden_channels))
        self.layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(nn.SiLU())
        self.layers.append(nn.Linear(hidden_channels, out_channels))

        if out_activation is not None:
            self.out_activation = out_activation
        else:
            self.out_activation = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        r"""Applies the MLP to the input tensor."""
        for layer in self.layers:
            x = layer(x)
        return self.out_activation(x)
