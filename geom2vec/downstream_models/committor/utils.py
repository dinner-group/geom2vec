import torch
import torch.nn as nn


class ScaledSigmoid(nn.Module):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def forward(self, x):
        return torch.sigmoid(x) * (1 + 2 * self.epsilon) - self.epsilon
