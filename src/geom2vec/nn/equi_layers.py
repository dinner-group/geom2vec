import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class _VectorDropout(nn.Module):
    """Drop full vector channels together to preserve equivariance."""

    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.drop_rate = drop_rate
        self._dummy = nn.Parameter(torch.empty(0))

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_rate
        mask = torch.bernoulli(
            keep_prob * torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)
        ).unsqueeze(-1)
        return mask * x / keep_prob


class EquivariantDropout(nn.Module):
    """Apply dropout to scalar and vector channels separately."""

    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.scalar_dropout = nn.Dropout(drop_rate)
        self.vector_dropout = _VectorDropout(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        scalar = x[:, :, 0, :]
        vector = x[:, :, 1:, :]
        scalar = self.scalar_dropout(scalar)
        vector = self.vector_dropout(vector)
        return torch.cat([scalar.unsqueeze(2), vector], dim=2)


class EquivariantLayerNorm(nn.Module):
    """Layer normalisation for tensors with 1 scalar and 3 vector channels."""

    def __init__(self, num_scalar_channels: int = 1, num_vector_channels: int = 3, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_scalar = num_scalar_channels
        self.num_vector = num_vector_channels
        self.eps = eps

        self.scale = nn.Parameter(torch.ones(num_scalar_channels + num_vector_channels))
        self.shift = nn.Parameter(torch.zeros(num_scalar_channels))

    def forward(self, x: Tensor) -> Tensor:
        scalar = x[:, :, 0, :]
        vector = x[:, :, 1:, :]

        scalar_mean = scalar.mean(dim=-1, keepdim=True)
        scalar_var = scalar.var(dim=-1, keepdim=True, unbiased=False)
        scalar_norm = (scalar - scalar_mean) / torch.sqrt(scalar_var + self.eps)
        scalar_out = self.scale[0] * scalar_norm + self.shift[0]

        vec_norm_sq = (vector ** 2).sum(dim=2)
        mean_norm_sq = vec_norm_sq.mean(dim=(0, 1), keepdim=True)
        vector_norm = vector / torch.sqrt(mean_norm_sq + self.eps)
        vector_scale = self.scale[1:].view(1, 1, self.num_vector, 1)
        vector_out = vector_scale * vector_norm

        return torch.cat([scalar_out.unsqueeze(2), vector_out], dim=2)


class EquivariantGatedFeedForward(nn.Module):
    """Feed-forward block that respects scalar/vector structure."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.scalar_lin1 = nn.Linear(input_dim, 2 * hidden_dim)
        self.scalar_lin2 = nn.Linear(hidden_dim, output_dim)
        self.vector_lin1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.vector_lin2 = nn.Linear(hidden_dim, output_dim, bias=False)

        self.dropout = EquivariantDropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        scalar, vector = x[:, :, 0, :], x[:, :, 1:, :]
        gate, scalar_hidden = torch.split(self.scalar_lin1(scalar), self.hidden_dim, dim=-1)
        gate = torch.sigmoid(gate)
        vector_hidden = self.vector_lin1(vector)
        vector_hidden = gate.unsqueeze(-2) * vector_hidden
        scalar_hidden = F.silu(scalar_hidden)

        fused = torch.cat([scalar_hidden.unsqueeze(2), vector_hidden], dim=2)
        fused = self.dropout(fused)

        scalar_out = self.scalar_lin2(fused[:, :, 0, :])
        vector_out = self.vector_lin2(fused[:, :, 1:, :])

        return torch.cat([scalar_out.unsqueeze(2), vector_out], dim=2)


class EquivariantAttentionBlock(nn.Module):
    """Full equivariant attention + FFN block mirroring EMT implementation."""

    def __init__(
        self,
        attention_module: nn.Module,
        hidden_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = attention_module
        self.norm1 = EquivariantLayerNorm()
        self.norm2 = EquivariantLayerNorm()
        self.dropout = EquivariantDropout(dropout)
        self.ffn = EquivariantGatedFeedForward(hidden_channels, hidden_channels, hidden_channels, dropout=dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        attn_out, _ = self.attention(self.norm1(x), mask=mask)
        attn_out = self.dropout(attn_out)
        x = residual + attn_out

        residual = x
        ffn_out = self.ffn(self.norm2(x))
        ffn_out = self.dropout(ffn_out)
        return residual + ffn_out
