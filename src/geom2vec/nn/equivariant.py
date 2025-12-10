from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, Sequential

try:
    from torch_scatter import scatter
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    def scatter(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "torch-scatter is required for EquivariantGraphConv. "
            "Install it via `pip install torch-scatter`."
        ) from exc


class GatedEquivariantBlock(nn.Module):
    r"""Applies a gated equivariant operation to scalar features and vector
    features from the `"Enhancing Geometric Representations for Molecules with
    Equivariant Vector-Scalar Interactive Message Passing"
    <https://arxiv.org/abs/2210.16518>`_ paper.

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
        out_channels (int): The number of output channels.
        intermediate_channels (int, optional): The number of channels in the
            intermediate layer, or :obj:`None` to use the same number as
            :obj:`hidden_channels`. (default: :obj:`None`)
        scalar_activation (bool, optional): Whether to apply a scalar
            activation function to the output node features.
            (default: obj:`False`)
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: Optional[int] = None,
        scalar_activation: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = Linear(hidden_channels, out_channels, bias=False)

        self.update_net = Sequential(
            Linear(hidden_channels * 2, intermediate_channels),
            nn.SiLU(),
            Linear(intermediate_channels, out_channels * 2),
        )

        self.act = nn.SiLU() if scalar_activation else None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.zero_()
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.zero_()

    def forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Applies a gated equivariant operation to node features and vector
        features.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            v (torch.Tensor): The vector features of the nodes.
        """
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)

        return x, v


class EquivariantScalar(nn.Module):
    r"""Computes final scalar outputs based on node features and vector
    features.

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
        out_channels (int): The number of output channels.
    """

    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels,
                    out_channels,
                    scalar_activation=False,
                ),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Computes the final scalar outputs.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            v (torch.Tensor): The vector features of the nodes.

        Returns:
            out (torch.Tensor): The final scalar outputs of the nodes.
        """
        for layer in self.output_network:
            x, v = layer(x, v)

        return x + v.sum() * 0, v


class EquivariantVec(torch.nn.Module):
    r"""Computes final scalar outputs based on node features and vector
    features.

    Args:
        hidden_channels (int): The number of hidden channels in the node
            embeddings.
    """

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()

        self.output_network = torch.nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    1,
                    scalar_activation=False,
                ),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets the parameters of the module."""
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Computes the final scalar outputs.

        Args:
            x (torch.Tensor): The scalar features of the nodes.
            v (torch.Tensor): The vector features of the nodes.

        Returns:
            out (torch.Tensor): The final scalar outputs of the nodes.
        """
        for layer in self.output_network:
            x, v = layer(x, v)

        return x + v.sum() * 0, v.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


class Merger(nn.Module):
    def __init__(self, window_size: int, hidden_channels: int):
        super().__init__()
        self.window_size = window_size
        self.down_sample = nn.Sequential(
            nn.Linear(window_size * hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, hidden_channels = x.size()
        if num_tokens % self.window_size != 0:
            pad_size = self.window_size - (num_tokens % self.window_size)
            pad = torch.zeros(
                batch_size,
                pad_size,
                hidden_channels,
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
            num_tokens += pad_size
        x = x.view(batch_size, num_tokens // self.window_size, self.window_size * hidden_channels)
        return self.down_sample(x)

class EquiLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scalar_linear = nn.Linear(in_features, out_features, bias=bias)
        self.vector_linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, num_channels, hidden_channels = x.shape
        if num_channels != 4:
            raise ValueError("EquiLinear expects 4 channels (1 scalar + 3 vectors).")

        scalar = x[:, :, 0, :]
        vector = x[:, :, 1:, :].permute(0, 1, 3, 2).reshape(batch_size, num_tokens, 3 * hidden_channels)

        scalar_out = self.scalar_linear(scalar)
        vector_out = self.vector_linear(vector).view(batch_size, num_tokens, 3, self.out_features)

        return torch.cat([scalar_out.unsqueeze(2), vector_out], dim=2)

class EquivariantTokenMerger(nn.Module):
    def __init__(self, window_size: int, hidden_channels: int):
        super().__init__()
        if window_size < 2:
            raise ValueError("window_size must be at least 2 for the equivariant merger.")
        self.window_size = window_size
        self.scalar_merge = nn.Linear(window_size * hidden_channels, hidden_channels, bias=True)
        self.vector_merge = nn.Linear(window_size * hidden_channels, hidden_channels, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, num_channels, hidden_channels = x.shape
        if num_channels != 4:
            raise ValueError("EquivariantTokenMerger expects 4 channels (1 scalar + 3 vectors).")

        if num_tokens % self.window_size != 0:
            pad_tokens = self.window_size - (num_tokens % self.window_size)
            pad = torch.zeros(
                batch_size,
                pad_tokens,
                num_channels,
                hidden_channels,
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
            num_tokens += pad_tokens

        new_tokens = num_tokens // self.window_size
        x = x.view(batch_size, new_tokens, self.window_size, num_channels, hidden_channels)

        scalar = x[:, :, :, 0, :].reshape(batch_size, new_tokens, self.window_size * hidden_channels)
        scalar = self.scalar_merge(scalar)

        vector = x[:, :, :, 1:, :].permute(0, 1, 3, 2, 4)
        vector = vector.reshape(batch_size, new_tokens, 3, self.window_size * hidden_channels)
        vector = self.vector_merge(vector)

        return torch.cat([scalar.unsqueeze(2), vector], dim=2)


class EquivariantGraphConv(nn.Module):
    def __init__(self, hidden_channels: int, aggr: str = "add"):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.aggr = aggr

        self.lin_scalar_rel = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.lin_scalar_root = nn.Linear(hidden_channels, hidden_channels)

        self.lin_vector_rel = nn.Linear(3 * hidden_channels, 3 * hidden_channels, bias=False)
        self.lin_vector_root = nn.Linear(3 * hidden_channels, 3 * hidden_channels, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if x.dim() != 3 or x.size(1) != 4:
            raise ValueError("EquivariantGraphConv expects input of shape (N, 4, hidden_channels).")

        scalar = x[:, 0, :]
        vector = x[:, 1:, :].reshape(x.size(0), -1)

        row, col = edge_index
        scalar_messages = self.lin_scalar_rel(scalar[col])
        vector_messages = self.lin_vector_rel(vector[col])

        scalar_agg = scatter(scalar_messages, row, dim=0, dim_size=scalar.size(0), reduce=self.aggr)
        vector_agg = scatter(vector_messages, row, dim=0, dim_size=vector.size(0), reduce=self.aggr)

        scalar_out = self.lin_scalar_root(scalar) + scalar_agg
        vector_out = self.lin_vector_root(vector) + vector_agg
        vector_out = vector_out.view(x.size(0), 3, self.hidden_channels)

        return torch.cat([scalar_out.unsqueeze(1), vector_out], dim=1)
