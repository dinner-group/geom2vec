from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import Dropout

from geom2vec.nn import EquivariantAttentionBlock, EquivariantSelfAttention
from geom2vec.nn.equivariant import (
    EquivariantGraphConv,
    EquivariantScalar,
    EquiLinear,
    EquivariantTokenMerger,
    PositionalEncoding,
)
from geom2vec.nn.mlps import MLP
from torch_geometric.nn import radius_graph


class ScalarMerger(nn.Module):
    """Merge scalar tokens over fixed windows followed by an MLP projection."""

    def __init__(self, window_size: int, hidden_channels: int):
        super().__init__()
        self.window_size = window_size
        self.projection = MLP(
            input_channels=hidden_channels * window_size,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=2,
            out_activation=nn.ReLU(),
        )
        self.normalize = nn.LayerNorm(hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, hidden_channels = x.shape
        if num_tokens % self.window_size != 0:
            pad_size = self.window_size - (num_tokens % self.window_size)
            x = torch.cat(
                [x, x[:, -pad_size:, :]], dim=1
            )
            num_tokens += pad_size

        x = x.reshape(batch_size, num_tokens // self.window_size, self.window_size * hidden_channels)
        # x = x.mean(dim=2)
        x = self.projection(x)
        x = self.normalize(x)
        return x

class Lobe(nn.Module):
    """Downstream head for mixing geom2vec token features.

    The Lobe consumes per-token scalar/vector embeddings together with CA coordinates,
    applies a graph-convolutional mixer (and optionally a transformer / equivariant
    token mixer), and finally projects to task-specific outputs via an MLP.

    Parameters
    ----------
    output_channels :
        Number of output channels (e.g. number of states or CV dimensions).
    input_channels :
        Dimensionality of the input scalar token features.
    hidden_channels :
        Hidden width used throughout the mixer and output MLP.
    num_tokens :
        Number of tokens per frame (e.g. residues or coarse-grained units).
    num_mlp_layers :
        Number of layers in the output MLP.
    mlp_dropout :
        Dropout probability applied before the output MLP.
    mlp_out_activation :
        Optional activation for the output of the MLP (default: identity).
    device :
        Device on which to run the module.
    equi_rep :
        If ``True``, use the equivariant mixer; otherwise a standard transformer.
    merger :
        If ``True``, downsample tokens using :class:`ScalarMerger` /
        :class:`EquivariantTokenMerger` before mixing.
    merger_window :
        Window size for token downsampling when :paramref:`merger` is enabled.
    num_mixer_layers :
        Number of transformer / equivariant attention blocks.
    expansion_factor :
        Feed-forward expansion factor inside the transformer encoder.
    nhead :
        Number of attention heads for the transformer / equivariant attention.
    pooling :
        Pooling strategy over tokens (``\"cls\"``, ``\"mean\"``, or ``\"sum\"``).
    dropout :
        Dropout probability inside the mixer.
    radius_cutoff :
        Cutoff (in Å) used when building the CA radius-graph.
    equi_backend :
        Backend for the equivariant mixer (``\"torch\"`` or ``\"triton\"``).
    """

    def __init__(
        self,
        output_channels: int = 2,
        input_channels: int = 64, 
        hidden_channels: int = 64,
        num_tokens: int = 10,
        num_mlp_layers: int = 3,
        mlp_dropout: float = 0.0,
        mlp_out_activation: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        equi_rep: bool = False,
        merger: bool = False,
        merger_window: int = 6,
        num_mixer_layers: int = 4,
        expansion_factor: int = 2,
        nhead: int = 8,
        pooling: str = "cls",
        dropout: float = 0.1,
        radius_cutoff: float = 8.0,
        equi_backend: str = "torch",
        ):
        super().__init__()

        if pooling not in {"cls", "mean", "sum"}:
            raise ValueError(f"Unsupported pooling '{pooling}'")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be a positive integer.")
        self.original_num_tokens = num_tokens
        self.num_tokens = num_tokens
        self.pooling = pooling
        self.equi_rep = equi_rep
        self.equi_backend = equi_backend.lower()
        self.radius_cutoff = radius_cutoff

        self.dropout = Dropout(mlp_dropout)
        self.input_projection = EquivariantScalar(
            input_channels, hidden_channels
        )

        self.gnn_layer = EquivariantGraphConv(hidden_channels)
        self.pos_encoding = PositionalEncoding(d_model=hidden_channels, max_len=8192)
        self.merger_window: Optional[int] = merger_window if merger else None
        self.mixer_dropout: Optional[Dropout] = None

        self.merger: Optional[Union[ScalarMerger, EquivariantTokenMerger]] = None
        if merger:
            if merger_window <= 1 or merger_window > num_tokens:
                raise ValueError("merger_window must be in the range [2, num_tokens].")
            if not equi_rep:
                self.merger = ScalarMerger(window_size=merger_window, hidden_channels=hidden_channels)
            else:
                self.merger = EquivariantTokenMerger(window_size=merger_window, hidden_channels=hidden_channels)

            self.num_tokens = (num_tokens + merger_window - 1) // merger_window

        if not equi_rep:
            from torch.nn import TransformerEncoderLayer, TransformerEncoder

            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=nhead,
                dim_feedforward=int(expansion_factor * hidden_channels),
                dropout=dropout,
                batch_first=True,
            )
            self.mixer = TransformerEncoder(encoder_layer, num_layers=num_mixer_layers)
        else:
            if hidden_channels % nhead != 0:
                raise ValueError("hidden_channels must be divisible by nhead for equivariant mixer.")
            if self.equi_backend not in {"torch", "triton"}:
                raise ValueError("equi_backend must be either 'torch' or 'triton'")

            blocks = []
            for _ in range(num_mixer_layers):
                if self.equi_backend == "torch":
                    attention_module = EquivariantSelfAttention(hidden_channels, nhead)
                else:
                    try:
                        from geom2vec.nn.triton.eqsdpa.attention import EquivariantSelfAttentionTriton
                    except ImportError as exc:
                        raise ImportError(
                            "EquivariantSelfAttentionTriton requires Triton to be installed."
                        ) from exc
                    attention_module = EquivariantSelfAttentionTriton(hidden_channels, nhead)

                blocks.append(
                    EquivariantAttentionBlock(
                        attention_module,
                        hidden_channels=hidden_channels,
                        dropout=dropout,
                    )
                )

            self.mixer = nn.ModuleList(blocks)
            self.mixer_dropout = None

        self.output_projection = MLP(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=output_channels,
            num_layers=num_mlp_layers,
            out_activation=mlp_out_activation,
        )

        self.to(device)
        self._last_ca_coords: Optional[torch.Tensor] = None
        if self.equi_rep:
            self.input_projection = EquiLinear(input_channels, hidden_channels)

    @staticmethod
    def _split_inputs(
        data: Union[
            torch.Tensor,
            Tuple[torch.Tensor, Optional[torch.Tensor]],
            list,
            dict,
        ],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return the graph features tensor and optional CA coordinates."""

        if isinstance(data, dict):
            if "graph_features" not in data:
                raise ValueError("Input dict must contain 'graph_features'.")
            graph_features = data["graph_features"]
            ca_coords = data.get("ca_coords")
        elif isinstance(data, (tuple, list)):
            if len(data) != 2:
                raise ValueError(
                    "Input tuple/list must have two elements: (graph_features, ca_coords)."
                )
            graph_features, ca_coords = data
        else:
            graph_features = data
            ca_coords = None

        if not isinstance(graph_features, torch.Tensor):
            raise TypeError("graph_features must be a torch.Tensor.")
        if ca_coords is not None and not isinstance(ca_coords, torch.Tensor):
            raise TypeError("ca_coords must be a torch.Tensor when provided.")

        token_count: Optional[int] = None
        if graph_features.dim() >= 2:
            token_count = graph_features.shape[1]

        if ca_coords is not None and ca_coords.dim() == 2:
            ca_coords = ca_coords.unsqueeze(1)
        if ca_coords is not None:
            if ca_coords.dim() != 3 or (token_count is not None and ca_coords.shape[1] != token_count) or ca_coords.shape[-1] != 3:
                raise ValueError(
                    "ca_coords must have shape (batch, num_tokens, 3)."
                )
            ca_coords = ca_coords.to(device=graph_features.device, dtype=graph_features.dtype)

        return graph_features, ca_coords

    def forward(
        self,
        data: Union[
            torch.Tensor,
            Tuple[torch.Tensor, Optional[torch.Tensor]],
            dict,
            list,
        ],
    ) -> torch.Tensor:
        """Forward pass through the token mixer head.

        Args:
            data: Either the graph feature tensor with shape ``(batch, num_tokens, 4, hidden_channels)``
                or a tuple/dict containing the graph features alongside optional CA coordinates
                with shape ``(batch, num_tokens, 3)``.
        """
        graph_features, ca_coords = self._split_inputs(data)
        self._last_ca_coords = ca_coords

        if graph_features.dim() != 4:
            raise ValueError(
                "Expected input of shape (batch, num_tokens, 4, hidden_channels)."
            )

        batch_size, num_tokens, _, _ = graph_features.shape
        batch_index = torch.arange(batch_size, device=graph_features.device).repeat_interleave(num_tokens)

        if ca_coords is None:
            raise ValueError("ca_coords must be provided for the Lobe model.")

        ca_coords_flat = ca_coords.reshape(batch_size * num_tokens, 3)
        edge_index = radius_graph(ca_coords_flat, r=self.radius_cutoff, batch=batch_index, loop=False)

        x = graph_features[:, :, 0, :].reshape(batch_size * num_tokens, -1)
        v = graph_features[:, :, 1:, :].reshape(batch_size * num_tokens, 3, -1)

        if not self.equi_rep:
            x, v = self.input_projection.pre_reduce(x=x, v=v)

        tokens = torch.cat([x.unsqueeze(1), v], dim=1)
        tokens = self.gnn_layer(tokens, edge_index)
        tokens = tokens.reshape(batch_size, num_tokens, 4, -1)

        scalar_tokens = tokens[:, :, 0, :]

        if not self.equi_rep:
            if self.merger is not None:
                scalar_tokens = self.merger(scalar_tokens)
            if self.pos_encoding is not None:
                scalar_tokens = self.pos_encoding(scalar_tokens)
            mixed = self.mixer(scalar_tokens)
            x = self._pool(mixed)
        else:
            if self.merger is not None:
                tokens = self.merger(tokens)
            for block in self.mixer:
                tokens = block(tokens)
            scalar_tokens = tokens[:, :, 0, :]
            x = self._pool(scalar_tokens)

        x = self.dropout(x)
        return self.output_projection(x)

    def _pool(self, features: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return features[:, 0, :]
        if self.pooling == "mean":
            return features.mean(dim=1)
        return features.sum(dim=1)

    @property
    def last_ca_coords(self) -> Optional[torch.Tensor]:
        """Return the CA coordinates from the most recent forward pass, if supplied."""

        return self._last_ca_coords

    @staticmethod
    def _random_rotation(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate a random 3x3 rotation matrix."""
        matrix = torch.randn(3, 3, device=device, dtype=dtype)
        q, _ = torch.linalg.qr(matrix)
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return q

    def verify_rotational_equivariance(
        self,
        data: Union[
            torch.Tensor,
            Tuple[torch.Tensor, Optional[torch.Tensor]],
            dict,
            list,
        ],
        trials: int = 3,
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> Tuple[bool, Optional[torch.Tensor]]:
        """Check rotational equivariance by applying random rotations.

        Returns:
            (is_equivariant, max_deviation)
        """
        graph_features, ca_coords = self._split_inputs(data)
        if ca_coords is None:
            raise ValueError("Rotational equivariance requires CA coordinates.")

        graph_features = graph_features.detach()
        ca_coords = ca_coords.detach()

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                baseline = self.forward((graph_features, ca_coords))
                max_dev = torch.tensor(0.0, device=baseline.device, dtype=baseline.dtype)

                for _ in range(trials):
                    rotation = self._random_rotation(graph_features.device, graph_features.dtype)

                    vectors = graph_features[:, :, 1:, :]
                    rotated_vectors = torch.einsum("ij,bnjh->bnih", rotation, vectors)
                    rotated_scalars = graph_features[:, :, 0, :].unsqueeze(2)
                    rotated_features = torch.cat([rotated_scalars, rotated_vectors], dim=2)

                    rotated_ca = torch.einsum("ij,bnj->bni", rotation, ca_coords)

                    rotated_output = self.forward((rotated_features, rotated_ca))
                    deviation = torch.abs(baseline - rotated_output).max()
                    max_dev = torch.maximum(max_dev, deviation)

                    if not torch.allclose(baseline, rotated_output, atol=atol, rtol=rtol):
                        return False, max_dev.cpu()

                return True, max_dev.cpu()
        finally:
            self.train(was_training)
