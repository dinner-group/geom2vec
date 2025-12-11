from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, Sequential

try:
    from torch_scatter import scatter
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def scatter(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "torch-scatter is required for EquivariantGraphConv. "
            "Install it via `pip install torch-scatter`."
        )


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
        # Apply the same linear transformation to each vector component to preserve equivariance
        self.vector_linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, num_channels, hidden_channels = x.shape
        if num_channels != 4:
            raise ValueError("EquiLinear expects 4 channels (1 scalar + 3 vectors).")

        scalar = x[:, :, 0, :]  # (batch, tokens, hidden)
        vector = x[:, :, 1:, :]  # (batch, tokens, 3, hidden)

        scalar_out = self.scalar_linear(scalar)  # (batch, tokens, out_features)

        # Apply the same linear transformation to each vector component independently
        # to preserve equivariance: reshape to (batch*tokens*3, hidden), apply linear,
        # then reshape back to (batch, tokens, 3, out_features)
        vector_flat = vector.reshape(batch_size * num_tokens * 3, hidden_channels)
        vector_out_flat = self.vector_linear(vector_flat)
        vector_out = vector_out_flat.view(batch_size, num_tokens, 3, self.out_features)

        return torch.cat([scalar_out.unsqueeze(2), vector_out], dim=2)

class EquivariantTokenMerger(nn.Module):
    def __init__(self, window_size: int, hidden_channels: int):
        super().__init__()
        self.window_size = window_size
        self.scalar_merge = nn.Linear(window_size * hidden_channels, hidden_channels, bias=True)
        # Apply the same linear transformation to each vector component to preserve equivariance
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

        # For vectors: (batch, new_tokens, window_size, 3, hidden)
        vector = x[:, :, :, 1:, :]  # (batch, new_tokens, window_size, 3, hidden)
        # Apply the same transformation to each vector component independently
        # Reshape to (batch*new_tokens*3, window_size*hidden), apply linear, reshape back
        vector_flat = vector.permute(0, 1, 3, 2, 4).reshape(
            batch_size * new_tokens * 3, self.window_size * hidden_channels
        )
        vector_out_flat = self.vector_merge(vector_flat)
        vector = vector_out_flat.view(batch_size, new_tokens, 3, hidden_channels)

        return torch.cat([scalar.unsqueeze(2), vector], dim=2)


class EquivariantGraphConvCheap(nn.Module):
    def __init__(self, hidden_channels: int, aggr: str = "add"):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.aggr = aggr

        self.lin_scalar_rel = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.lin_scalar_root = nn.Linear(hidden_channels, hidden_channels)

        # Apply the same linear transformation to each vector component to preserve equivariance
        self.lin_vector_rel = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.lin_vector_root = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if x.dim() != 3 or x.size(1) != 4:
            raise ValueError("EquivariantGraphConv expects input of shape (N, 4, hidden_channels).")

        scalar = x[:, 0, :]  # (N, hidden)
        vector = x[:, 1:, :]  # (N, 3, hidden)

        row, col = edge_index
        scalar_messages = self.lin_scalar_rel(scalar[col])

        # Apply the same linear transformation to each vector component independently
        # Reshape to (N*3, hidden), apply linear, reshape back to (N, 3, hidden)
        num_nodes = vector.size(0)
        vector_flat = vector[col].reshape(col.size(0) * 3, self.hidden_channels)
        vector_messages_flat = self.lin_vector_rel(vector_flat)
        vector_messages = vector_messages_flat.view(col.size(0), 3, self.hidden_channels)

        scalar_agg = scatter(scalar_messages, row, dim=0, dim_size=scalar.size(0), reduce=self.aggr)
        vector_messages_for_scatter = vector_messages.reshape(col.size(0), 3 * self.hidden_channels)
        vector_agg_flat = scatter(vector_messages_for_scatter, row, dim=0, dim_size=num_nodes, reduce=self.aggr)
        vector_agg = vector_agg_flat.view(num_nodes, 3, self.hidden_channels)

        scalar_out = self.lin_scalar_root(scalar) + scalar_agg

        # Apply root transformation to each vector component independently
        vector_flat_root = vector.reshape(num_nodes * 3, self.hidden_channels)
        vector_root_flat = self.lin_vector_root(vector_flat_root)
        vector_root = vector_root_flat.view(num_nodes, 3, self.hidden_channels)
        vector_out = vector_root + vector_agg

        return torch.cat([scalar_out.unsqueeze(1), vector_out], dim=1)

class EquivariantGraphConv(nn.Module):
    """
    Equivariant message-passing conv with scalar+vector features and
    geometry-aware gating on actual graph connectivity.

    Input:
        x: (N, 4, H)  where:
           x[:, 0, :]   scalar features
           x[:, 1:, :]  vector features (3, H)
        edge_index: (2, E)  with row, col indices (messages from col -> row)

    Geometry:
        - Vector channels (H) are linearly compressed to a small
          'geom_channels' subspace (G).
        - All 3D geometry (angles, dihedrals, chirality) is computed
          in this subspace (much cheaper than full H).
        - Per-edge scalar gates are produced in geom space and expanded
          back to H for scalar and vector messages.

    Args:
        hidden_channels:  H, scalar/vector channel dimension.
        aggr:             'add' / 'mean' / 'max' for message aggregation.
        eps:              numerical epsilon.
        chiral:           if True, use signed dihedral (pseudoscalar),
                          breaking reflection equivariance (SO(3) only).
        geom_channels:    G, size of geometry subspace (e.g. 8 or 16). If None,
                          defaults to 1 (full pooling).
        mlp_hidden:       hidden size for edge MLPs.
    """

    def __init__(
        self,
        hidden_channels: int,
        aggr: str = "add",
        eps: float = 1e-8,
        chiral: bool = False,
        geom_channels: int = 16,
        mlp_hidden: int = 16,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.aggr = aggr
        self.eps = eps
        self.chiral = chiral
        self.geom_channels = geom_channels or 1  # G

        H = hidden_channels
        G = self.geom_channels

        self.lin_scalar_rel = nn.Linear(H, H, bias=False)
        self.lin_scalar_root = nn.Linear(H, H)

        self.lin_vector_rel = nn.Linear(H, H, bias=False)
        self.lin_vector_root = nn.Linear(H, H, bias=False)

        self.vec_to_geom = nn.Linear(H, G, bias=False)

        in_dim = 3 if chiral else 2
        mlp_hidden = max(mlp_hidden, 4)

        self.edge_mlp_s = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.edge_mlp_v = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )

        self.expand_s = nn.Linear(G, H)
        self.expand_v = nn.Linear(G, H)


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if x.dim() != 3 or x.size(1) != 4:
            raise ValueError("EquivariantGraphConv expects input of shape (N, 4, H).")

        N, _, H = x.shape
        if H != self.hidden_channels:
            raise ValueError(
                f"hidden_channels mismatch: got {H}, expected {self.hidden_channels}"
            )

        scalar = x[:, 0, :]     # (N, H)
        vector = x[:, 1:, :]    # (N, 3, H)

        if edge_index.numel() == 0:
            # No edges: just apply root transforms
            scalar_out = self.lin_scalar_root(scalar)
            vec_flat = vector.reshape(N * 3, H)
            vec_root_flat = self.lin_vector_root(vec_flat)
            vector_out = vec_root_flat.view(N, 3, H)
            return torch.cat([scalar_out.unsqueeze(1), vector_out], dim=1)

        row, col = edge_index    # messages go col -> row
        E = col.size(0)
        G = self.geom_channels
        eps = self.eps

        vec_flat = vector.reshape(N * 3, H)         # (N*3, H)
        vec_geom_flat = self.vec_to_geom(vec_flat)  # (N*3, G)
        vec_geom = vec_geom_flat.view(N, 3, G)      # (N, 3, G)

        v_src = vec_geom[col]                       # (E, 3, G)
        v_dst = vec_geom[row]                       # (E, 3, G)
        b_ij = v_dst - v_src                        # (E, 3, G)

        norm_b = b_ij.norm(dim=1, keepdim=True).clamp_min(eps)  # (E,1,G)
        e_ij = b_ij / norm_b                                  # (E,3,G) unit-ish

        e_flat = e_ij.reshape(E, 3 * G)                       # (E,3G)
        u_node_flat = scatter(e_flat, col, dim=0, dim_size=N, reduce="add")
        u_node = u_node_flat.view(N, 3, G)                    # (N,3,G)

        u_i = u_node[col]                                     # (E,3,G)
        u_j = u_node[row]                                     # (E,3,G)

        dot_ui_e = (u_i * e_ij).sum(dim=1)                    # (E,G)
        norm_ui = u_i.norm(dim=1).clamp_min(eps)              # (E,G)
        ang_ij = (dot_ui_e / norm_ui).clamp(-1.0, 1.0)        # (E,G)

        proj_i = dot_ui_e.unsqueeze(1) * e_ij                 # (E,3,G)
        dot_uj_e = (u_j * e_ij).sum(dim=1)                    # (E,G)
        proj_j = dot_uj_e.unsqueeze(1) * e_ij                 # (E,3,G)

        u_i_perp = u_i - proj_i                               # (E,3,G)
        u_j_perp = u_j - proj_j                               # (E,3,G)

        dot_perp = (u_i_perp * u_j_perp).sum(dim=1)           # (E,G)
        norm_perp_i = u_i_perp.norm(dim=1).clamp_min(eps)     # (E,G)
        norm_perp_j = u_j_perp.norm(dim=1).clamp_min(eps)     # (E,G)

        denom_cos = (norm_perp_i * norm_perp_j).clamp_min(eps)
        dih_cos = (dot_perp / denom_cos).clamp(-1.0, 1.0)     # (E,G)

        if self.chiral:
            cross_ij = torch.cross(u_i_perp, u_j_perp, dim=1) # (E,3,G)
            triple = (cross_ij * e_ij).sum(dim=1)             # (E,G)

            norm_e = e_ij.norm(dim=1).clamp_min(eps)          # (E,G)
            denom_sin = (norm_perp_i * norm_perp_j * norm_e).clamp_min(eps)
            dih_sin = (triple / denom_sin).clamp(-1.0, 1.0)   # (E,G)

            feats = torch.stack([ang_ij, dih_cos, dih_sin], dim=-1)  # (E,G,3)
        else:
            feats = torch.stack([ang_ij, dih_cos], dim=-1)           # (E,G,2)

        E_, G_, D = feats.shape
        assert E_ == E and G_ == G
        feats_flat = feats.view(E * G, D)                     # (E*G,D)

        g_s_geom = self.edge_mlp_s(feats_flat).view(E, G)     # (E,G)
        g_v_geom = self.edge_mlp_v(feats_flat).view(E, G)     # (E,G)

        g_s_full_logits = self.expand_s(g_s_geom)             # (E,H)
        g_v_full_logits = self.expand_v(g_v_geom)             # (E,H)

        g_s = torch.sigmoid(g_s_full_logits)                  # (E,H)
        g_v = torch.sigmoid(g_v_full_logits)                  # (E,H)

        scalar_rel = self.lin_scalar_rel(scalar[col])         # (E,H)
        scalar_messages = g_s * scalar_rel                    # (E,H)

        scalar_agg = scatter(
            scalar_messages, row, dim=0, dim_size=N, reduce=self.aggr
        )                                                     # (N,H)

        vector_flat = vector[col].reshape(E * 3, H)           # (E*3,H)
        vector_rel_flat = self.lin_vector_rel(vector_flat)    # (E*3,H)
        vector_rel = vector_rel_flat.view(E, 3, H)            # (E,3,H)

        vector_messages = g_v.unsqueeze(1) * vector_rel       # (E,3,H)

        vector_messages_flat = vector_messages.reshape(E, 3 * H)
        vector_agg_flat = scatter(
            vector_messages_flat, row, dim=0, dim_size=N, reduce=self.aggr
        )                                                     # (N,3H)
        vector_agg = vector_agg_flat.view(N, 3, H)            # (N,3,H)

        scalar_out = self.lin_scalar_root(scalar) + scalar_agg   # (N,H)

        vec_flat_root = vector.reshape(N * 3, H)
        vec_root_flat = self.lin_vector_root(vec_flat_root)
        vector_root = vec_root_flat.view(N, 3, H)                # (N,3,H)

        vector_out = vector_root + vector_agg                    # (N,3,H)

        return torch.cat([scalar_out.unsqueeze(1), vector_out], dim=1)
