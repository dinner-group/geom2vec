from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear

from ..layers.equivariant import EquivariantScalar
from ..layers.mixers import SubFormer, SubMixer
from ..layers.mlps import MLP


class Lobe(nn.Module):
    """
    Initialize the Lobe model for geom2vec

    Args:
        hidden_channels: Number of input channels.
        intermediate_channels: Number of intermediate channels.
        output_channels: Number of output channels.
        num_layers: Number of layers in the MLP.
        batch_norm: Whether to use batch normalization. Defaults to False.
        vector_feature: Whether the input features are vector features.
            Defaults to True.
        mlp_dropout: Dropout probability for the MLP layers. Defaults to 0.0.
        mlp_out_activation: Activation function for the output layer of the
            MLP. Defaults to None.
        device: Device to use for computation. Defaults to torch.device("cpu").
        token_mixer: Type of token mixer to use. Can be "none", "subformer",
            or "submixer". Defaults to "none".
        num_mixer_layers: Number of layers for the token mixer. Defaults to 4.
        expansion_factor: Expansion factor for the transformer feed-forward
            layers. Defaults to 2.
        nhead: Number of attention heads for the transformer. Defaults to 8.
        pooling: Pooling strategy. Can be "cls", "mean", or "sum". Defaults
            to "cls".
        dropout: Dropout probability for the token mixer and transformer.
            Defaults to 0.1.
        attn_map: Whether to return the attention map of the transformer.
            Defaults to False.
        num_tokens: Number of tokens for the token mixer. Defaults to 1.
        token_dim: Dimension of tokens for the token mixer. Defaults to 64.
        attn_mask: Attention mask for the transformer. Defaults to None.
        pool_mask: Pool mask for the token mixer or transformer. Defaults
            to None.
    """

    def __init__(
        self,
        hidden_channels: int,
        intermediate_channels: int,
        output_channels: int,
        num_layers: int,
        batch_norm: bool = False,
        vector_feature: bool = True,
        mlp_dropout: float = 0.0,
        mlp_out_activation: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        token_mixer: str = "none",
        num_mixer_layers: int = 4,
        expansion_factor: int = 2,
        nhead: int = 8,
        pooling: str = "cls",
        dropout: float = 0.1,
        attn_map: bool = False,
        num_tokens: int = 1,
        token_dim: int = 64,
        attn_mask: Tensor = None,
        pool_mask: Tensor = None,
    ):
        super(Lobe, self).__init__()

        assert token_mixer in ["none", "subformer", "submixer"]
        assert pooling in ["cls", "mean", "sum"]

        if token_mixer == "submixer" and pooling == "cls":
            raise ValueError("Submixer does not support cls pooling")

        self.pooling = pooling

        self.token_mixer = token_mixer
        self.vector_feature = vector_feature
        self.dropout = Dropout(mlp_dropout)

        self.input_projection = EquivariantScalar(
            hidden_channels, intermediate_channels
        )
        if not vector_feature:
            self.input_projection = Linear(hidden_channels, intermediate_channels)

        attn_mask = attn_mask.to(device) if attn_mask is not None else None
        pool_mask = pool_mask.to(device) if pool_mask is not None else None

        if token_mixer == "none":
            self.mixer = None

        elif token_mixer == "subformer":
            self.mixer = SubFormer(
                hidden_channels=intermediate_channels,
                encoder_layers=num_mixer_layers,
                nhead=nhead,
                dim_feedforward=int(expansion_factor * intermediate_channels),
                pool=pooling,
                dropout=dropout,
                attn_map=attn_map,
                attn_mask=attn_mask,
                pool_mask=pool_mask,
                device=device,
                use_pos_embedding=True,
            )
        elif token_mixer == "submixer":
            self.mixer = SubMixer(
                num_patch=num_tokens,
                depth=num_mixer_layers,
                dropout=dropout,
                dim=intermediate_channels,
                token_dim=token_dim,
                channel_dim=int(expansion_factor * intermediate_channels),
                pool=pooling,
                pool_mask=pool_mask,
                device=device,
            )

        self.output_projection = MLP(
            input_channels=intermediate_channels,
            hidden_channels=intermediate_channels,
            out_channels=output_channels,
            num_layers=num_layers,
            out_activation=mlp_out_activation,
        )

        self.batch_norm = batch_norm

        if batch_norm:
            self.batchnorm = BatchNorm1d(intermediate_channels)

        self.to(device)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # several assumptions:
        # 1. if no mixer is used, input shape is (batch, 4, hidden_channels)
        # 2. if mixer is used, input shape is (batch,token,4,hidden_channels)
        # 3. the first dim of 4 is scalar feature, the rest are vector features

        assert data.shape[-2] == 4

        x = None
        if self.token_mixer == "none":
            x_rep = data[:, 0, :]
            v_rep = data[:, 1:, :]

            if not self.vector_feature:
                x_rep = self.input_projection(x_rep)
            else:
                x_rep, _ = self.input_projection.pre_reduce(x=x_rep, v=v_rep)

            x_rep = self.dropout(x_rep)
            if self.batch_norm:
                x_rep = self.batchnorm(x_rep)
            x = self.output_projection(x_rep)

        elif self.token_mixer == "subformer":
            batch_size, num_nodes, _, _ = data.shape
            x_rep = data[:, :, 0, :].reshape(batch_size * num_nodes, -1)
            v_rep = data[:, :, 1:, :].reshape(batch_size * num_nodes, 3, -1)
            if not self.vector_feature:
                x = self.input_projection(x_rep)
            else:
                x, _ = self.input_projection.pre_reduce(x=x_rep, v=v_rep)
            x = x.reshape(batch_size, num_nodes, -1)
            x = self.mixer(x)
            x = self.output_projection(x)

        elif self.token_mixer == "submixer":
            batch_size, num_nodes, _, dim = data.shape
            x_rep = data[:, :, 0, :].reshape(batch_size * num_nodes, -1)
            v_rep = data[:, :, 1:, :].reshape(batch_size * num_nodes, 3, -1)
            if not self.vector_feature:
                x = self.input_projection(x_rep)
            else:
                x, _ = self.input_projection.pre_reduce(x=x_rep, v=v_rep)
            x = x.reshape(batch_size, num_nodes, dim)
            x = self.mixer(x)
            x = self.output_projection(x)

        return x

    def fetch_attnmap(self, data: torch.Tensor) -> torch.Tensor:
        """
        Fetches the attention map for the given data.

        Args:
            data: The input data of shape (batch_size, num_nodes, num_features, feature_dim).

        Returns:
            torch.Tensor: The attention map of shape (batch_size, num_nodes, num_nodes).
        """
        assert self.token_mixer == "subformer"
        batch_size, num_nodes, _, _ = data.shape
        x_rep = data[:, :, 0, :].reshape(batch_size * num_nodes, -1)
        v_rep = data[:, :, 1:, :].reshape(batch_size * num_nodes, 3, -1)
        if not self.vector_feature:
            x = self.input_projection(x_rep)
        else:
            x, _ = self.input_projection.pre_reduce(x=x_rep, v=v_rep)
        x = x.reshape(batch_size, num_nodes, -1)
        attn_map = self.mixer.get_weights(x)
        return attn_map
