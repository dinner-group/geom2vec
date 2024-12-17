from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..layers.equivariant import EquivariantScalar
from ..layers.mixers import SubFormer, SubMixer
from ..layers.mlps import MLP


class OTFLobe(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_channels=128,
                 vector_features=True,
                 representation_model=None,
                 atomic_numbers=None
                 ):
        super(OTFLobe, self).__init__()

        self.atom_projection = EquivariantScalar(input_dim, hidden_channels)

        if not vector_features:
            self.atom_projection = nn.Linear(input_dim, hidden_channels)

        self.vector_features = vector_features

        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, output_dim)
        )

        self.activation = nn.SiLU()
        self.representation_model = representation_model
        self.z = torch.from_numpy(atomic_numbers)

    def forward(self, data):
        # suppose the input shape is (bz,atoms,3)
        n_samples, n_atoms, _ = data.shape
        z_batch = self.z.expand(n_samples, -1).reshape(-1).to(data.device)
        batch_batch = (
            torch.arange(n_samples).unsqueeze(1).expand(-1, n_atoms).reshape(-1)
        ).to(data.device)
        x_rep, v_rep, _ = self.representation_model(z=z_batch,
                                                    pos=data.reshape(-1, 3).contiguous().to(data.device),
                                                    batch=batch_batch)
        x_rep = x_rep.reshape(n_samples, n_atoms, -1).sum(dim=1)
        v_rep = v_rep.reshape(n_samples, n_atoms, 3, -1).sum(dim=1)

        if not self.vector_features:
            x_rep = self.atom_projection(x_rep)
        else:
            x_rep, _ = self.atom_projection.pre_reduce(x=x_rep, v=v_rep)
        out = self.graph_projection(x_rep)

        return out


class Lobe(nn.Module):
    """
    Initialize the Lobe model for geom2vec for on-the-fly representation learning.

    Args:
        hidden_channels: Number of input channels.
        intermediate_channels: Number of intermediate channels.
        output_channels: Number of output channels.
        num_layers: Number of layers in the MLP.
        atomic_numbers: Atomic numbers of the atoms.
        representation_model: Representation model.
        batch_norm: Whether to use batch normalization. Defaults to False.
        vector_feature: Whether the input features are vector features
         Defaults to True.
        mlp_dropout: Dropout probability for the MLP layers. Defaults to 0.0.
        mlp_out_activation: Activation function for the output layer of the MLP.
            Defaults to None.
        device: Device to use for computation. Defaults to torch.device("cpu").
        token_mixer: Type of token mixer to use. Can be "none", "subformer", or
            "submixer". Defaults to "none".
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
        pool_mask: Pool mask for the token mixer or transformer. Defaults to
            None.
    """

    def __init__(
            self,
            hidden_channels: int,
            intermediate_channels: int,
            output_channels: int,
            num_layers: int,
            atomic_numbers: torch.Tensor,
            representation_model: nn.Module,
            batch_norm: bool = False,
            vector_feature: bool = True,
            mlp_dropout: float = 0.0,
            mlp_out_activation=Optional[nn.Module],
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
        self.dropout = torch.nn.Dropout(mlp_dropout)

        self.z = atomic_numbers.to(device)
        self.rep_model = representation_model.to(device)
        self.device = device
        self.hidden_channels = hidden_channels

        self.input_projection = EquivariantScalar(
            hidden_channels, intermediate_channels
        )
        if not vector_feature:
            self.input_projection = torch.nn.Linear(
                hidden_channels, intermediate_channels
            )

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
            self.batchnorm = nn.BatchNorm1d(intermediate_channels)

        self.to(device)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # now the input for the on the fly model is the atomic positions (batch, num_atoms, 3)

        n_samples, n_atoms, _ = data.shape
        z_batch = self.z.expand(n_samples, -1).reshape(-1).to(self.device)
        batch_batch = (
            torch.arange(n_samples).unsqueeze(1).expand(-1, n_atoms).reshape(-1)
        ).to(self.device)
        x_rep, v_rep, _ = self.rep_model(
            z=z_batch,
            pos=data.reshape(-1, 3).contiguous().to(self.device),
            batch=batch_batch,
        )

        x_rep = x_rep.reshape(-1, n_atoms, 1, self.hidden_channels)
        v_rep = v_rep.reshape(-1, n_atoms, 3, self.hidden_channels)
        data = torch.cat([x_rep, v_rep], dim=-2)

        if self.token_mixer == "none":
            data = data.sum(1)

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

    def fetch_attnmap(self, data):
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
