import copy

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import radius_graph
from .gvp import GVP, GVPConvLayer
from .gvp import LayerNorm as gvp_layer_norm


def get_pos_embedding(n_pos, hidden_channels):
    r"""
    Get the positional embedding using sine and cosine functions.

    Args:
        n_pos (int): The number of positions.
        hidden_channels (int): The number of hidden units.
    """
    pos_embedding = torch.zeros(n_pos, hidden_channels)
    position = torch.arange(0, n_pos, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_channels, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_channels))
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    return pos_embedding  # [n_pos, hidden_channels]


class CustomTransformerEncoderLayer(TransformerEncoderLayer):
    """Transformer layer on coarsed graph, i.e. residue level (SubFormer)

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.
    """

    def __init__(self, *args, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class CustomTransformerEncoder(nn.Module):
    """Transformer encoder on coarsed graph, i.e. residue level (SubFormer)

    Args:
        encoder_layer (nn.Module): An instance of the CustomTransformerEncoderLayer class.
        num_layers (int): The number of sub-encoder-layers in the encoder.
    """

    def __init__(self, encoder_layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attention_weights = []

        for layer in self.layers:
            output, attn_weights = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
            attention_weights.append(attn_weights)

        return output, attention_weights


class SubFormer(nn.Module):
    r"""
    Transformer model on coarsed graph, i.e. residue level.

    Args:
        hidden_channels (int): The number of hidden units.
        encoder_layers (int): The number of encoder layers.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        pool (str): The pooling method.
        dropout (float): The dropout rate.
        attn_map (bool): Whether to return the attention map. i.e. using the custom transformer encoder layer.
    """

    def __init__(
            self,
            num_tokens,
            hidden_channels,
            encoder_layers,
            nhead,
            dim_feedforward,
            pool="cls",
            dropout=0.1,
            attn_map=True,
            attn_mask=None,
            pool_mask=None,
            device=torch.device("cpu"),
            use_pos_embedding=False,
    ):
        super(SubFormer, self).__init__()

        ## transformer encoder
        self.attn_map = attn_map
        if attn_map:
            encoder_layer = CustomTransformerEncoderLayer(
                batch_first=True,
                d_model=hidden_channels,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

            self.transformer_encoder = CustomTransformerEncoder(
                encoder_layer, num_layers=encoder_layers
            )
        else:
            encoder_layer = TransformerEncoderLayer(
                batch_first=True,
                d_model=hidden_channels,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

            self.transformer_encoder = TransformerEncoder(
                encoder_layer, num_layers=encoder_layers
            )

        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_channels))

        if attn_mask is not None:
            # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
            # suppose user input is (seq_len) true/false mask or additive mask
            if attn_mask.dtype == torch.bool:
                # create the square attention mask, True means masked and not allowed to attend
                # add a False to the first element of the mask, so that the cls token can attend to all nodes
                attn_mask = torch.cat(
                    [torch.tensor([False]).to(device), attn_mask], dim=0
                )
                # seq_len -> (seq_len, seq_len)
                attn_mask = attn_mask.unsqueeze(0) | attn_mask.unsqueeze(1)
                attn_mask = (
                    attn_mask.float()
                    .masked_fill(attn_mask == 0, float(0.0))
                    .masked_fill(attn_mask == 1, float("-inf"))
                )

            elif torch.is_floating_point(attn_mask):
                # additive weight to the attention score, can bias the attention score

                # add a 0 to the first element of the mask, so that the cls token is not affected by the mask
                attn_mask = torch.cat(
                    [torch.tensor([0.0]).to(device), attn_mask], dim=0
                )
                attn_mask = attn_mask.unsqueeze(0) + attn_mask.unsqueeze(1)

            else:
                raise ValueError("attn_mask must be a boolean tensor or a float tensor")

        self.attn_mask = attn_mask

        if pool_mask is not None:
            # add a False to the first element of the mask for the cls token
            self.pool_mask = torch.cat(
                [torch.tensor([False]).to(device), pool_mask], dim=0
            )
            assert (
                    self.pool != "cls"
            ), "pool_mask is not compatible with cls token pooling"

        if use_pos_embedding:
            self.pos_emb = get_pos_embedding(num_tokens, hidden_channels).to(device)

        self.to(device)

    def get_weights(self, data):
        src = data
        if hasattr(self, "pos_emb"):
            src = data + self.pos_emb

        if self.pool == "cls":
            src = torch.cat((self.cls_token.expand(src.size(0), -1, -1), src), dim=1)

        src, attention_weights = self.transformer_encoder(
            src, mask=self.attn_mask, src_key_padding_mask=None
        )
        return attention_weights

    def forward(self, data):
        src = data
        if hasattr(self, "pos_emb"):
            src = data + self.pos_emb

        src = torch.cat((self.cls_token.expand(src.size(0), -1, -1), src), dim=1)

        if self.attn_map:
            src, _ = self.transformer_encoder(
                src, mask=self.attn_mask, src_key_padding_mask=None
            )
        else:
            src = self.transformer_encoder(
                src, mask=self.attn_mask, src_key_padding_mask=None
            )

        if hasattr(self, "pool_mask"):
            src = src[:, ~self.pool_mask, :]  # remove the masked patches

            if self.pool == "mean":
                out = src[:, 1:, :].mean(dim=1)
            elif self.pool == "sum":
                out = src[:, 1:, :].sum(dim=1)

        else:
            if self.pool == "cls":
                out = src[:, 0, :]
            elif self.pool == "mean":
                out = src[:, 1:, :].mean(dim=1)
            elif self.pool == "sum":
                out = src[:, 1:, :].sum(dim=1)
            elif self.pool == "skip":
                out = src

        return out


# MLP-Mixer on coarsed graph, i.e. residue level (SubMixer)


class FeedForward(nn.Module):
    r"""Feedforward neural network with GELU activation.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden units.
        dropout (float): The dropout rate.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    r"""Mixer block.

    Args:
        dim (int): The number of input features.
        num_patch (int): The number of patches.
        token_dim (int): The number of token dimensions.
        channel_dim (int): The number of channel dimensions.
        dropout (float): The dropout rate.
    """

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.0):
        super().__init__()
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n d -> b d n"),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange("b d n -> b n d"),
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim, channel_dim, dropout)
        )

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class SubMixer(nn.Module):
    r"""Mixer model.

    Args:
        num_patch (int): The number of patches.
        depth (int): The number of mixer blocks.
        dropout (float): The dropout rate.
        dim (int): The number of input features.
        token_dim (int): The number of token dimensions.
        channel_dim (int): The number of channel dimensions.
        pooling (str): The pooling method.
    """

    def __init__(
            self,
            num_patch,
            depth,
            dropout,
            dim,
            token_dim,
            channel_dim,
            pool="mean",
            pool_mask=None,
            device=torch.device("cpu"),
    ):
        super().__init__()
        self.num_patch = num_patch
        self.depth = depth
        self.mixer = nn.Sequential(
            *[
                MixerBlock(dim, num_patch, token_dim, channel_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.pool = pool
        if pool not in ["mean", "sum"]:
            raise ValueError(f"Pooling should be either 'mean' or 'sum' but got {pool}")

        if pool_mask is not None:
            assert (
                    pool_mask.shape[0] == num_patch
            ), f"Input tensor has {pool_mask.shape[0]} patches, but expected {num_patch}"

            # assume pool_mask is a boolean tensor
            assert pool_mask.dtype == torch.bool, "pool_mask must be a boolean tensor"

            self.pool_mask = pool_mask

    def forward(self, x):
        assert (
                x.shape[1] == self.num_patch
        ), f"Input tensor has {x.shape[1]} patches, but expected {self.num_patch}"
        x = self.mixer(x)

        if hasattr(self, "pool_mask"):
            x = x[:, ~self.pool_mask, :]  # remove the masked patches

            if self.pool == "mean":
                x = x.mean(1)
            elif self.pool == "sum":
                x = x.sum(1)

        else:
            if self.pool == "mean":
                x = x.mean(1)
            elif self.pool == "sum":
                x = x.sum(1)

        return x


class SubGVP(nn.Module):
    r"""
    Graph Vector Perceptron model
    """

    def __init__(self,
                 num_tokens,
                 hidden_channels,
                 num_gvp_ff_layers,
                 num_layers,
                 dropout,
                 radius_cutoff=8.0,
                 vector_gating=False,
                 pooling="sum",
                 ):
        super(SubGVP, self).__init__()

        self.input_proj = GVP(
            in_dims=(hidden_channels, hidden_channels),
            out_dims=(hidden_channels, hidden_channels),
            vector_gate=vector_gating
        )
        self.input_norm = gvp_layer_norm((hidden_channels, hidden_channels))

        self.mp_layers = nn.ModuleList([
            GVPConvLayer(
                node_dims=(hidden_channels, hidden_channels),
                drop_rate=dropout,
                vector_gate=vector_gating,
                n_feedforward=num_gvp_ff_layers
            ) for _ in range(num_layers)
        ])

        self.radius_cutoff = radius_cutoff
        self.num_tokens = num_tokens
        self.pooling = pooling

    def forward(self, x, v, ca_coords):
        # input shape:
        # x: (batch_size, num_nodes, hidden_channels)
        # v: (batch_size, num_nodes, 3, hidden_channels)
        # ca_coords: (batch_size, num_nodes, 3)

        # creating the batching index
        batch_size, num_nodes, feature_dim = x.shape
        batch_index = torch.arange(batch_size).view(-1, 1).repeat(1, num_nodes).view(-1).to(x.device)
        ca_coords = ca_coords.reshape(-1, 3)

        # flatten the input as PyG format: (batch_size * num_nodes, hidden_channels)
        x = x.view(-1, feature_dim)
        v = v.view(-1, 3, feature_dim)
        # permute the vector features to (batch_size * num_nodes, feature_dim, 3)
        v = v.permute(0, 2, 1)
        # ca_coords = ca_coords.reshape(-1, 3)

        # construct the radius graph
        edge_index = radius_graph(ca_coords, r=self.radius_cutoff, batch=batch_index, loop=False)

        input_tuple = (x, v)  # gvp takes in a tuple of (scaler features, vector features)
        feature_tuples = self.input_proj(input_tuple)
        feature_tuples = self.input_norm(feature_tuples)

        for layer in self.mp_layers:
            feature_tuples = layer(feature_tuples, edge_index)
        x, v = feature_tuples
        # permute the vector features back to (batch_size*num_nodes, 3, feature_dim)
        v = v.permute(0, 2, 1)

        # reshape the output to the original shape
        x = x.view(batch_size, num_nodes, -1)
        v = v.view(batch_size, num_nodes, 3, -1)
        if self.pooling == "sum":
            x = x.sum(dim=1)
            v = v.sum(dim=1)
        elif self.pooling == "mean":
            x = x.mean(dim=1)
            v = v.mean(dim=1)
        elif self.pooling == "skip":
            pass

        return x, v
