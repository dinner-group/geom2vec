import argparse
import re
import warnings
from typing import Optional, Tuple

import torch
from torch import nn

from .output_modules import EquivariantScalar
from .et import TorchMD_ET
from .tensornet import TensorNet


def get_args(
    hidden_channels, num_layers, num_rbf, num_heads, cutoff=5.0, rep_model="et"
):
    # Directly create a Namespace object with the required arguments
    args = argparse.Namespace(
        embedding_dimension=hidden_channels,
        num_layers=num_layers,
        num_rbf=num_rbf,
        activation="silu",
        rbf_type="expnorm",
        trainable_rbf=False,
        neighbor_embedding=False,
        aggr="add",
        distance_influence="both",
        attn_activation="silu",
        num_heads=num_heads,
        layernorm_on_vec=None,
        derivative=False,
        cutoff_lower=0.0,
        cutoff_upper=cutoff,
        atom_filter=-1,
        max_z=100,
        max_num_neighbors=32,
        standardize=False,
        reduce_op="add",
        rep_model=rep_model,
    )
    return args


def create_model(args, prior_model=None, mean=None, std=None):
    shared_args = dict(
        hidden_channels=args.embedding_dimension,
        num_layers=args.num_layers,
        num_rbf=args.num_rbf,
        rbf_type=args.rbf_type,
        trainable_rbf=args.trainable_rbf,
        activation=args.activation,
        neighbor_embedding=args.neighbor_embedding,
        cutoff_lower=args.cutoff_lower,
        cutoff_upper=args.cutoff_upper,
        max_z=args.max_z,
        max_num_neighbors=args.max_num_neighbors,
    )

    # representation network

    representation_model = TorchMD_ET(
        attn_activation=args.attn_activation,
        num_heads=args.num_heads,
        distance_influence=args.distance_influence,
        layernorm_on_vec=args.layernorm_on_vec,
        **shared_args,
    )

    if args.rep_model == "tensornet":
        representation_model = TensorNet(
            hidden_channels=args.embedding_dimension,
            num_layers=args.num_layers,
            num_rbf=args.num_rbf,
            rbf_type=args.rbf_type,
            trainable_rbf=args.trainable_rbf,
            activation=args.activation,
            cutoff_lower=args.cutoff_lower,
            cutoff_upper=args.cutoff_upper,
            max_num_neighbors=args.max_num_neighbors,
            vector_output=True,
        )

    # create output network

    # create the denoising output network
    output_model_noise = EquivariantScalar(
        hidden_channels=args.embedding_dimension,
        activation=args.activation,
    )

    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        mean=mean,
        std=std,
        derivative=args.derivative,
        output_model_noise=output_model_noise,
    )
    return model


def load_model(filepath, args=None, device="cpu", mean=None, std=None, **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    loading_return = model.load_state_dict(state_dict, strict=False)

    if len(loading_return.unexpected_keys) > 0:
        # Should only happen if not applying denoising during fine-tuning.
        assert all(
            ("output_model_noise" in k or "pos_normalizer" in k)
            for k in loading_return.unexpected_keys
        )
    assert (
        len(loading_return.missing_keys) == 0
    ), f"Missing keys: {loading_return.missing_keys}"

    if mean:
        model.mean = mean
    if std:
        model.std = std

    return model.to(device)


class TorchMD_Net(nn.Module):
    def __init__(
        self,
        representation_model,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        output_model_noise=None,
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model

        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise

        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)

        self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()

    def forward(self, z, pos, batch: Optional[torch.Tensor] = None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        # run the potentially wrapped representation model
        x_rep, v_rep, z, pos, batch = self.representation_model(
            z=z, pos=pos, batch=batch
        )

        _, noise_pred = self.output_model_noise.pre_reduce(x_rep, v_rep, z, pos, batch)

        # print(noise_pred.shape)
        # print(x_rep.shape)
        # print(v_rep.shape)

        return x_rep, v_rep, noise_pred.squeeze()


class AccumulatedNormalization(torch.nn.Module):
    """Running normalization of a tensor."""

    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor, training: bool = True):
        if training:
            self.update_statistics(batch)
        return (batch - self.mean) / self.std


def main():
    args = get_args()
    print(args)
    model = create_model(args=args)
    print(model)
    z = torch.randint(0, 100, (10,))
    pos = torch.randn(10, 3)
    batch = torch.zeros(10)
    x, v, noise = model(z, pos, batch)
    print(x.shape, v.shape, noise.shape)


if __name__ == "__main__":
    main()
