from __future__ import annotations

from pathlib import Path
import re

from typing import TYPE_CHECKING, Union

__all__ = ["create_model", "infer_checkpoint_config", "create_model_from_checkpoint"]

if TYPE_CHECKING:
    import torch


def create_model(
    model_type: str,
    checkpoint_path: str | None = None,
    cutoff: float = 7.5,
    hidden_channels: int = 128,
    num_layers: int = 6,
    num_rbf: int = 64,
    device: Union[str, "torch.device"] = "cuda",
) -> "torch.nn.Module":
    """Instantiate a pretrained representation model."""
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        raise ModuleNotFoundError(
            "geom2vec.models.create_model requires PyTorch (`torch`) to be installed."
        ) from exc

    if model_type not in {"et", "vis", "tn"}:
        raise ValueError("Unsupported model_type {!r}. Supported models are 'et', 'vis', and 'tn'.".format(model_type))

    model = None

    if model_type == "et":
        from geom2vec.models.representation.torchmd.main_model import (
            create_model as build_model,
            get_args,
        )

        args = get_args(
            hidden_channels,
            num_layers,
            num_rbf,
            num_heads=8,
            cutoff=cutoff,
            rep_model="et",
        )
        model = build_model(args)
    elif model_type == "vis":
        from geom2vec.models.representation.visnet import ViSNet

        model = ViSNet(
            hidden_channels=hidden_channels,
            cutoff=cutoff,
            num_rbf=num_rbf,
            vecnorm_type="max_min",
            trainable_vecnorm=True,
        )
    elif model_type == "tn":
        from geom2vec.models.representation.torchmd.main_model import (
            create_model as build_model,
            get_args,
        )

        args = get_args(
            hidden_channels,
            num_layers,
            num_rbf,
            num_heads=8,
            cutoff=cutoff,
            rep_model="tensornet",
        )
        model = build_model(args)

    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print(f"Model loaded from {checkpoint_path}")
            model.eval()
        except Exception as e:
            print(f"Error loading the model from {checkpoint_path}")
            print(e)
    else:
        print("Model created from scratch.")

    return model.to(device)


def infer_checkpoint_config(checkpoint_path: str) -> dict:
    """Infer model hyperparameters from a checkpoint filename.

    Parameters
    ----------
    checkpoint_path : str
        Path to a checkpoint named with the convention
        `<arch>_l{layers}_h{hidden}_rbf{rbf}_r{cutoff*10}.pth`.

    Returns
    -------
    dict
        Parsed keyword arguments compatible with :func:`create_model` (includes `model_type`).
    """

    stem = Path(checkpoint_path).stem.lower()
    tokens = stem.split("_")
    if not tokens:
        raise ValueError(f"Cannot parse checkpoint name from '{checkpoint_path}'.")

    arch = tokens[0]
    model_map = {
        "tensornet": "tn",
        "visnet": "vis",
        "et": "et",
        "torchmdet": "et",
    }
    try:
        model_type = model_map[arch]
    except KeyError as exc:  # pragma: no cover - defensive guard for unknown patterns
        raise ValueError(f"Unsupported checkpoint architecture prefix '{arch}'.") from exc

    config: dict[str, object] = {"model_type": model_type}

    for token in tokens[1:]:
        match = re.match(r"([a-z]+)([0-9]+)", token)
        if not match:
            continue
        key, value = match.groups()
        if key == "l":
            config.setdefault("num_layers", int(value))
        elif key == "h":
            config.setdefault("hidden_channels", int(value))
        elif key == "rbf":
            config.setdefault("num_rbf", int(value))
        elif key == "r":
            cutoff_int = int(value)
            cutoff = cutoff_int / 10.0 if cutoff_int > 10 else float(cutoff_int)
            config.setdefault("cutoff", cutoff)

    return config


def create_model_from_checkpoint(
    checkpoint_path: str,
    *,
    device: Union[str, "torch.device"] = "cuda",
    **overrides,
) -> "torch.nn.Module":
    """Instantiate a model using hyperparameters inferred from the checkpoint name."""

    config = infer_checkpoint_config(checkpoint_path)
    config.update(overrides)
    model_type = config.pop("model_type")
    return create_model(
        model_type=model_type,  # type: ignore[arg-type]
        checkpoint_path=str(checkpoint_path),
        device=device,
        **config,
    )
