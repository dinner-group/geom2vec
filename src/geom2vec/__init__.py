from __future__ import annotations

from typing import Any, Callable, TypeVar

FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def _require_torch(name: str, error: ModuleNotFoundError) -> Callable[..., Any]:
    def _raiser(*_args: Any, **_kwargs: Any) -> None:
        raise ModuleNotFoundError(
            f"`geom2vec.{name}` requires PyTorch and related packages. "
            "Install optional extras (e.g. `pip install geom2vec[torch]`) to enable this feature."
        ) from error

    return _raiser


def create_model(*args: Any, **kwargs: Any):
    from geom2vec.models.factory import create_model as _create_model

    return _create_model(*args, **kwargs)

def infer_checkpoint_config(checkpoint_path: str):
    from geom2vec.models.factory import infer_checkpoint_config as _infer

    return _infer(checkpoint_path)


def create_model_from_checkpoint(*args: Any, **kwargs: Any):
    from geom2vec.models.factory import create_model_from_checkpoint as _create

    return _create(*args, **kwargs)


try:
    from geom2vec.models.downstream.lobe import Lobe
except ModuleNotFoundError as _torch_error:  # pragma: no cover - defensive guard
    Lobe = _require_torch("Lobe", _torch_error)

__all__ = ["create_model", "create_model_from_checkpoint", "infer_checkpoint_config", "Lobe"]
