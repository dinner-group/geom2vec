from __future__ import annotations

from typing import Any, Callable


def _torch_guard(symbol: str, error: ModuleNotFoundError) -> Callable[..., Any]:
    def _raiser(*_args: Any, **_kwargs: Any) -> None:
        raise ModuleNotFoundError(
            f"`{symbol}` requires PyTorch and related geometric packages. "
            "Install optional extras (e.g. `pip install geom2vec[torch,pyg]`)."
        ) from error

    return _raiser


try:
    from .lobe import Lobe as _Lobe
except ModuleNotFoundError as _torch_error:
    Lobe = _torch_guard("geom2vec.models.downstream.Lobe", _torch_error)
else:
    Lobe = _Lobe

try:
    from .spib import SPIB as _SPIB, SPIBVAE as _SPIBVAE, SPIBModel as _SPIBModel
except ModuleNotFoundError as _torch_error:
    SPIB = _torch_guard("geom2vec.models.downstream.SPIB", _torch_error)
    SPIBVAE = _torch_guard("geom2vec.models.downstream.SPIBVAE", _torch_error)
    SPIBModel = _torch_guard("geom2vec.models.downstream.SPIBModel", _torch_error)
else:
    SPIB = _SPIB
    SPIBVAE = _SPIBVAE
    SPIBModel = _SPIBModel

try:
    from .vamp import (
        VAMPNet as _VAMPNet,
        VAMPWorkflow as _VAMPWorkflow,
        BiasedVAMPWorkflow as _BiasedVAMPWorkflow,
    )
except ModuleNotFoundError as _torch_error:
    VAMPNet = _torch_guard("geom2vec.models.downstream.VAMPNet", _torch_error)
    VAMPWorkflow = _torch_guard("geom2vec.models.downstream.VAMPWorkflow", _torch_error)
    BiasedVAMPWorkflow = _torch_guard("geom2vec.models.downstream.BiasedVAMPWorkflow", _torch_error)
else:
    VAMPNet = _VAMPNet
    VAMPWorkflow = _VAMPWorkflow
    BiasedVAMPWorkflow = _BiasedVAMPWorkflow

__all__ = [
    "Lobe",
    "SPIB",
    "SPIBVAE",
    "SPIBModel",
    "VAMPNet",
    "VAMPWorkflow",
    "BiasedVAMPWorkflow",
]
