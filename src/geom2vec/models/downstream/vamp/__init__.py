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
    from .vampnet import VAMPNet as _VAMPNet
    from .stopvampnet import StopVAMPNet as _StopVAMPNet
    from .workflow import (
        VAMPWorkflow as _VAMPWorkflow,
        StopVAMPWorkflow as _StopVAMPWorkflow,
        BiasedVAMPWorkflow as _BiasedVAMPWorkflow,
    )
except ModuleNotFoundError as _torch_error:
    VAMPNet = _torch_guard("geom2vec.models.downstream.vamp.VAMPNet", _torch_error)
    StopVAMPNet = _torch_guard("geom2vec.models.downstream.vamp.StopVAMPNet", _torch_error)
    VAMPWorkflow = _torch_guard("geom2vec.models.downstream.vamp.VAMPWorkflow", _torch_error)
    StopVAMPWorkflow = _torch_guard("geom2vec.models.downstream.vamp.StopVAMPWorkflow", _torch_error)
    BiasedVAMPWorkflow = _torch_guard("geom2vec.models.downstream.vamp.BiasedVAMPWorkflow", _torch_error)
else:
    VAMPNet = _VAMPNet
    StopVAMPNet = _StopVAMPNet
    VAMPWorkflow = _VAMPWorkflow
    StopVAMPWorkflow = _StopVAMPWorkflow
    BiasedVAMPWorkflow = _BiasedVAMPWorkflow

__all__ = ["VAMPNet", "StopVAMPNet", "VAMPWorkflow", "StopVAMPWorkflow", "BiasedVAMPWorkflow"]
