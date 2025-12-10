from __future__ import annotations

from typing import Any, Callable

from .plotting import ContourPlot2D


def _torch_guard(symbol: str, error: ModuleNotFoundError) -> Callable[..., Any]:
    def _raiser(*_args: Any, **_kwargs: Any) -> None:
        raise ModuleNotFoundError(
            f"`{symbol}` requires PyTorch and related geometric packages. "
            "Install optional extras (e.g. `pip install geom2vec[torch,pyg]`)."
        ) from error

    return _raiser


try:
    from .vampnet import VAMPNet as _VAMPNet
    from .workflow import VAMPWorkflow as _VAMPWorkflow, BiasedVAMPWorkflow as _BiasedVAMPWorkflow
except ModuleNotFoundError as _torch_error:
    VAMPNet = _torch_guard("geom2vec.models.downstream.vamp.VAMPNet", _torch_error)
    VAMPWorkflow = _torch_guard("geom2vec.models.downstream.vamp.VAMPWorkflow", _torch_error)
    BiasedVAMPWorkflow = _torch_guard("geom2vec.models.downstream.vamp.BiasedVAMPWorkflow", _torch_error)
else:
    VAMPNet = _VAMPNet
    VAMPWorkflow = _VAMPWorkflow
    BiasedVAMPWorkflow = _BiasedVAMPWorkflow

__all__ = ["VAMPNet", "VAMPWorkflow", "BiasedVAMPWorkflow", "ContourPlot2D"]
