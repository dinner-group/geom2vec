from __future__ import annotations

from typing import Optional

import torch


def fuse_env_features(
    graph_features: torch.Tensor,
    *,
    env_scalar: Optional[torch.Tensor] = None,
    env_vector: Optional[torch.Tensor] = None,
    mode: str = "concat",
) -> torch.Tensor:
    """Fuse implicit environment features into Geom2Vec scalar/vector token tensors.

    Parameters
    ----------
    graph_features:
        Base token tensor of shape ``(B, N, 4, C)`` (1 scalar + 3 vector components).
    env_scalar:
        Optional per-token scalar environment features of shape ``(B, N, C_s)``.
    env_vector:
        Optional per-token vector environment features of shape ``(B, N, 3, C_v)``.
    mode:
        Fusion mode. v1 supports ``"concat"`` only.
    """

    if mode != "concat":
        raise ValueError("Only mode='concat' is supported for environment fusion (v1).")

    if not isinstance(graph_features, torch.Tensor):
        raise TypeError("graph_features must be a torch.Tensor.")
    if graph_features.dim() != 4 or graph_features.shape[2] != 4:
        raise ValueError("graph_features must have shape (B, N, 4, C).")

    if env_scalar is None and env_vector is None:
        return graph_features

    batch_size, num_tokens, _, _ = graph_features.shape
    device = graph_features.device
    dtype = graph_features.dtype

    c_s = 0
    c_v = 0

    env_scalar_tensor: Optional[torch.Tensor] = None
    if env_scalar is not None:
        env_scalar_tensor = torch.as_tensor(env_scalar, device=device, dtype=dtype)
        if env_scalar_tensor.dim() != 3:
            raise ValueError("env_scalar must have shape (B, N, C_s).")
        if env_scalar_tensor.shape[0] != batch_size or env_scalar_tensor.shape[1] != num_tokens:
            raise ValueError("env_scalar batch/token dimensions must match graph_features.")
        c_s = int(env_scalar_tensor.shape[-1])

    env_vector_tensor: Optional[torch.Tensor] = None
    if env_vector is not None:
        env_vector_tensor = torch.as_tensor(env_vector, device=device, dtype=dtype)
        if env_vector_tensor.dim() != 4 or env_vector_tensor.shape[2] != 3:
            raise ValueError("env_vector must have shape (B, N, 3, C_v).")
        if env_vector_tensor.shape[0] != batch_size or env_vector_tensor.shape[1] != num_tokens:
            raise ValueError("env_vector batch/token dimensions must match graph_features.")
        c_v = int(env_vector_tensor.shape[-1])

    c_env = max(c_s, c_v)
    if c_env <= 0:
        return graph_features

    env_tensor = torch.zeros((batch_size, num_tokens, 4, c_env), device=device, dtype=dtype)
    if env_scalar_tensor is not None and c_s:
        env_tensor[:, :, 0, :c_s] = env_scalar_tensor
    if env_vector_tensor is not None and c_v:
        env_tensor[:, :, 1:, :c_v] = env_vector_tensor

    return torch.cat([graph_features, env_tensor], dim=-1)

