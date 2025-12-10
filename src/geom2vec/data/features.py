from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch


def packing_features(
    graph_features: Union[torch.Tensor, List[torch.Tensor]],
    num_tokens: int,
    ca_coords: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Pack scalar/vector graph features (and optional CA coordinates) into flat tensors."""

    if isinstance(graph_features, list):
        if ca_coords is not None:
            if not isinstance(ca_coords, list):
                raise ValueError("ca_coords must be a list when graph_features is a list.")
            if len(ca_coords) != len(graph_features):
                raise ValueError("graph_features and ca_coords lists must have the same length.")

        packed: List[torch.Tensor] = []
        for idx, gf in enumerate(graph_features):
            ca = ca_coords[idx] if ca_coords is not None else None
            packed.append(packing_features(gf, num_tokens, ca))
        return packed

    if graph_features.dim() == 3:
        graph_features = graph_features.unsqueeze(1)
    elif graph_features.dim() != 4:
        raise ValueError(
            "graph_features must have 4 dimensions (batch, tokens, components, hidden) or 3 dimensions "
            "(batch, components, hidden) when num_tokens == 1."
        )

    batch_size, tensor_tokens, components, hidden_dim = graph_features.shape
    if components != 4:
        raise ValueError(f"Expected 4 components (scalar + 3 vectors), but received {components}.")
    if tensor_tokens != num_tokens:
        raise ValueError(
            f"graph_features tensor encodes {tensor_tokens} tokens, but num_tokens={num_tokens} was provided."
        )

    if ca_coords is None:
        if num_tokens > 1:
            raise ValueError("ca_coords must be provided when num_tokens > 1")
        ca_coords_tensor = torch.zeros(
            batch_size,
            num_tokens,
            3,
            dtype=graph_features.dtype,
            device=graph_features.device,
        )
    else:
        if isinstance(ca_coords, list):
            raise ValueError("ca_coords must be a tensor when graph_features is a tensor.")
        if ca_coords.dim() == 2:
            ca_coords_tensor = ca_coords.unsqueeze(1)
        elif ca_coords.dim() == 3:
            ca_coords_tensor = ca_coords
        else:
            raise ValueError("ca_coords must have shape (batch, tokens, 3) or (batch, 3) for num_tokens == 1.")

        expected_ca_shape = (batch_size, num_tokens, 3)
        if ca_coords_tensor.shape != expected_ca_shape:
            raise ValueError(f"ca_coords must have shape {expected_ca_shape}, but got {ca_coords_tensor.shape}")
        ca_coords_tensor = ca_coords_tensor.to(graph_features.dtype).to(graph_features.device)

    scaler = graph_features[:, :, 0, :]
    vectors = graph_features[:, :, 1:, :]
    vectors = torch.cat([vectors, ca_coords_tensor.unsqueeze(-1)], dim=-1)

    scaler_flat = scaler.reshape(batch_size, -1)
    vectors_flat = vectors.reshape(batch_size, -1)
    return torch.cat([scaler_flat, vectors_flat], dim=-1)


def unpacking_features(
    packed_features: torch.Tensor,
    num_tokens: int,
    hidden_dim: int,
) -> dict:
    """Unpack flattened tensors produced by :func:`packing_features`."""

    batch_size = packed_features.size(0)
    total_features = packed_features.size(1)
    features_per_token = 4 * hidden_dim + 3
    expected_with_ca = num_tokens * features_per_token
    legacy_features_per_token = 4 * hidden_dim
    expected_legacy = num_tokens * legacy_features_per_token

    if total_features == expected_with_ca:
        scaler_size = num_tokens * hidden_dim
        vector_size = num_tokens * 3 * (hidden_dim + 1)

        scaler_flat = packed_features[:, :scaler_size]
        vector_flat = packed_features[:, scaler_size:scaler_size + vector_size]

        scaler = scaler_flat.reshape(batch_size, num_tokens, hidden_dim)
        vector = vector_flat.reshape(batch_size, num_tokens, 3, hidden_dim + 1)

        vector_features = vector[:, :, :, :hidden_dim]
        ca_coords = vector[:, :, :, hidden_dim]

        graph_features = torch.cat([scaler.unsqueeze(2), vector_features], dim=2)
    elif num_tokens == 1 and total_features == expected_legacy:
        scaler_size = hidden_dim
        vector_size = 3 * hidden_dim

        scaler_flat = packed_features[:, :scaler_size]
        vector_flat = packed_features[:, scaler_size:scaler_size + vector_size]

        scaler = scaler_flat.reshape(batch_size, num_tokens, hidden_dim)
        vector = vector_flat.reshape(batch_size, num_tokens, 3, hidden_dim)

        graph_features = torch.cat([scaler.unsqueeze(2), vector], dim=2)
        ca_coords = None
    else:
        raise ValueError(
            "packed_features has incorrect shape. Expected second dimension to be "
            f"{expected_with_ca} (flat representation with coordinates) "
            + (
                f"or {expected_legacy} for legacy single-token data, "
                if num_tokens == 1
                else ""
            )
            + f"but received {total_features}."
        )

    return {
        "graph_features": graph_features,
        "ca_coords": ca_coords,
    }


@dataclass(frozen=True)
class FlatFeatureSpec:
    """Metadata describing flattened graph features."""

    num_tokens: int
    hidden_dim: int

    def unpack(self, packed: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = unpacking_features(packed, self.num_tokens, self.hidden_dim)
        return out["graph_features"], out["ca_coords"]





def forward_stop(in_domain):
    """
    Find the first exit time from the domain.

    Parameters
    ----------
    in_domain : (n,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (n,) ndarray of int
        First exit time from the domain for trajectories starting at
        each frame of the input trajectory. A first exit time not within
        the trajectory is indicated by len(d).

    """
    (t,) = np.nonzero(np.logical_not(in_domain))
    t = np.concatenate([[-1], t, [len(in_domain)]])
    return np.repeat(t[1:], np.diff(t))[:-1]


def backward_stop(in_domain):
    """
    Find the last entry time into the domain.

    Parameters
    ----------
    in_domain : (n,) ndarray of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (n,) ndarray of int
        Last entry time into the domain for trajectories starting at
        each frame of the input trajectory. A last entry time not within
        the trajectory is indicated by -1.

    """
    (t,) = np.nonzero(np.logical_not(in_domain))
    t = np.concatenate([[-1], t, [len(in_domain)]])
    return np.repeat(t[:-1], np.diff(t))[1:]


def count_transition_paths(in_domain, in_reactant, in_product):
    """
    Count the number of complete transition paths within the trajectory.

    Parameters
    ----------
    in_domain : (n,) ndarray of {bool, int}
        Whether each frame is in the domain.
    in_reactant : (n,) ndarray of {bool, int}
        Whether each frame is in the reactant.
    in_product : (n,) ndarray of {bool, int}
        Whether each frame is in the product.

    Returns
    -------
    int
        Number of complete transition paths.

    """
    (t,) = np.nonzero(np.logical_not(in_domain))
    return np.sum(in_reactant[t[:-1]] * in_product[t[1:]])


def count_transition_paths_windows(in_domain, in_reactant, in_product, lag_time):
    """
    Count the number of complete transition paths within each window.

    Parameters
    ----------
    in_domain : (n,) ndarray of {bool, int}
        Whether each frame is in the domain.
    in_reactant : (n,) ndarray of {bool, int}
        Whether each frame is in the reactant.
    in_product : (n,) ndarray of {bool, int}
        Whether each frame is in the product.
    lag_time : int
        Lag time in frames. Each window contains ``lag_time + 1`` frames.

    Returns
    -------
    (n - lag,) ndarray of int
        Number of complete transition paths within each window.

    """
    (t,) = np.nonzero(np.logical_not(in_domain))

    is_transition_path = np.logical_and(in_reactant[t[:-1]], in_product[t[1:]])

    initial_count = np.zeros(len(in_domain), dtype=int)
    initial_count[t[:-1][is_transition_path]] = 1
    initial_count = np.concatenate([[0], np.cumsum(initial_count)])

    final_count = np.zeros(len(in_domain), dtype=int)
    final_count[t[1:][is_transition_path]] = 1
    final_count = np.concatenate([[0], np.cumsum(final_count)])

    out = final_count[lag_time + 1:] - initial_count[: -(lag_time + 1)]
    out = np.maximum(out, 0)
    return out


def forward_stop_torch(in_domain):
    """
    Find the first exit time from the domain.

    Parameters
    ----------
    in_domain : (n,) tensor of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (n,) tensor of int
        First exit time from the domain for trajectories starting at
        each frame of the input trajectory. A first exit time not within
        the trajectory is indicated by len(d).

    """
    (t,) = torch.nonzero(torch.logical_not(in_domain), as_tuple=True)
    t = torch.cat((torch.tensor([-1]), t, torch.tensor([len(in_domain)])))
    return torch.repeat_interleave(t[1:], torch.diff(t))[:-1]


def backward_stop_torch(in_domain):
    """
    Find the last entry time into the domain.

    Parameters
    ----------
    in_domain : (n,) tensor of bool
        Input trajectory indicating whether each frame is in the domain.

    Returns
    -------
    (n,) tensor of int
        Last entry time into the domain for trajectories starting at
        each frame of the input trajectory. A last entry time not within
        the trajectory is indicated by -1.

    """
    (t,) = torch.nonzero(torch.logical_not(in_domain), as_tuple=True)
    t = torch.cat((torch.tensor([-1]), t, torch.tensor([len(in_domain)])))
    return torch.repeat_interleave(t[:-1], torch.diff(t))[1:]


def count_transition_paths_windows_torch(in_domain, in_reactant, in_product, lag_time):
    """
    Count the number of complete transition paths within each window.

    Parameters
    ----------
    in_domain : (n,) tensor of {bool, int}
        Whether each frame is in the domain.
    in_reactant : (n,) tensor of {bool, int}
        Whether each frame is in the reactant.
    in_product : (n,) tensor of {bool, int}
        Whether each frame is in the product.
    lag_time : int
        Lag time in frames. Each window contains ``lag_time + 1`` frames.

    Returns
    -------
    (n - lag,) tensor of int
        Number of complete transition paths within each window.

    """
    (t,) = torch.nonzero(torch.logical_not(in_domain), as_tuple=True)

    is_transition_path = torch.logical_and(in_reactant[t[:-1]], in_product[t[1:]])

    initial_count = torch.zeros(len(in_domain), dtype=torch.int)
    initial_count[t[:-1][is_transition_path]] = 1
    initial_count = torch.cat(
        [torch.zeros(1, dtype=torch.int), torch.cumsum(initial_count, dim=0)]
    )

    final_count = torch.zeros(len(in_domain), dtype=torch.int)
    final_count[t[1:][is_transition_path]] = 1
    final_count = torch.cat(
        [torch.zeros(1, dtype=torch.int), torch.cumsum(final_count, dim=0)]
    )

    out = final_count[lag_time:] - initial_count[:-lag_time]
    out = torch.maximum(out, torch.tensor(0, dtype=torch.int))
    return out


def count_transition_paths_torch(in_domain, in_reactant, in_product):
    """
    Count the number of complete transition paths within the trajectory.

    Parameters
    ----------
    in_domain : (n,) tensor of {bool, int}
        Whether each frame is in the domain.
    in_reactant : (n,) tensor of {bool, int}
        Whether each frame is in the reactant.
    in_product : (n,) tensor of {bool, int}
        Whether each frame is in the product.

    Returns
    -------
    int
        Number of complete transition paths.

    """
    (t,) = torch.nonzero(torch.logical_not(in_domain), as_tuple=True)
    return torch.sum(in_reactant[t[:-1]] * in_product[t[1:]])
