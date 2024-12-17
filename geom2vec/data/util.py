import numpy as np
import torch
from typing import Optional, Union, List


def packing_features(
        graph_features: Union[torch.Tensor, List[torch.Tensor]],
        num_tokens: int,
        global_features: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        ca_coords: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Prepare the input data as a flat tensor by combining graph features, CA coordinates, and global features.
    Supports both single trajectories and a list of trajectories.

    Parameters
    ----------
    graph_features : torch.Tensor or List[torch.Tensor]
        Tensor or list of tensors containing graph features.
        - Single trajectory:
            Shape: (batch, num_tokens, 4, hidden) if num_tokens > 1
                   (batch, 4, hidden) if num_tokens == 1
        - Multiple trajectories:
            Each element in the list should follow the single trajectory shape.
    num_tokens : int
        Number of tokens (e.g., amino acids).
        - Single trajectory: int
        - Multiple trajectories: list of ints
    global_features : torch.Tensor or List[torch.Tensor], optional
        Tensor or list of tensors containing global features.
        - Single trajectory:
            Shape: (batch, global_dim)
        - Multiple trajectories:
            Each element in the list should follow the single trajectory shape.
        If None, global features are not included.
    ca_coords : torch.Tensor or List[torch.Tensor], optional
        Tensor or list of tensors containing CA coordinates.
        - Single trajectory:
            Shape: (batch, num_tokens, 3)
        - Multiple trajectories:
            Each element in the list should follow the single trajectory shape.
        Required if num_tokens > 1.

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        Packed feature tensor or list of packed feature tensors.
        - Single trajectory:
            Packed tensor as per the original function.
        - Multiple trajectories:
            List of packed tensors, one for each trajectory.

    Raises
    ------
    ValueError
        If input tensor dimensions do not align with expectations or input types mismatch.
    """
    # Check if inputs are lists (i.e., multiple trajectories)
    is_list_input = isinstance(graph_features, list)

    if is_list_input:
        # Validate that other inputs are lists or None
        if global_features is not None and not isinstance(global_features, list):
            raise ValueError("global_features must be a list when graph_features is a list.")
        if ca_coords is not None and not isinstance(ca_coords, list):
            raise ValueError("ca_coords must be a list when graph_features is a list.")

        print("Multiple trajectories detected.")
        print('length of graph_features:', len(graph_features))
        print('length of global_features:', len(global_features) if global_features is not None else None)
        print('length of ca_coords:', len(ca_coords) if ca_coords is not None else None)


        if not ((global_features is None or len(global_features) == len(graph_features)) and
                (ca_coords is None or len(ca_coords) == len(graph_features))):
            raise ValueError("All input lists must have the same length.")

        packed_features_list = []
        for i in range(len(graph_features)):
            gf = graph_features[i]
            nt = num_tokens
            gf_global = global_features[i] if global_features is not None else None
            ca = ca_coords[i] if ca_coords is not None else None

            packed = packing_features(gf, nt, gf_global, ca)  # Recursive call for single trajectory
            packed_features_list.append(packed)

        return packed_features_list

    else:
        # Single trajectory processing (original function logic)
        batch_size = graph_features.size(0)

        if isinstance(num_tokens, list):
            raise ValueError("num_tokens should be an integer when graph_features is not a list.")

        if num_tokens != 1:
            expected_shape = (batch_size, num_tokens, 4, graph_features.size(-1))
            if graph_features.dim() != 4 or graph_features.shape[:3] != expected_shape[:3]:
                raise ValueError(
                    f"graph_features must have shape {expected_shape}, but got {graph_features.shape}"
                )
            if ca_coords is None:
                raise ValueError("ca_coords must be provided when num_tokens > 1")
            if ca_coords.shape != (batch_size, num_tokens, 3):
                raise ValueError(
                    f"ca_coords must have shape ({batch_size}, {num_tokens}, 3), but got {ca_coords.shape}"
                )
        else:
            expected_shape = (batch_size, 4, graph_features.size(-1))
            if graph_features.dim() != 3 or graph_features.shape != expected_shape:
                raise ValueError(
                    f"graph_features must have shape {expected_shape}, but got {graph_features.shape}"
                )
            if ca_coords is not None:
                raise ValueError("ca_coords should be None when num_tokens == 1")

        if global_features is not None:
            if global_features.size(0) != batch_size:
                raise ValueError(
                    "Number of samples in global_features does not match the batch size of graph_features"
                )

        if num_tokens != 1:
            # Split graph features into scaler and vector components
            scaler_features = graph_features[:, :, 0, :]  # Shape: (batch, num_tokens, hidden)
            vector_features = graph_features[:, :, 1:, :]  # Shape: (batch, num_tokens, 3, hidden)

            # Concatenate CA coordinates to vector features
            ca_coords_expanded = ca_coords.unsqueeze(-1)  # Shape: (batch, num_tokens, 3, 1)
            vector_features = torch.cat([vector_features, ca_coords_expanded],
                                        dim=-1)  # (batch, num_tokens, 3, hidden +1)

            # Flatten scaler and vector features
            scaler_flat = scaler_features.reshape(batch_size, -1)  # (batch, num_tokens * hidden)
            vector_flat = vector_features.reshape(batch_size, -1)  # (batch, num_tokens * 3 * (hidden +1))

            # Concatenate all features
            if global_features is not None:
                packed_features = torch.cat([scaler_flat, vector_flat, global_features], dim=-1)
            else:
                packed_features = torch.cat([scaler_flat, vector_flat], dim=-1)
        else:
            # Split graph features into scaler and vector components
            scaler_features = graph_features[:, 0, :]  # Shape: (batch, hidden)
            vector_features = graph_features[:, 1:, :]  # Shape: (batch, 3, hidden)

            # Flatten scaler and vector features
            scaler_flat = scaler_features.reshape(batch_size, -1)  # (batch, hidden)
            vector_flat = vector_features.reshape(batch_size, -1)  # (batch, 3 * hidden)

            # Concatenate all features
            if global_features is not None:
                packed_features = torch.cat([scaler_flat, vector_flat, global_features], dim=-1)
            else:
                packed_features = torch.cat([scaler_flat, vector_flat], dim=-1)

        return packed_features



def unpacking_features(
        packed_features: torch.Tensor,
        num_tokens: int,
        hidden_dim: int,
        global_dim: int
) -> dict:
    """
    Unpack the packed feature tensor into graph features, CA coordinates, and global features.

    Parameters
    ----------
    packed_features : torch.Tensor
        Tensor containing packed features.
        Shape:
            - If num_tokens > 1:
                (batch, num_tokens * hidden + num_tokens * 3 * (hidden + 1) + global_dim)
            - If num_tokens == 1:
                (batch, 4 * hidden + global_dim)
    num_tokens : int
        Number of tokens (e.g., amino acids) per sample.
    hidden_dim : int
        Dimension of the hidden features in graph_features.
    global_dim : int
        Dimension of the global features.

    Returns
    -------
    dict
        A dictionary containing:
            - 'graph_features': torch.Tensor
                Original graph features.
                Shape: (batch, num_tokens, 4, hidden) if num_tokens > 1
                       (batch, 4, hidden) if num_tokens == 1
            - 'ca_coords': torch.Tensor or None
                CA coordinates.
                Shape: (batch, num_tokens, 3) if num_tokens > 1
                None if num_tokens == 1
            - 'global_features': torch.Tensor
                Global features.
                Shape: (batch, global_dim)
    """
    batch_size = packed_features.size(0)

    if num_tokens != 1:
        # Calculate sizes based on packing logic
        scaler_size = num_tokens * hidden_dim
        vector_size = num_tokens * 3 * (hidden_dim + 1)
        expected_size = scaler_size + vector_size + global_dim

        if packed_features.size(1) != scaler_size + vector_size + global_dim:
            raise ValueError(
                f"packed_features has incorrect shape. Expected second dimension to be {scaler_size + vector_size + global_dim}, but got {packed_features.size(1)}"
            )

        # Extract scaler, vector, and global features
        scaler_flat = packed_features[:, :scaler_size]  # (batch, num_tokens * hidden_dim)
        vector_flat = packed_features[:,
                      scaler_size:scaler_size + vector_size]  # (batch, num_tokens * 3 * (hidden_dim +1))
        global_features = packed_features[:, scaler_size + vector_size:]  # (batch, global_dim)

        # Reshape scaler and vector features
        scaler = scaler_flat.reshape(batch_size, num_tokens, hidden_dim)  # (batch, num_tokens, hidden_dim)
        vector = vector_flat.reshape(batch_size, num_tokens, 3, hidden_dim + 1)  # (batch, num_tokens, 3, hidden_dim +1)

        # Split vector into original vector features and CA coordinates
        vector_features = vector[:, :, :, :hidden_dim]  # (batch, num_tokens, 3, hidden_dim)
        ca_coords = vector[:, :, :, hidden_dim]  # (batch, num_tokens, 3)

        # Reconstruct graph_features by concatenating scaler and vector
        graph_features = torch.cat(
            [scaler.unsqueeze(2), vector_features],
            dim=2
        )  # (batch, num_tokens, 4, hidden_dim)

    else:
        # Calculate sizes based on packing logic
        scaler_size = hidden_dim
        vector_size = 3 * hidden_dim
        expected_size = scaler_size + vector_size + global_dim

        if packed_features.size(1) != scaler_size + vector_size + global_dim:
            raise ValueError(
                f"packed_features has incorrect shape. Expected second dimension to be {scaler_size + vector_size + global_dim}, but got {packed_features.size(1)}"
            )

        # Extract scaler, vector, and global features
        scaler_flat = packed_features[:, :scaler_size]  # (batch, hidden_dim)
        vector_flat = packed_features[:, scaler_size:scaler_size + vector_size]  # (batch, 3 * hidden_dim)
        global_features = packed_features[:, scaler_size + vector_size:]  # (batch, global_dim)

        # Reshape scaler and vector features
        scaler = scaler_flat.reshape(batch_size, hidden_dim)  # (batch, hidden_dim)
        vector = vector_flat.reshape(batch_size, 3, hidden_dim)  # (batch, 3, hidden_dim)

        # Reconstruct graph_features by concatenating scaler and vector
        graph_features = torch.cat(
            [scaler.unsqueeze(1), vector],
            dim=1
        )  # (batch, 4, hidden_dim)

        ca_coords = None  # CA coordinates are not present when num_tokens ==1

    return {
        'graph_features': graph_features,
        'ca_coords': ca_coords,
        'global_features': global_features
    }


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
