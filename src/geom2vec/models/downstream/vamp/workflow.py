
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import re
import warnings

from geom2vec.data.features import FlatFeatureSpec, packing_features
from .vampnet import VAMPNet, VAMPNetConfig

Trajectory = Union[np.ndarray, torch.Tensor]


def _ensure_ca_coords_tensor(
    coords: Union[torch.Tensor, np.ndarray],
    *,
    frames: int,
    num_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert arbitrary coordinate inputs to a validated tensor.

    Accepts either per-token coordinates of shape ``(num_tokens, 3)`` or a
    full trajectory of shape ``(frames, num_tokens, 3)`` and broadcasts /
    validates them against the requested :paramref:`frames` and
    :paramref:`num_tokens`.
    """
    tensor = torch.as_tensor(coords, dtype=dtype)
    if tensor.dim() == 2:
        if tensor.shape[0] != num_tokens or tensor.shape[1] != 3:
            raise ValueError(
                "ca_coords must have shape (num_tokens, 3) when providing a 2D tensor."
            )
        tensor = tensor.unsqueeze(0).expand(frames, -1, -1)
    elif tensor.dim() == 3:
        if tensor.shape[1] != num_tokens or tensor.shape[2] != 3:
            raise ValueError(
                "ca_coords must have shape (frames, num_tokens, 3) when providing a 3D tensor."
            )
        if tensor.shape[0] == 1 and frames != 1:
            tensor = tensor.expand(frames, -1, -1)
        elif tensor.shape[0] != frames:
            raise ValueError(
                "ca_coords first dimension must match number of frames or be 1 for broadcasting."
            )
    else:
        raise ValueError("ca_coords must be a 2D or 3D tensor.")
    return tensor


def _default_ca_coords(graph_features: torch.Tensor, *, dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Synthesize fallback CA coordinates from token indices for 4D inputs.

    Tokens are laid out on a 1D line along the x-axis with zeros in the
    remaining components; this is intended only as a geometric placeholder
    when real coordinates are unavailable.
    """
    if graph_features.ndim < 2:
        return None
    num_tokens = graph_features.shape[1]
    if num_tokens <= 0:
        raise ValueError("Number of tokens must be positive to build default coordinates.")
    frames = graph_features.shape[0]
    token_axis = torch.arange(num_tokens, dtype=dtype, device="cpu").view(1, num_tokens, 1)
    token_axis = token_axis.expand(frames, -1, -1)
    zeros = torch.zeros(frames, num_tokens, 2, dtype=dtype, device="cpu")
    return torch.cat([token_axis, zeros], dim=-1)


def _ensure_frame_weights(
    weights: Union[torch.Tensor, np.ndarray],
    *,
    frames: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Validate and convert per-frame weights to a tensor.

    Ensures a 1D array of length :paramref:`frames`, optionally squeezing a
    trailing singleton dimension.
    """
    tensor = torch.as_tensor(weights, dtype=dtype, device="cpu")
    if tensor.dim() == 2 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.dim() != 1:
        raise ValueError("frame_weights must be a 1D tensor or array.")
    if tensor.shape[0] != frames:
        raise ValueError("frame_weights must match the number of frames in the trajectory.")
    return tensor


def build_trajectories_from_embedding_dir(
    embedding_dir: Union[str, Path],
    *,
    require_ca: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[List[dict], List[torch.Tensor], List[torch.Tensor]]:
    """Load graph embeddings and optional CA coordinates from a directory.

    Expects embeddings saved as ``*.pt`` and CA coordinates saved as
    ``*_ca.pt`` in the same folder (as produced by :func:`infer_mdanalysis_folder`),
    with matching numeric suffixes (e.g. ``traj_48.pt`` and ``traj_48_ca.pt``).

    Returns
    -------
    trajectories :
        List of dictionaries suitable for :class:`VAMPWorkflow`, each with
        ``\"graph_features\"`` and, when present, ``\"ca_coords\"``.
    graph_trajectories :
        Raw graph feature tensors in the same order.
    ca_trajectories :
        Raw CA coordinate tensors in the same order (may be empty if
        ``require_ca=False`` and no CA files are found).
    """

    embedding_dir = Path(embedding_dir)

    def _get_number(name: str) -> int:
        """Extract the numeric suffix used for ordering trajectories.

        For filenames like ``traj_48.pt`` or ``CLN025-0-protein-036.pt``, we
        sort by the *last* integer in the stem so that corresponding embedding
        and CA files are aligned.
        """
        matches = re.findall(r"(\d+)", name)
        return int(matches[-1]) if matches else -1

    embedding_paths = [p for p in embedding_dir.glob("*.pt") if not p.name.endswith("_ca.pt")]
    ca_paths = list(embedding_dir.glob("*_ca.pt"))

    # Build maps by logical trajectory name (stem without the optional '_ca' suffix)
    emb_map = {p.stem: p for p in embedding_paths}
    ca_map = {p.stem.replace("_ca", ""): p for p in ca_paths}

    missing_ca = sorted(set(emb_map) - set(ca_map))
    missing_emb = sorted(set(ca_map) - set(emb_map))

    if not embedding_paths:
        raise FileNotFoundError(f"No embeddings found in {embedding_dir}. Run the inference notebook first.")
    if require_ca and missing_ca:
        raise ValueError(
            "Missing CA coordinate files for embeddings: "
            + ", ".join(missing_ca[:5])
            + ("..." if len(missing_ca) > 5 else "")
        )
    if require_ca and missing_emb:
        raise ValueError(
            "Found CA coordinate files without matching embeddings: "
            + ", ".join(missing_emb[:5])
            + ("..." if len(missing_emb) > 5 else "")
        )

    # Determine canonical ordering by the numeric suffix in the logical name.
    common_names = sorted(set(emb_map) & set(ca_map), key=_get_number)
    embedding_paths = [emb_map[name] for name in common_names]
    ca_paths = [ca_map[name] for name in common_names]

    graph_trajectories = [torch.load(path) for path in embedding_paths]
    ca_trajectories: List[torch.Tensor] = [torch.load(path) for path in ca_paths] if ca_paths else []

    trajectories: List[dict] = []
    for idx, feat in enumerate(graph_trajectories):
        if feat.dim() == 3:
            feat = feat.unsqueeze(1)  # (frames, num_tokens, 4, H)

        frames = feat.shape[0]
        num_tokens = feat.shape[1] if feat.ndim >= 2 else 1
        entry: dict = {"graph_features": feat}

        if ca_trajectories:
            ca = ca_trajectories[idx]
            ca_tensor = torch.as_tensor(ca, dtype=dtype)
            if ca_tensor.dim() == 3 and ca_tensor.shape[0] != frames:
                raise ValueError(
                    f"Frame mismatch between embeddings and CA coordinates for trajectory index {idx} "
                    f"({embedding_paths[idx].name} vs {ca_paths[idx].name}): "
                    f"{frames} feature frames vs {ca_tensor.shape[0]} CA frames. "
                    "Please regenerate these files with a consistent inference run."
                )

            coords = _ensure_ca_coords_tensor(ca, frames=frames, num_tokens=num_tokens, dtype=dtype)
            entry["ca_coords"] = coords

        trajectories.append(entry)

    return trajectories, graph_trajectories, ca_trajectories


class _TimeLaggedDataset(Dataset):
    """Create time-lagged (instantaneous, lagged) pairs from trajectory features.

    This dataset wraps one or more trajectories and returns pairs
    ``(x_t, x_tlag)`` (and optionally CA coordinates and per-pair weights)
    suitable for training a :class:`VAMPNet` lobe.
    """

    def __init__(
        self,
        sequences: Sequence[Union[torch.Tensor, dict, Tuple[torch.Tensor, Optional[torch.Tensor]]]],
        lag_time: int,
        dtype: torch.dtype,
    ) -> None:
        if lag_time <= 0:
            raise ValueError("`lag_time` must be a positive integer.")
        self._lag_time = lag_time
        self._dtype = dtype

        instantaneous_features: List[torch.Tensor] = []
        lagged_features: List[torch.Tensor] = []
        instantaneous_coords: List[torch.Tensor] = []
        lagged_coords: List[torch.Tensor] = []
        pair_weights_tensors: List[torch.Tensor] = []
        pair_weights_lag_tensors: List[torch.Tensor] = []
        feature_shape: Optional[torch.Size] = None

        for idx, seq in enumerate(sequences):
            tensor, coords, weights_t, weights_lag, rescaled_time = self._normalize_sequence(seq, dtype=dtype)

            if tensor.ndim < 2:
                raise ValueError(f"Trajectory {idx} must have shape (frames, features...).")
            frames = tensor.shape[0]
            if frames <= lag_time:
                continue

            if feature_shape is None:
                feature_shape = tensor.shape[1:]
            elif tensor.shape[1:] != feature_shape:
                raise ValueError("All trajectories must share the same feature shape.")

            idx_t: Optional[torch.Tensor] = None
            idx_lag: Optional[torch.Tensor] = None
            weight_pairs_t: Optional[torch.Tensor] = None
            weight_pairs_lag: Optional[torch.Tensor] = None

            if rescaled_time is not None:
                tprime = rescaled_time.detach().cpu().numpy()
                if tprime.size <= 1:
                    continue
                lag = float(self._lag_time)
                idx_end = np.searchsorted(tprime, tprime[-1] - lag, side="right") - 1
                if idx_end <= 0:
                    continue
                inst_indices = []
                lag_indices = []
                w_t_list = []
                w_lag_list = []
                for i in range(idx_end):
                    t_start = tprime[i]
                    t_next = tprime[i + 1]
                    stop_condition = lag + t_next
                    n_j = 0
                    for j in range(i, len(tprime) - 1):
                        if tprime[j] < stop_condition and tprime[j + 1] > t_start + lag:
                            inst_indices.append(i)
                            lag_indices.append(j)
                            delta_tau = min(t_next + lag, tprime[j + 1]) - max(t_start + lag, tprime[j])
                            w_lag_list.append(delta_tau)
                            n_j += 1
                        elif tprime[j] > stop_condition:
                            break
                    if n_j == 0:
                        continue
                    base = max(t_next - t_start, 0.0)
                    share = base / n_j if n_j else 0.0
                    for _ in range(n_j):
                        w_t_list.append(share)
                if not inst_indices:
                    continue
                idx_t = torch.tensor(inst_indices, dtype=torch.long)
                idx_lag = torch.tensor(lag_indices, dtype=torch.long)
                weight_pairs_t = torch.tensor(w_t_list, dtype=dtype)
                weight_pairs_lag = torch.tensor(w_lag_list, dtype=dtype)
            else:
                lag = self._lag_time
                if frames <= lag:
                    continue
                idx_t = torch.arange(frames - lag, dtype=torch.long)
                idx_lag = idx_t + lag
                if weights_t is not None or weights_lag is not None:
                    if weights_t is None:
                        weights_t = torch.ones(frames, dtype=dtype)
                    if weights_lag is None:
                        weights_lag = torch.ones(frames, dtype=dtype)
                    weight_pairs_t = weights_t.index_select(0, idx_t)
                    weight_pairs_lag = weights_lag.index_select(0, idx_lag)

            instantaneous_features.append(tensor.index_select(0, idx_t))
            lagged_features.append(tensor.index_select(0, idx_lag))

            if coords is not None:
                instantaneous_coords.append(coords.index_select(0, idx_t))
                lagged_coords.append(coords.index_select(0, idx_lag))

            if weight_pairs_t is not None and weight_pairs_lag is not None:
                pair_weights_tensors.append(weight_pairs_t)
                pair_weights_lag_tensors.append(weight_pairs_lag)

        if not instantaneous_features:
            raise ValueError("At least one trajectory with frames > lag_time is required.")

        self._instantaneous = torch.cat(instantaneous_features, dim=0)
        self._lagged = torch.cat(lagged_features, dim=0)
        if instantaneous_coords:
            self._instantaneous_coords = torch.cat(instantaneous_coords, dim=0)
            self._lagged_coords = torch.cat(lagged_coords, dim=0)
            self._has_coords = True
        else:
            self._instantaneous_coords = None
            self._lagged_coords = None
            self._has_coords = False

        if pair_weights_tensors:
            self._weights_t = torch.cat(pair_weights_tensors, dim=0)
            self._weights_lag = torch.cat(pair_weights_lag_tensors, dim=0)
            self._has_weights = True
        else:
            self._weights_t = None
            self._weights_lag = None
            self._has_weights = False

    def __len__(self) -> int:
        return self._instantaneous.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[Union[torch.Tensor, dict], Union[torch.Tensor, dict]]:
        if self._has_coords or self._has_weights:
            inst: dict = {"graph_features": self._instantaneous[index]}
            lag: dict = {"graph_features": self._lagged[index]}
            if self._has_coords:
                inst["ca_coords"] = self._instantaneous_coords[index]
                lag["ca_coords"] = self._lagged_coords[index]
            if self._has_weights:
                inst["weights"] = self._weights_t[index]
                lag["weights"] = self._weights_lag[index]
            return inst, lag
        return self._instantaneous[index], self._lagged[index]

    @property
    def instantaneous(self) -> Union[torch.Tensor, dict]:
        if self._has_coords or self._has_weights:
            data: dict = {"graph_features": self._instantaneous}
            if self._has_coords:
                data["ca_coords"] = self._instantaneous_coords
            if self._has_weights:
                data["weights"] = self._weights_t
            return data
        return self._instantaneous

    @property
    def lagged(self) -> Union[torch.Tensor, dict]:
        if self._has_coords or self._has_weights:
            data: dict = {"graph_features": self._lagged}
            if self._has_coords:
                data["ca_coords"] = self._lagged_coords
            if self._has_weights:
                data["weights"] = self._weights_lag
            return data
        return self._lagged

    @staticmethod
    def _normalize_sequence(
        sequence: Union[torch.Tensor, dict, Tuple[torch.Tensor, Optional[torch.Tensor]]],
        *,
        dtype: torch.dtype,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        rescaled_time_obj = None

        if isinstance(sequence, dict):
            if "graph_features" not in sequence:
                raise ValueError("Sequence dict must contain 'graph_features'.")
            graph = torch.as_tensor(sequence["graph_features"], dtype=dtype).to(device="cpu")
            coords_obj = sequence.get("ca_coords")
            weights_t_obj = sequence.get("frame_weights_t")
            weights_lag_obj = sequence.get("frame_weights_lag")
            rescaled_time_obj = sequence.get("rescaled_time")
            if weights_t_obj is None and weights_lag_obj is None:
                weights_obj = sequence.get("frame_weights")
            else:
                weights_obj = None
        elif isinstance(sequence, (tuple, list)):
            if len(sequence) not in {2, 3, 4, 5}:
                raise ValueError(
                    "Tuple/list trajectories must be (graph_features, ca_coords[, frame_weights_t, frame_weights_lag])."
                )
            graph = torch.as_tensor(sequence[0], dtype=dtype).to(device="cpu")
            coords_obj = sequence[1]
            weights_t_obj = None
            weights_lag_obj = None
            weights_obj = None
            rescaled_time_obj = None
            if len(sequence) == 3:
                weights_obj = sequence[2]
            elif len(sequence) == 4:
                weights_t_obj = sequence[2]
                weights_lag_obj = sequence[3]
            elif len(sequence) == 5:
                weights_t_obj = sequence[2]
                weights_lag_obj = sequence[3]
                rescaled_time_obj = sequence[4]
        else:
            graph = torch.as_tensor(sequence, dtype=dtype).to(device="cpu")
            coords_obj = None
            weights_t_obj = None
            weights_lag_obj = None
            weights_obj = None
            rescaled_time_obj = None

        coords: Optional[torch.Tensor] = None
        weights_t_tensor: Optional[torch.Tensor] = None
        weights_lag_tensor: Optional[torch.Tensor] = None
        rescaled_time_tensor: Optional[torch.Tensor] = None
        if coords_obj is not None:
            frames = graph.shape[0]
            num_tokens = graph.shape[1] if graph.ndim >= 2 else 1
            coords = _ensure_ca_coords_tensor(coords_obj, frames=frames, num_tokens=num_tokens, dtype=dtype)
        elif graph.ndim >= 4:
            coords = _default_ca_coords(graph, dtype=dtype)

        if coords is not None:
            coords = coords.to(device="cpu", dtype=dtype)

        frames = graph.shape[0]
        if weights_obj is not None:
            weights_t_obj = weights_lag_obj = weights_obj

        if weights_t_obj is not None:
            weights_t_tensor = _ensure_frame_weights(weights_t_obj, frames=frames, dtype=dtype)
            weights_t_tensor = weights_t_tensor.to(device="cpu", dtype=dtype)
        if weights_lag_obj is not None:
            weights_lag_tensor = _ensure_frame_weights(weights_lag_obj, frames=frames, dtype=dtype)
            weights_lag_tensor = weights_lag_tensor.to(device="cpu", dtype=dtype)
        if weights_t_tensor is None and weights_lag_tensor is not None:
            weights_t_tensor = weights_lag_tensor.clone()
        if weights_lag_tensor is None and weights_t_tensor is not None:
            weights_lag_tensor = weights_t_tensor.clone()

        if rescaled_time_obj is not None:
            rescaled_time_tensor = torch.as_tensor(rescaled_time_obj, dtype=torch.float64, device="cpu")
            if rescaled_time_tensor.dim() != 1 or rescaled_time_tensor.shape[0] != graph.shape[0]:
                raise ValueError("rescaled_time must be a 1D array matching number of frames")

        return graph, coords, weights_t_tensor, weights_lag_tensor, rescaled_time_tensor
@dataclass
class WorkflowDataSplit:
    train_sequences: List[Union[torch.Tensor, dict]]
    valid_sequences: List[Union[torch.Tensor, dict]]


class VAMPWorkflow:
    """High-level helper for preparing and training VAMPNet models.

    VAMPWorkflow takes raw trajectories (NumPy arrays or tensors), constructs
    time-lagged training pairs, and wraps a :class:`VAMPNet` instance with
    convenient ``fit`` / ``get_cvs`` / ``transform`` methods.
    """

    def __init__(
        self,
        lobe: torch.nn.Module,
        *,
        trajectories: Sequence[Trajectory],
        lag_time: int,
        frame_weights: Optional[Sequence[Trajectory]] = None,
        lobe_lagged: Optional[torch.nn.Module] = None,
        config: Optional[VAMPNetConfig] = None,
        train_fraction: float = 0.8,
        enable_validation: bool = True,
        shuffle_trajectories: bool = True,
        seed: Optional[int] = None,
        batch_size: int = 128,
        dtype: torch.dtype = torch.float32,
        drop_last: bool = False,
        num_workers: int = 0,
        train_shuffle: bool = True,
        loader_kwargs: Optional[dict] = None,
        concat_trajectories: bool = False,
        flatten: bool = False,
        token_reduction: Optional[str] = None,
    ) -> None:
        """Prepare trajectories and build data loaders for VAMPNet training.

        Parameters
        ----------
        lobe :
            Neural network mapping frames to features (see :class:`geom2vec.models.downstream.lobe.Lobe`).
        trajectories :
            Sequence of trajectories, each either an array/tensor of shape
            ``(frames, ...)`` or a dict/tuple carrying ``graph_features``,
            optional CA coordinates, frame weights and rescaled times.
        lag_time :
            Time lag (in frames) used to build ``(x_t, x_tlag)`` pairs.
        frame_weights :
            Optional per-frame weights for each trajectory. Can be provided
            directly or inferred from biased simulations (see
            :class:`BiasedVAMPWorkflow`).
        train_fraction :
            Fraction of available frames/trajectories to use for training
            (the remainder is used for validation when enabled).
        enable_validation :
            If ``True``, keep out a validation split and track validation scores.
        shuffle_trajectories :
            Whether to shuffle trajectories before splitting into train/validation.
        seed :
            Optional seed for trajectory shuffling.
        batch_size :
            Mini-batch size for the underlying :class:`DataLoader`.
        dtype :
            Torch dtype used when converting trajectories.
        drop_last :
            Whether to drop the last incomplete batch during training.
        num_workers :
            Number of workers for the underlying :class:`DataLoader`.
        train_shuffle :
            Whether to shuffle samples *within* the training loader.
        loader_kwargs :
            Additional keyword arguments forwarded to :class:`DataLoader`.
        concat_trajectories :
            If ``True``, concatenate all sequences into a single long trajectory
            before building time-lagged pairs.
        flatten :
            If ``True``, flatten token-shaped graph features into 2D arrays
            (optionally using :func:`packing_features` with CA coordinates).
        token_reduction :
            Optional reduction over tokens for 4D inputs (``None``, ``\"sum\"``,
            or ``\"mean\"``).
        """
        if not trajectories:
            raise ValueError("`trajectories` must contain at least one array or tensor.")
        if not (0.0 < train_fraction <= 1.0):
            raise ValueError("`train_fraction` must be in the interval (0, 1].")

        if frame_weights is not None and len(frame_weights) != len(trajectories):
            raise ValueError("`frame_weights` must have the same length as `trajectories`.")

        self._lag_time = int(lag_time)
        self._dtype = dtype
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._num_workers = num_workers
        self._train_shuffle = train_shuffle
        self._enable_validation = enable_validation and train_fraction < 1.0
        self._loader_kwargs = loader_kwargs or {}
        self._concat_trajectories = concat_trajectories
        self._flatten = flatten
        self._token_reduction = token_reduction
        self._frame_weights = frame_weights

        self._flat_spec: Optional[FlatFeatureSpec] = None

        prepared_sequences = self._prepare_sequences(trajectories, dtype, frame_weights=frame_weights)
        if self._concat_trajectories and len(prepared_sequences) > 1:
            concat_axis = 0
            graphs = [seq["graph_features"] for seq in prepared_sequences]
            graph_cat = torch.cat(graphs, dim=concat_axis)

            coord_list = [seq.get("ca_coords") for seq in prepared_sequences]
            if all(coord is None for coord in coord_list):
                coord_cat = None
            elif all(coord is not None for coord in coord_list):
                coord_cat = torch.cat(coord_list, dim=concat_axis)
            else:
                coord_cat = None if self._flatten else _default_ca_coords(graph_cat, dtype=dtype)

            weights_t_list = [seq.get("frame_weights_t") for seq in prepared_sequences]
            weights_lag_list = [seq.get("frame_weights_lag") for seq in prepared_sequences]
            if all(w is None for w in weights_t_list) and all(w is None for w in weights_lag_list):
                weight_t_cat = None
                weight_lag_cat = None
            elif all(w is not None for w in weights_t_list) and all(w is not None for w in weights_lag_list):
                weight_t_cat = torch.cat(weights_t_list, dim=concat_axis)
                weight_lag_cat = torch.cat(weights_lag_list, dim=concat_axis)
            else:
                raise ValueError("Cannot concatenate sequences with inconsistent frame weight availability.")

            combined = {"graph_features": graph_cat}
            if coord_cat is not None:
                combined["ca_coords"] = coord_cat
            if weight_t_cat is not None:
                combined["frame_weights_t"] = weight_t_cat
            if weight_lag_cat is not None:
                combined["frame_weights_lag"] = weight_lag_cat
            prepared_sequences = [combined]

        data_split = self._split_sequences(
            prepared_sequences,
            train_fraction=train_fraction,
            enable_validation=self._enable_validation,
            shuffle=shuffle_trajectories,
            seed=seed,
        )

        self._train_sequences = data_split.train_sequences
        self._valid_sequences = data_split.valid_sequences

        self._train_dataset, self._train_loader = self._build_loader(
            self._train_sequences,
            shuffle=self._train_shuffle,
        )
        self._valid_dataset, self._valid_loader = self._build_loader(
            self._valid_sequences,
            shuffle=False,
        )

        if self._train_loader is None:
            raise ValueError("Training split produced no samples. Check `lag_time` and data length.")

        self._vampnet = VAMPNet(
            lobe=lobe,
            lobe_lagged=lobe_lagged,
            config=config,
        )
        self._fitted = False

    def _prepare_sequences(
        self,
        trajectories: Sequence[Trajectory],
        dtype: torch.dtype,
        frame_weights: Optional[Sequence[Trajectory]] = None,
    ) -> List[dict]:
        prepared: List[dict] = []
        feature_shape: Optional[torch.Size] = None
        weight_sources: List[Optional[Trajectory]]
        if frame_weights is not None:
            weight_sources = list(frame_weights)
        else:
            weight_sources = [None] * len(trajectories)

        for idx, traj in enumerate(trajectories):
            weight_source = weight_sources[idx] if idx < len(weight_sources) else None
            coords_obj = None
            weights_t_obj = None
            weights_lag_obj = None
            fallback_weight_obj = None
            rescaled_time_obj = None

            if isinstance(traj, dict):
                if "graph_features" not in traj:
                    raise ValueError("Trajectory dict must contain 'graph_features'.")
                tensor = torch.as_tensor(traj["graph_features"], dtype=dtype).to(device="cpu")
                coords_obj = traj.get("ca_coords")
                weights_t_obj = traj.get("frame_weights_t")
                weights_lag_obj = traj.get("frame_weights_lag")
                rescaled_time_obj = traj.get("rescaled_time")
                fallback_weight_obj = traj.get("frame_weights")
            elif isinstance(traj, (tuple, list)):
                if len(traj) not in {2, 3, 4, 5}:
                    raise ValueError(
                        "Trajectory tuples must be (graph_features, ca_coords[, frame_weights_t, frame_weights_lag])."
                    )
                tensor = torch.as_tensor(traj[0], dtype=dtype).to(device="cpu")
                coords_obj = traj[1]
                if len(traj) == 3:
                    fallback_weight_obj = traj[2]
                elif len(traj) == 4:
                    weights_t_obj = traj[2]
                    weights_lag_obj = traj[3]
                elif len(traj) == 5:
                    weights_t_obj = traj[2]
                    weights_lag_obj = traj[3]
                    rescaled_time_obj = traj[4]
            else:
                tensor = torch.as_tensor(traj, dtype=dtype).to(device="cpu")
                coords_obj = None
                fallback_weight_obj = None
                rescaled_time_obj = None

            if weight_source is not None:
                if isinstance(weight_source, dict):
                    weights_t_obj = weight_source.get("t")
                    if weights_t_obj is None:
                        weights_t_obj = weight_source.get("frame_weights_t")
                    weights_lag_obj = weight_source.get("lag")
                    if weights_lag_obj is None:
                        weights_lag_obj = weight_source.get("frame_weights_lag")
                    if weights_t_obj is None and weights_lag_obj is None:
                        fallback_weight_obj = weight_source.get("frame_weights") or weight_source.get("w")
                elif isinstance(weight_source, (tuple, list)) and len(weight_source) == 2:
                    weights_t_obj, weights_lag_obj = weight_source
                else:
                    fallback_weight_obj = weight_source

            if weights_t_obj is None and weights_lag_obj is None and fallback_weight_obj is not None:
                weights_t_obj = fallback_weight_obj
                weights_lag_obj = fallback_weight_obj

            if tensor.ndim < 2:
                raise ValueError(f"Trajectory {idx} must be at least 2D (frames, features...).")
            if feature_shape is None:
                feature_shape = tensor.shape[1:]
            elif tensor.shape[1:] != feature_shape:
                raise ValueError("All trajectories must share the same feature dimensions.")

            if tensor.ndim == 4 and self._token_reduction is not None:
                if self._token_reduction == 'sum':
                    tensor = tensor.sum(dim=1)
                elif self._token_reduction == 'mean':
                    tensor = tensor.mean(dim=1)
                else:
                    raise ValueError("token_reduction must be None, 'sum', or 'mean'.")

            coords: Optional[torch.Tensor] = None
            frames = tensor.shape[0]
            num_tokens = tensor.shape[1] if tensor.ndim >= 2 else 1
            if coords_obj is not None:
                coords = _ensure_ca_coords_tensor(coords_obj, frames=frames, num_tokens=num_tokens, dtype=dtype).to(device="cpu")
            elif tensor.ndim >= 4:
                coords = _default_ca_coords(tensor, dtype=dtype)

            weights_t_tensor: Optional[torch.Tensor] = None
            weights_lag_tensor: Optional[torch.Tensor] = None
            rescaled_time_tensor: Optional[torch.Tensor] = None
            if weights_t_obj is not None:
                weights_t_tensor = _ensure_frame_weights(weights_t_obj, frames=frames, dtype=dtype).to(device="cpu")
            if weights_lag_obj is not None:
                weights_lag_tensor = _ensure_frame_weights(weights_lag_obj, frames=frames, dtype=dtype).to(device="cpu")
            if weights_t_tensor is None and weights_lag_tensor is not None:
                weights_t_tensor = weights_lag_tensor.clone()
            if weights_lag_tensor is None and weights_t_tensor is not None:
                weights_lag_tensor = weights_t_tensor.clone()

            if rescaled_time_obj is not None:
                rescaled_time_tensor = torch.as_tensor(rescaled_time_obj, dtype=torch.float64, device="cpu")
                if rescaled_time_tensor.dim() != 1 or rescaled_time_tensor.shape[0] != frames:
                    raise ValueError("rescaled_time must be 1D and match number of frames")

            if self._flatten:
                if tensor.ndim == 4:
                    num_tokens = tensor.shape[1]
                    hidden_dim = tensor.shape[-1]
                    if coords is None:
                        coords = _default_ca_coords(tensor, dtype=dtype)
                    tensor = packing_features(graph_features=tensor, num_tokens=num_tokens, ca_coords=coords)
                    if getattr(self, '_flat_spec', None) is None:
                        self._flat_spec = FlatFeatureSpec(num_tokens=num_tokens, hidden_dim=hidden_dim)
                    else:
                        if self._flat_spec.num_tokens != num_tokens or self._flat_spec.hidden_dim != hidden_dim:
                            raise ValueError("Flattened trajectories must share the same num_tokens/hidden_dim.")
                elif tensor.ndim not in {2, 3}:
                    tensor = tensor.reshape(tensor.shape[0], -1)
                coords_entry = None
            else:
                coords_entry = coords

            sequence_entry = {"graph_features": tensor}
            if coords_entry is not None:
                sequence_entry["ca_coords"] = coords_entry
            if weights_t_tensor is not None:
                sequence_entry["frame_weights_t"] = weights_t_tensor
            if weights_lag_tensor is not None:
                sequence_entry["frame_weights_lag"] = weights_lag_tensor
            if rescaled_time_tensor is not None:
                sequence_entry["rescaled_time"] = rescaled_time_tensor
            prepared.append(sequence_entry)
        return prepared
    def _split_sequences(
        self,
        sequences: Sequence[Union[torch.Tensor, dict]],
        *,
        train_fraction: float,
        enable_validation: bool,
        shuffle: bool,
        seed: Optional[int],
    ) -> WorkflowDataSplit:
        def _seq_length(seq: Union[torch.Tensor, dict]) -> int:
            tensor = seq if isinstance(seq, torch.Tensor) else seq["graph_features"]
            return tensor.shape[0]

        def _slice_sequence(seq: Union[torch.Tensor, dict], start: Optional[int], end: Optional[int]):
            slicing = slice(start, end)
            if isinstance(seq, torch.Tensor):
                return seq[slicing]
            graph = seq["graph_features"][slicing]
            coords = seq.get("ca_coords")
            if coords is not None:
                coords = coords[slicing]
            weights_t = seq.get("frame_weights_t")
            if weights_t is not None:
                weights_t = weights_t[slicing]
            weights_lag = seq.get("frame_weights_lag")
            if weights_lag is not None:
                weights_lag = weights_lag[slicing]
            result = {"graph_features": graph}
            if coords is not None:
                result["ca_coords"] = coords
            if weights_t is not None:
                result["frame_weights_t"] = weights_t
            if weights_lag is not None:
                result["frame_weights_lag"] = weights_lag
            return result

        if len(sequences) == 1:
            seq = sequences[0]
            if not enable_validation:
                return WorkflowDataSplit([seq], [])

            frames = _seq_length(seq)
            max_start = frames - self._lag_time
            if max_start <= 0:
                raise ValueError("Trajectory must have more frames than `lag_time` to create pairs.")

            split_frame = int(max_start * train_fraction)
            split_frame = max(self._lag_time + 1, split_frame)
            split_frame = min(split_frame, max_start)

            train_seq = _slice_sequence(seq, None, split_frame)
            val_seq = _slice_sequence(seq, split_frame, None)
            valid = [val_seq] if _seq_length(val_seq) > self._lag_time else []
            return WorkflowDataSplit([train_seq], valid)

        indices = list(range(len(sequences)))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(indices)

        n_train = max(1, int(round(len(sequences) * train_fraction)))
        if enable_validation and n_train >= len(sequences):
            n_train = len(sequences) - 1
        n_train = max(1, n_train)

        train_indices = indices[:n_train]
        valid_indices = indices[n_train:] if enable_validation else []

        train_sequences = [sequences[i] for i in train_indices]
        valid_sequences = [sequences[i] for i in valid_indices] if valid_indices else []
        return WorkflowDataSplit(train_sequences, valid_sequences)
    def _build_loader(
        self,
        sequences: Sequence[Union[torch.Tensor, dict]],
        *,
        shuffle: bool,
    ) -> Tuple[Optional[_TimeLaggedDataset], Optional[DataLoader]]:
        if not sequences:
            return None, None

        dataset = _TimeLaggedDataset(sequences, lag_time=self._lag_time, dtype=self._dtype)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            drop_last=self._drop_last,
            num_workers=self._num_workers,
            **self._loader_kwargs,
        )
        return dataset, loader

    @property
    def train_loader(self) -> DataLoader:
        assert self._train_loader is not None
        return self._train_loader

    @property
    def validation_loader(self) -> Optional[DataLoader]:
        return self._valid_loader

    @property
    def train_dataset(self) -> Optional[_TimeLaggedDataset]:
        return self._train_dataset

    @property
    def validation_dataset(self) -> Optional[_TimeLaggedDataset]:
        return self._valid_dataset



    def fit(
        self,
        n_epochs: int = 1,
        *,
        show_progress: bool = True,
        **kwargs,
    ) -> VAMPNet:
        """Train the underlying VAMPNet using the prepared data splits."""

        progress_fn = kwargs.pop("progress", None)
        if progress_fn is None:
            if show_progress:
                from tqdm.auto import tqdm

                def _progress(iterable, **tqdm_kwargs):
                    return tqdm(iterable, **tqdm_kwargs)

                progress_fn = _progress
            else:
                progress_fn = lambda iterable, **_: iterable

        score_bar = None
        if show_progress and self.validation_loader is not None:
            from tqdm.auto import tqdm

            score_bar = tqdm(total=0, leave=False, desc="best val: -inf")

            def _update_best(score: float) -> None:
                score_bar.set_description(f"best val: {score:.4f}")

        else:
            _update_best = None

        try:
            vamp = self._vampnet.fit(
                train_loader=self.train_loader,
                n_epochs=n_epochs,
                validation_loader=self.validation_loader,
                progress=progress_fn,
                score_callback=_update_best,
                **kwargs,
            )
        finally:
            if score_bar is not None:
                score_bar.close()

        self._fitted = True
        return vamp

    def get_cvs(
        self,
        *,
        split: str = "train",
        data: Optional[Sequence[Trajectory]] = None,
        instantaneous: bool = True,
        batch_size: int = 200,
    ) -> Sequence[np.ndarray]:
        """Return collective variables for the requested data split."""
        if not self._fitted and data is None:
            raise RuntimeError("Call `fit()` before requesting CVs, or provide `data` explicitly.")

        target_sequences: Sequence[Union[torch.Tensor, dict]]
        if data is not None:
            target_sequences = self._prepare_sequences(data, self._dtype)
        else:
            target_sequences = self._get_sequences_for_split(split)

        arrays = []
        for seq in target_sequences:
            tensor = seq if isinstance(seq, torch.Tensor) else seq["graph_features"]
            arrays.append(tensor.detach().cpu().numpy())
        return self._vampnet.transform(
            arrays,
            instantaneous=instantaneous,
            return_cv=True,
            lag_time=self._lag_time,
            batch_size=batch_size,
        )

    def transform(
        self,
        data: Sequence[Trajectory],
        *,
        instantaneous: bool = True,
        return_cv: bool = False,
        batch_size: int = 200,
    ):
        """Transform arbitrary trajectories via the underlying :class:`VAMPNet`.

        Accepts the same trajectory formats as the constructor and returns
        either feature space outputs or collective variables depending on
        :paramref:`return_cv`.
        """
        prepared_sequences = self._prepare_sequences(data, self._dtype)
        arrays = []
        for seq in prepared_sequences:
            tensor = seq if isinstance(seq, torch.Tensor) else seq["graph_features"]
            arrays.append(tensor.detach().cpu().numpy())
        return self._vampnet.transform(
            arrays,
            instantaneous=instantaneous,
            return_cv=return_cv,
            lag_time=self._lag_time,
            batch_size=batch_size,
        )

    def _get_sequences_for_split(self, split: str) -> Sequence[Union[torch.Tensor, dict]]:
        split = split.lower()
        if split == "train":
            return self._train_sequences
        if split in {"valid", "validation"}:
            if not self._valid_sequences:
                raise ValueError("Validation split is empty.")
            return self._valid_sequences
        if split == "all":
            return self._train_sequences + self._valid_sequences
        raise ValueError(f"Unknown split '{split}'. Choose from ['train', 'validation', 'all'].")


class BiasedVAMPWorkflow(VAMPWorkflow):
    """VAMP workflow variant that supports time-dependent bias reweighting.

    This subclass computes or accepts per-frame weights derived from bias
    potentials or log-weights and feeds them into :class:`VAMPWorkflow` so
    that the resulting time-lagged pairs are properly reweighted.
    """

    def __init__(
        self,
        lobe: torch.nn.Module,
        *,
        trajectories: Sequence[Trajectory],
        lag_time: int,
        biases: Optional[Sequence[Trajectory]] = None,
        log_weights: Optional[Sequence[Trajectory]] = None,
        beta: Optional[float] = None,
        normalize_log_weights: bool = True,
        frame_weights: Optional[Sequence[Trajectory]] = None,
        weights_mode: str = "weights_t",
        **kwargs,
    ) -> None:
        dtype: torch.dtype = kwargs.get("dtype", torch.float32)

        if frame_weights is not None and (biases is not None or log_weights is not None):
            raise ValueError("Provide either `frame_weights` or (`biases`/`log_weights`), not both.")

        if weights_mode not in {"weights_t", "both"}:
            raise ValueError("`weights_mode` must be either 'weights_t' or 'both'.")

        if frame_weights is None:
            if log_weights is None:
                if biases is None or beta is None:
                    raise ValueError(
                        "Specify either `log_weights` directly or both `biases` and `beta` to compute them."
                    )
                log_weights = [beta * np.asarray(bias) for bias in biases]

            frame_weights = []
            for logw in log_weights:
                logw_array = np.asarray(logw, dtype=np.float64).reshape(-1)
                if normalize_log_weights:
                    logw_array = logw_array - np.max(logw_array)
                weights = np.exp(logw_array)
                if weights_mode == "weights_t":
                    lag = int(lag_time)
                    w_t = weights.copy()
                    w_lag = np.zeros_like(w_t)
                    if lag <= 0:
                        w_lag[:] = w_t
                    elif lag < len(w_t):
                        w_lag[lag:] = w_t[:-lag]
                    frame_weights.append({"t": w_t, "lag": w_lag})
                else:
                    frame_weights.append({"t": weights.copy(), "lag": weights.copy()})
        else:
            processed_weights: List[Trajectory] = []
            for w in frame_weights:
                if isinstance(w, dict):
                    processed_weights.append({k: np.asarray(v) for k, v in w.items()})
                elif isinstance(w, (tuple, list)) and len(w) == 2:
                    processed_weights.append((np.asarray(w[0]), np.asarray(w[1])))
                else:
                    processed_weights.append(np.asarray(w))
            frame_weights = processed_weights

        super().__init__(
            lobe=lobe,
            trajectories=trajectories,
            lag_time=lag_time,
            frame_weights=frame_weights,
            **kwargs,
        )


__all__ = ["VAMPWorkflow", "BiasedVAMPWorkflow"]
