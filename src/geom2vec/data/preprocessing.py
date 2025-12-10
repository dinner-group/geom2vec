import os
from copy import copy
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from geom2vec.data.features import FlatFeatureSpec, packing_features


class Preprocessing:
    """
    Preprocess the original trajectories to create datasets for training.
    All data is converted to PyTorch tensors.

    Parameters
    ----------
    dtype : torch.dtype, optional
        Data type of the tensors. Default is torch.float32.
    num_tokens : int
        The number of tokens to be used for encoding the data. e.g. num. amino acids in a protein sequence.
    traj_objects : list of objects of mdtraj or mdanalysis
    backend : str, default='mdtraj'
        The backend to be used for loading the trajectories. Currently only 'mdtraj' and 'mda' are supported.
    stride : int, default=1
        In case of using mda backend, the stride to be used for loading the trajectories. mdtraj already handled this.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        num_tokens: int = 1,
        backend: Optional[str] = "mdtraj",
        stride: int = 1,
    ) -> None:
        backend = backend or "none"
        if backend not in {"mdtraj", "mda", "none"}:
            raise ValueError("Unsupported backend '{}'. Choose from 'mdtraj', 'mda', or 'none'.".format(backend))

        self._dtype = dtype
        self.num_tokens = num_tokens
        self.backend = backend
        self.stride = stride

        self._mdtraj = None
        self._mdanalysis = None
        if backend == "mdtraj":
            self._mdtraj = self._require_mdtraj()
        elif backend == "mda":
            self._mdanalysis = self._require_mdanalysis()

    def _seq_trajs(self, data: Sequence[np.ndarray | torch.Tensor]) -> List[torch.Tensor]:
        """
        Convert input data to a list of PyTorch tensors.

        Parameters
        ----------
        data : array-like or list of array-like
            The input data to be converted.

        Returns
        -------
        list of torch.Tensor
            The processed data as a list of tensors.
        """

        data = copy(data)
        if not isinstance(data, list):
            data = [data]

        for i in range(len(data)):
            if not isinstance(data[i], torch.Tensor):
                data[i] = torch.tensor(data[i], dtype=self._dtype)
            else:
                data[i] = data[i].clone().detach().type(self._dtype)

        return data

    def _require_mdtraj(self):
        try:
            import mdtraj as _mdtraj  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise ImportError(
                "mdtraj is required for 'mdtraj' backend. Install it via `pip install mdtraj`."
            ) from exc
        return _mdtraj

    def _require_mdanalysis(self):
        try:
            import MDAnalysis as _mda  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise ImportError(
                "MDAnalysis is required for 'mda' backend. Install it via `pip install MDAnalysis`."
            ) from exc
        return _mda

    def _extract_ca_coords(self, traj_objects: Iterable) -> List[torch.Tensor]:
        """
        Extract the coordinates of the alpha carbons from the trajectories.
        Returns
        -------
        ca_coords : list of torch.Tensor
            The coordinates of the alpha carbons in the trajectories.
        """

        if self.backend == "none":
            raise RuntimeError("CA coordinate extraction requires 'mdtraj' or 'mda' backend.")

        ca_coords: List[torch.Tensor] = []
        if self.backend == "mdtraj":
            assert self._mdtraj is not None
            for traj in tqdm(traj_objects, desc="Extracting Ca coordinates (mdtraj)"):
                ca_indices = traj.top.select("name CA")
                coords = torch.from_numpy(traj.xyz[:, ca_indices]).to(self._dtype) * 10
                ca_coords.append(coords)
        else:
            assert self._mdanalysis is not None
            for traj in tqdm(traj_objects, desc="Extracting Ca coordinates (MDAnalysis)"):
                ca = traj.select_atoms("name CA")
                positions = []
                for _ in tqdm(traj.trajectory[:: self.stride], desc="Frames", leave=False):
                    positions.append(ca.positions.copy())
                coords = torch.from_numpy(np.asarray(positions)).to(self._dtype)
                ca_coords.append(coords)
        return ca_coords

    def create_time_lagged_dataset_flat(
        self,
        data: Sequence[np.ndarray | torch.Tensor],
        lag_time: int,
        ca_coords: Sequence[np.ndarray | torch.Tensor],
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], FlatFeatureSpec]:
        """Create a flattened time-lagged dataset from packed graph/coordinate tensors."""

        graph_features = self._seq_trajs(data)
        ca_coords = self._seq_trajs(ca_coords)

        assert len(graph_features) == len(ca_coords)
        if lag_time <= 0:
            raise ValueError("lag_time must be a positive integer")

        num_trajs = len(graph_features)

        packed_features: List[torch.Tensor] = []
        for i in range(num_trajs):
            flat_features = packing_features(
                graph_features=graph_features[i],
                ca_coords=ca_coords[i],
                num_tokens=self.num_tokens,
            )
            packed_features.append(flat_features)

        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(num_trajs):
            L_all = packed_features[i].shape[0]
            L_re = L_all - lag_time
            for j in range(L_re):
                dataset.append((packed_features[i][j, :], packed_features[i][j + lag_time, :]))
        spec = FlatFeatureSpec(num_tokens=self.num_tokens, hidden_dim=graph_features[0].shape[-1])
        return dataset, spec

    def create_time_lagged_dataset(
        self, data: Sequence[np.ndarray | torch.Tensor], lag_time: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create a time-lagged dataset.

        This dataset is used for VAMPnet/SRV training/validation.

        Parameters
        ----------
        data : list or ndarray or torch.Tensor
            The original trajectories.

        lag_time : int
            The lag_time used to create the dataset consisting of time-instant and time-lagged data.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has two elements: one is the instantaneous data frame, the other is the corresponding time-lagged data frame.

        """

        data = self._seq_trajs(data)
        if lag_time <= 0:
            raise ValueError("lag_time must be a positive integer")

        num_trajs = len(data)
        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lag_time
            for i in range(L_re):
                dataset.append((data[k][i, :], data[k][i + lag_time, :]))

        return dataset

    def create_spib_dataset(
        self,
        data_list: Sequence[np.ndarray],
        label_list: Sequence[np.ndarray],
        weight_list: Optional[Sequence[np.ndarray]],
        output_dim: int,
        lag_time: int = 1,
        subsampling_timestep: int = 1,
    ) -> "SPIBDataset":
        """
        Prepare data for SPIB training and validation

        Parameters
        ----------
        data_list : List of trajectory data
            The data which is wrapped into a dataset.
        label_list : List of corresponding labels
            Corresponding label data. Must be of the same length.
        weight_list: List of corresponding weights, optional, default=None
            Corresponding weight data. Must be of the same length.
        output_dim: int
            The total number of states in label_list.
        lag_time: int, default=1
            The lag time used to produce timeshifted blocks.
        subsampling_timestep: int, default=1
            The step size for subsampling.
        """

        if weight_list is None:
            dataset = SPIBDataset(data_list, label_list, None, lag_time=lag_time,
                                  subsampling_timestep=subsampling_timestep,
                                  output_dim=output_dim)

        else:
            dataset = SPIBDataset(data_list, label_list, weight_list, lag_time=lag_time,
                                  subsampling_timestep=subsampling_timestep,
                                  output_dim=output_dim)

        return dataset

    def create_spib_train_test_datasets(
        self,
        data_list: Sequence[np.ndarray | torch.Tensor],
        label_list: Sequence[np.ndarray | torch.Tensor],
        train_fraction: float,
        lag_time: int,
        output_dim: int,
        weight_list: Optional[Sequence[np.ndarray | torch.Tensor]] = None,
        subsampling_timestep: int = 1,
    ) -> Tuple["SPIBDataset", "SPIBDataset"]:
        """
        Convenience helper to split trajectories into train/test and construct SPIB datasets.

        Parameters
        ----------
        data_list : sequence of trajectory arrays or tensors
            Each element has shape (frames, ...), e.g. (frames, num_tokens, 4, hidden_dim).
        label_list : sequence of label arrays or tensors
            One label per frame for each trajectory.
        train_fraction : float
            Fraction of frames per trajectory used for training (0, 1].
        lag_time : int
            Lag time used when constructing time-lagged pairs.
        output_dim : int
            Number of discrete states (e.g. clusters).
        weight_list : optional sequence of weight arrays or tensors
            Per-frame weights aligned with data_list/label_list.
        subsampling_timestep : int, default=1
            Subsampling step for SPIBDataset.
        """

        if len(data_list) != len(label_list):
            raise ValueError(
                f"Length of data_list and label_list does not match "
                f"({len(data_list)} != {len(label_list)})"
            )
        if not (0.0 < train_fraction <= 1.0):
            raise ValueError("train_fraction must be in (0, 1].")
        if lag_time <= 0:
            raise ValueError("lag_time must be a positive integer.")

        def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        train_data_list_np: List[np.ndarray] = []
        train_label_list_np: List[np.ndarray] = []
        test_data_list_np: List[np.ndarray] = []
        test_label_list_np: List[np.ndarray] = []

        train_weight_list_np: Optional[List[np.ndarray]] = [] if weight_list is not None else None
        test_weight_list_np: Optional[List[np.ndarray]] = [] if weight_list is not None else None

        for idx, (data, labels) in enumerate(zip(data_list, label_list)):
            data_np = _to_numpy(data)
            labels_np = _to_numpy(labels)
            if data_np.shape[0] != labels_np.shape[0]:
                raise ValueError(
                    f"Trajectory {idx} has mismatched frames between data and labels "
                    f"({data_np.shape[0]} != {labels_np.shape[0]})."
                )

            n_frames = data_np.shape[0]
            n_train = int(n_frames * train_fraction)
            if n_train <= lag_time:
                # skip trajectories that are too short to form time-lagged pairs
                continue

            weights_np = None
            if weight_list is not None:
                weights_np = _to_numpy(weight_list[idx])
                if weights_np.shape[0] != n_frames:
                    raise ValueError(
                        f"Trajectory {idx} has mismatched frames between data and weights "
                        f"({n_frames} != {weights_np.shape[0]})."
                    )

            if n_frames - n_train <= lag_time:
                train_data_list_np.append(data_np)
                train_label_list_np.append(labels_np)
                if train_weight_list_np is not None and weights_np is not None:
                    train_weight_list_np.append(weights_np)
            else:
                train_data_list_np.append(data_np[:n_train])
                train_label_list_np.append(labels_np[:n_train])
                if train_weight_list_np is not None and weights_np is not None:
                    train_weight_list_np.append(weights_np[:n_train])

                test_data_list_np.append(data_np[n_train:])
                test_label_list_np.append(labels_np[n_train:])
                if test_weight_list_np is not None and weights_np is not None:
                    test_weight_list_np.append(weights_np[n_train:])

        if not train_data_list_np:
            raise ValueError("No trajectories are long enough to form time-lagged training pairs.")

        train_dataset = self.create_spib_dataset(
            data_list=train_data_list_np,
            label_list=train_label_list_np,
            weight_list=train_weight_list_np,
            lag_time=lag_time,
            subsampling_timestep=subsampling_timestep,
            output_dim=output_dim,
        )

        if test_data_list_np:
            test_dataset = self.create_spib_dataset(
                data_list=test_data_list_np,
                label_list=test_label_list_np,
                weight_list=test_weight_list_np,
                lag_time=lag_time,
                subsampling_timestep=subsampling_timestep,
                output_dim=output_dim,
            )
        else:
            test_dataset = train_dataset

        return train_dataset, test_dataset

    def load_dataset(self, data_path: str, mmap_mode: Optional[str] = "r") -> List[torch.Tensor]:
        """Load the dataset from the file.

        Parameters
        ----------
        data_path : str
            The path to the file.
            The type of data to be stored in the output file.
        mmap_mode: str, default = 'r'
            The mode to open the file. If None, the file will be opened in the default mode.

        Returns
        -------
        data : list
            The dataset.
        """
        assert mmap_mode in ["r", "r+", "w+", "c", None]

        files = os.listdir(data_path)
        files = [os.path.join(data_path, f) for f in files if f.endswith(".pt")]
        files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        data = []
        for file in tqdm(files):
            traj = torch.load(file, map_location="cpu", mmap=mmap_mode).to(self._dtype)
            data.append(traj)

        return data

    def load_dataset_folder(
        self, data_path: str, mmap_mode: Optional[str] = "r", sorting: bool = True
    ) -> List[torch.Tensor]:
        """Load the dataset from the file.

        Parameters
        ----------
        data_path : str
            The path to the file.
            The type of data to be stored in the output file.
        mmap_mode: str, default = 'r'
            The mode to open the file. If None, the file will be opened in the default mode.
        sorting: bool, default=True
            Sort the files in the folder

        Returns
        -------
        data : list
            The dataset.
        """
        assert mmap_mode in ["r", "r+", "w+", "c", None]

        files = []
        for root, _, filenames in os.walk(data_path):
            files.extend([os.path.join(root, f) for f in filenames if f.endswith(".pt")])

        if sorting:
            files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        data = []
        for file in tqdm(files):
            traj = torch.load(file, map_location="cpu", mmap=mmap_mode).to(self._dtype)
            data.append(traj)

        return data


class SPIBDataset(Dataset):
    """
    High-level container for time-lagged time-series data

    Parameters
    ----------
    data_list : List of trajectory data
        The data which is wrapped into a dataset.
    label_list : List of corresponding labels
        Corresponding label data. Must be of the same length.
    weight_list: List of corresponding weights, optional, default=None
        Corresponding weight data. Must be of the same length.
    lag_time: int, default=1
        The lag time used to produce timeshifted blocks.
    subsampling_timestep: int, default=1
        The step size for subsampling.
    output_dim: int, optional
        The total number of states in label_list.
    device: torch device, default=torch.device("cpu")
        The device on which the torch modules are executed.
    """

    def __init__(
        self,
        data_list: Sequence[np.ndarray],
        label_list: Sequence[np.ndarray],
        weight_list: Optional[Sequence[np.ndarray]] = None,
        lag_time: int = 1,
        subsampling_timestep: int = 1,
        output_dim: Optional[int] = None,
    ) -> None:
        assert len(data_list) == len(label_list), \
            f"Length of data_list and label_list does not match ({len(data_list)} != {len(label_list)})"

        self.lag_time = lag_time
        self.subsampling_timestep = subsampling_timestep
        self.traj_num = len(data_list)

        if weight_list is None:
            # Set weights as ones
            weight_list = [np.ones_like(label_list[i]) for i in range(len(label_list))]

        data_init_list = []
        for i in range(len(data_list)):
            data_init_list.append(
                self._data_init(self.lag_time, self.subsampling_timestep,
                                data_list[i], label_list[i], weight_list[i])
            )

        # Concatenate and convert to tensors
        self.data_weights = torch.from_numpy(
            np.concatenate([item[3] for item in data_init_list], axis=0)
        ).float()

        self.past_data = torch.from_numpy(
            np.concatenate([item[0] for item in data_init_list], axis=0)
        ).float()

        self.future_data = torch.from_numpy(
            np.concatenate([item[1] for item in data_init_list], axis=0)
        ).float()

        label_data = torch.from_numpy(
            np.concatenate([item[2] for item in data_init_list], axis=0)
        ).long()

        # Record the lengths of trajectories
        self.split_lengths = [len(item[2]) for item in data_init_list]

        # One-hot encode labels
        if output_dim is None:
            self.future_labels = F.one_hot(label_data)
        else:
            self.future_labels = F.one_hot(label_data, num_classes=output_dim)

    def __len__(self) -> int:
        return len(self.past_data)

    def __getitem__(self, idx: int):
        return self.past_data[idx], self.future_labels[idx], self.data_weights[idx]

    def update_labels(self, new_labels):
        self.future_labels = new_labels

    def _data_init(
        self,
        lag_time: int,
        subsampling_timestep: int,
        traj_data: np.ndarray,
        traj_label: np.ndarray,
        traj_weights: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert len(traj_data) == len(traj_label), \
            f"Length of traj_data and traj_label does not match ({len(traj_data)} != {len(traj_label)})"

        # Subsample and time-shift data
        past_data = traj_data[:(len(traj_data) - lag_time):subsampling_timestep]
        future_data = traj_data[lag_time::subsampling_timestep]
        label = traj_label[lag_time::subsampling_timestep]

        if traj_weights is not None:
            assert len(traj_data) == len(traj_weights)
            weights = traj_weights[: (len(traj_weights) - lag_time) : subsampling_timestep]
        else:
            weights = np.ones_like(label, dtype=np.float32)

        return past_data, future_data, label, weights
