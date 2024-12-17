import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import copy
from torch.utils.data import Dataset
import MDAnalysis as mda
import mdtraj as md


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

    def __init__(self, dtype=torch.float32, num_tokens=1, backend='mdtraj', stride=1):
        assert backend in ['mdtraj', 'mda'], f"Backend {backend} is not supported."

        self._dtype = dtype
        self.num_tokens = num_tokens
        self.backend = backend
        self.stride = stride

    def _seq_trajs(self, data):
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

    def _extract_ca_coords(self, traj_objects):
        """
        Extract the coordinates of the alpha carbons from the trajectories.
        Returns
        -------
        ca_coords : list of torch.Tensor
            The coordinates of the alpha carbons in the trajectories.
        """

        ca_coords = []
        if self.backend == 'mdtraj':
            for traj in tqdm(traj_objects, desc="Extracting Ca coordinates (mdtraj)"):
                ca_indices = traj.top.select('name CA')
                coords = torch.from_numpy(traj.xyz[:, ca_indices]).to(self._dtype) * 10  # Convert to Angstrom
                ca_coords.append(coords)
        elif self.backend == 'mda':
            for traj in tqdm(traj_objects, desc="Extracting Ca coordinates (MDAnalysis)"):
                ca = traj.select_atoms('name CA')
                ca_positions = []
                for ts in tqdm(traj.trajectory[::self.stride], desc="Frames", leave=False):
                    ca_positions.append(ca.positions.copy())
                coords = torch.from_numpy(np.array(ca_positions)).to(self._dtype)
                ca_coords.append(coords)
        return ca_coords

    def _extract_ca_pairwise_dist(self,ca_coords):
        """
        Extract the pairwise distances between the alpha carbons from the trajectories.
        Returns
        -------
        ca_pairwise_dist : list of torch.Tensor
            The pairwise distances between the alpha carbons in the trajectories.
        """
        # check if the ca_coords is a list or not
        if not isinstance(ca_coords, list):
            ca_coords = [ca_coords]

        ca_pw_dist = []
        for coords in tqdm(ca_coords, desc="Extracting Ca pairwise distances"):
            n_frames, n_atoms, _ = coords.shape
            i_upper, j_upper = np.triu_indices(n_atoms, k=1)
            coords_i = coords[:, i_upper, :]
            coords_j = coords[:, j_upper, :]
            diff = coords_i - coords_j
            sq_dist = torch.sum(diff ** 2, dim=-1)
            pairwise_distances = np.sqrt(sq_dist)
            ca_pw_dist.append((pairwise_distances).float())

        print(f"Extracted pairwise distances for {len(ca_pw_dist)} trajectories.")
        print(f"Shape of the pairwise distances: {ca_pw_dist[0].shape}")

        return ca_pw_dist

    def create_time_lagged_dataset_flat(self, data, lag_time, ca_coords, ca_pairwise_dist):
        # testing the new function to create time-lagged dataset in a flat format including new features

        graph_features = self._seq_trajs(data)
        ca_coords = self._seq_trajs(ca_coords)
        ca_pairwise_dist = self._seq_trajs(ca_pairwise_dist)

        assert len(graph_features) == len(ca_coords) == len(ca_pairwise_dist)

        num_trajs = len(graph_features)
        from .util import packing_features

        packed_features = []
        for i in range(num_trajs):
            flat_features = packing_features(graph_features=graph_features[i],
                                             ca_coords=ca_coords[i],
                                             global_features=ca_pairwise_dist[i],
                                             num_tokens=self.num_tokens)
            packed_features.append(flat_features)

        dataset = []
        for i in range(num_trajs):
            L_all = packed_features[i].shape[0]
            L_re = L_all - lag_time
            for j in range(L_re):
                dataset.append((packed_features[i][j, :], packed_features[i][j + lag_time, :]))
        return dataset

    def create_time_lagged_dataset(self, data, lag_time):
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

        num_trajs = len(data)
        dataset = []
        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lag_time
            for i in range(L_re):
                dataset.append((data[k][i, :], data[k][i + lag_time, :]))

        return dataset

    def create_spib_dataset(self, data_list, label_list, weight_list, output_dim, lag_time=1, subsampling_timestep=1):
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

    def create_boundary_dataset(self, data, ina, inb):
        """
        Create a dataset for the boundary condition.

        This dataset is used for VCN/SVCN prediction.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.

        ina : list or ndarray
            The initial condition.

        inb : list or ndarray
            The final condition.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has three elements: one is the instantaneous data frame, the other two are the boundary conditions

        """

        assert len(data) == len(ina) == len(inb)

        data = self._seq_trajs(data)
        ina = self._seq_trajs(ina)
        inb = self._seq_trajs(inb)

        num_trajs = len(data)
        dataset = []
        for k in range(num_trajs):
            L_all = data[k].shape[0]
            for i in range(L_all):
                dataset.append((data[k][i, :], ina[k][i], inb[k][i]))

        return dataset

    def create_vcn_dataset(self, data, ina, inb, lag_time):
        """
        Create a dataset for training VCNs.

        This dataset is used for VCN training/validation.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.
        ina : list or ndarray
            The initial condition.
        inb : list or ndarray
            The final condition.
        lag_time : int
            The lag time used to create the dataset.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has 6 elements:
            instantaneous and time-lagged data frames,
            instantaneous and time-lagged initial conditions,
            instantaneous and time-lagged final conditions.

        """
        assert len(data) == len(ina) == len(inb)

        data = self._seq_trajs(data)
        ina = self._seq_trajs(ina)
        inb = self._seq_trajs(inb)

        num_trajs = len(data)
        dataset = []

        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lag_time

            assert ina[k].shape == inb[k].shape == (L_all, 1)

            data_traj = data[k]

            ina_traj = ina[k].bool()
            inb_traj = inb[k].bool()

            for i in range(L_re):
                dataset.append(
                    (
                        data_traj[i],
                        data_traj[i + lag_time],
                        ina_traj[i],
                        ina_traj[i + lag_time],
                        inb_traj[i],
                        inb_traj[i + lag_time],
                    )
                )

        return dataset

    def create_svcn_dataset(self, data, ina, inb, lag_time):
        """
        Create a dataset for training SVCNs.

        This dataset is used for SVCN training/validation.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.
        ina : list or ndarray
            The initial condition.
        inb : list or ndarray
            The final condition.
        lag_time : int
            The lag time used to create the dataset.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has 12 elements:
            instantaneous and time-lagged data frames,
            instantaneous and time-lagged initial conditions,
            instantaneous and time-lagged final conditions,
            and 6 other entries.

        """
        from .util import (
            forward_stop_torch as forward_stop,
            backward_stop_torch as backward_stop,
            count_transition_paths_windows_torch as count_transition_paths_windows,
        )

        assert len(data) == len(ina) == len(inb)

        data = self._seq_trajs(data)
        ina = self._seq_trajs(ina)
        inb = self._seq_trajs(inb)

        num_trajs = len(data)
        dataset = []

        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lag_time

            assert ina[k].shape == inb[k].shape == (L_all, 1)

            data_traj = data[k]

            ina_traj = ina[k].bool()
            inb_traj = inb[k].bool()
            ind_traj = torch.logical_not(torch.logical_or(ina_traj, inb_traj))

            r = backward_stop(ind_traj[:, 0])
            s = forward_stop(ind_traj[:, 0])
            ab = count_transition_paths_windows(
                ind_traj[:, 0], ina_traj[:, 0], inb_traj[:, 0], lag_time
            )[:, None]
            ba = count_transition_paths_windows(
                ind_traj[:, 0], inb_traj[:, 0], ina_traj[:, 0], lag_time
            )[:, None]

            for i in range(L_re):
                dataset.append(
                    (
                        data_traj[i],
                        data_traj[i + lag_time],
                        ina_traj[i],
                        ina_traj[i + lag_time],
                        inb_traj[i],
                        inb_traj[i + lag_time],
                        ind_traj[min(i + lag_time, s[i])],
                        ina_traj[min(i + lag_time, s[i])],
                        inb_traj[min(i + lag_time, s[i])],
                        ina_traj[max(i, r[i + lag_time])],
                        inb_traj[max(i, r[i + lag_time])],
                        ab[i] + ba[i],
                    )
                )

        return dataset

    def create_committor_dataset(self, data, ina, inb, drop_first=0, drop_last=0):
        """
        Create a dataset for committor regression.

        This dataset is also useful for testing predicted committors.

        Note that the target will be nan if the previous or next stopping time is not within the trajectory.
        To avoid this, set `drop_first` and `drop_last` to a sufficiently large number.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.
        ina : list or ndarray
            The initial condition.
        inb : list or ndarray
            The final condition.
        drop_first : int, default=0
            Number of frames to drop from the start of each trajectory.
        drop_last : int, default=0
            Number of frames to drop from the last of each trajectory.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            Each tuple has 4 elements: data frame, initial condition, final condition, target.

        """
        assert len(data) == len(ina) == len(inb)

        data = self._seq_trajs(data)
        ina = self._seq_trajs(ina)
        inb = self._seq_trajs(inb)

        num_trajs = len(data)
        dataset = []

        for k in range(num_trajs):
            L_all = data[k].shape[0]

            assert ina[k].shape == inb[k].shape == (L_all, 1)

            data_traj = data[k]

            from .util import backward_stop_torch, forward_stop_torch

            ina_traj = ina[k].bool()
            inb_traj = inb[k].bool()
            assert not torch.any(torch.logical_and(ina_traj, inb_traj))

            ind_traj = torch.logical_not(torch.logical_or(ina_traj, inb_traj))
            r = backward_stop_torch(ind_traj[:, 0])
            s = forward_stop_torch(ind_traj[:, 0])

            target_r = torch.full((L_all, 1), np.nan, dtype=torch.float32)
            target_s = torch.full((L_all, 1), np.nan, dtype=torch.float32)

            target_r[r >= 0] = inb_traj[r[r >= 0]].float()
            target_s[s < L_all] = inb_traj[s[s < L_all]].float()

            for i in range(drop_first, L_all - drop_last):
                dataset.append((data_traj[i], ina_traj[i], inb_traj[i], target_r[i]))
                dataset.append((data_traj[i], ina_traj[i], inb_traj[i], target_s[i]))

        return dataset

    def create_time_lagged_stop_dataset(self, data, ina, inb, lag_time):
        """Create a time-lagged dataset for the boundary condition with stopped time.

        Parameters
        ----------
        data : list or ndarray
            The original trajectories.

        ina : list or ndarray
            The initial condition.

        inb : list or ndarray
            The final condition.

        lag_time : int
            The lag_time used to create the dataset consisting of time-instant and time-lagged data.

        Returns
        -------
        dataset : list
            List of tuples: the length of the list represents the number of data.
            each tuple has six elements: first two represents the instantaneous data frame
            and the corresponding time-lagged data frame.
        """
        assert len(data) == len(ina) == len(inb)
        from .util import forward_stop_torch as forward_stop

        data = self._seq_trajs(data)
        ina = self._seq_trajs(ina)
        inb = self._seq_trajs(inb)

        num_trajs = len(data)
        dataset = []

        for k in range(num_trajs):
            L_all = data[k].shape[0]
            L_re = L_all - lag_time

            ina_traj = ina[k].int()
            inb_traj = inb[k].int()

            ind_traj = 1 - ina_traj - inb_traj
            ind_traj = ind_traj.squeeze()

            t0 = torch.arange(L_re)
            t1 = t0 + lag_time
            ts = torch.minimum(t1, forward_stop(ind_traj)[t0])

            data_traj = data[k][t0]
            data_traj_lag = data[k][t1]
            ind_traj_out = ind_traj[t0].unsqueeze(1)
            ind_traj_lag = ind_traj[t1].unsqueeze(1)
            ind_traj_stop = ind_traj[ts].unsqueeze(1)

            assert len(data_traj) == len(data_traj_lag) == len(ind_traj_lag) == len(ind_traj_stop)

            for i in range(L_re):
                dataset.append((data_traj[i], data_traj_lag[i], ind_traj_stop[i]))

        return dataset

    def load_dataset(self, data_path, mmap_mode="r"):
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

    def load_dataset_folder(self, data_path, mmap_mode="r", sorting=True):
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


class SPIBDataset(torch.utils.data.Dataset):
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

    def __init__(self, data_list, label_list, weight_list=None, lag_time=1, subsampling_timestep=1, output_dim=None):
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

    def __len__(self):
        return len(self.past_data)

    def __getitem__(self, idx):
        return self.past_data[idx], self.future_labels[idx], self.data_weights[idx]

    def update_labels(self, new_labels):
        self.future_labels = new_labels

    def _data_init(self, lag_time, subsampling_timestep, traj_data, traj_label, traj_weights):
        assert len(traj_data) == len(traj_label), \
            f"Length of traj_data and traj_label does not match ({len(traj_data)} != {len(traj_label)})"

        # Subsample and time-shift data
        past_data = traj_data[:(len(traj_data) - lag_time):subsampling_timestep]
        future_data = traj_data[lag_time::subsampling_timestep]
        label = traj_label[lag_time::subsampling_timestep]

        if traj_weights is not None:
            assert len(traj_data) == len(traj_weights)
            weights = traj_weights[:(len(traj_weights) - lag_time):subsampling_timestep]
        else:
            weights = None

        return past_data, future_data, label, weights
