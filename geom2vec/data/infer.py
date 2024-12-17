import os
from collections import Counter
from typing import List

import MDAnalysis as mda
import mdtraj as md
import numpy as np
import torch
from torch_scatter import scatter
from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mass_mapping = {
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "H": 1.008,
    "S": 32.06,
    "F": 18.998,
    "Cl": 35.453,
}
atomic_mapping = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16, "F": 9, "Cl": 17}


def create_model(
        model_type,
        checkpoint_path=None,
        cutoff=7.5,
        hidden_channels=128,
        num_layers=6,
        num_rbf=64,
        device="cuda",
):
    assert model_type in ["et", "vis", "tn"]  # only support ET, ViSNet, and TensorNet

    model = None

    if model_type == "et":
        from geom2vec.representation_models.torchmd.main_model import (
            create_model, get_args)

        args = get_args(
            hidden_channels,
            num_layers,
            num_rbf,
            num_heads=8,
            cutoff=cutoff,
            rep_model="et",
        )
        model = create_model(args)
    elif model_type == "vis":
        from geom2vec.representation_models.visnet import ViSNet

        model = ViSNet(
            hidden_channels=hidden_channels,
            cutoff=cutoff,
            num_rbf=num_rbf,
            vecnorm_type="max_min",
            trainable_vecnorm=True,
        )
    elif model_type == "tn":
        from geom2vec.representation_models.torchmd.main_model import (
            create_model, get_args)

        args = get_args(
            hidden_channels,
            num_layers,
            num_rbf,
            num_heads=8,
            cutoff=cutoff,
            rep_model="tensornet",
        )
        model = create_model(args)

    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print(f"Model loaded from {checkpoint_path}")
            model.eval()
        except Exception as e:
            print(f"Error loading the model from {checkpoint_path}")
            #print the error message
            print(e)

    else:
        print("Model created from scratch.")

    return model.to(device)


def infer_traj(
        model: torch.nn.Module,
        hidden_channels: int,
        data: List[np.ndarray],
        atomic_numbers: np.ndarray,
        device: torch.device,
        saving_path: str,
        batch_size: int = 32,
        sum_or_mean: str = "sum",
        cg_mapping: np.ndarray = None,
        atom_mask: np.ndarray = None,
        cg_mask: np.ndarray = None,
        # torch_or_numpy: str = "torch",
        file_name_list: List[str] = None,
):
    """
    This function is used to infer the trajectories of data.

    Input:
    - model: torch.nn.Module
        The model to be used for inference.
    - data: list of ndarray
        The trajectories of data in xyz.
    - atomic_numbers: ndarray
        The atomic numbers of the atoms.
    - device: torch.device
        The device on which the torch modules are executed.
    - saving_path: str
        The path to save the inferred data.
    - cg_mapping: ndarray, default = None
        The coarse-grained mapping to be used.
    - batch_size: int, default = 32
        The batch size to be used for inference.
    - sum_or_mean: str, default = 'sum'
        The type of aggregation to be used for the atoms.
    - atom_mask: ndarray, default = None
        The mask of the atoms for the inference to save the data.
    - cg_mask: ndarray, default = None
        The mask of the coarse-grained beads for the inference to save the data.
    - torch_or_numpy: str, default = 'torch'
        The type of data to be stored in the output file.
    - file_name_list: list of str, default = None
        The list of file names to be saved.

    """
    model.eval()
    model.to(device=device)

    num_atoms = len(atomic_numbers)

    cg_map = None
    if cg_mapping is not None:
        assert sum(cg_mapping) == num_atoms
        # The sum of the coarse-grained mapping should be equal to the number of atoms
        cg_mapping = torch.from_numpy(cg_mapping).to(device)
        cg_map = torch.repeat_interleave(
            torch.arange(cg_mapping.shape[0], device=device), cg_mapping, dim=0
        )

    z = torch.from_numpy(atomic_numbers).to(device)
    with torch.no_grad():
        for i in range(len(data)):
            traj = data[i]

            out_list = []
            traj = torch.from_numpy(traj).float().to(device)
            for pos_batch in tqdm(torch.split(traj, batch_size, dim=0)):
                n_samples, n_atoms, _ = pos_batch.shape
                z_batch = z.expand(n_samples, -1).reshape(-1).to(device)
                batch_batch = (
                    torch.arange(n_samples).unsqueeze(1).expand(-1, n_atoms).reshape(-1)
                ).to(device)
                x_rep, v_rep, _ = model(
                    z=z_batch,
                    pos=pos_batch.reshape(-1, 3).contiguous().to(device),
                    batch=batch_batch,
                )
                # Move the data to CPU and append to the output list
                x_rep = x_rep.reshape(-1, num_atoms, 1, hidden_channels)
                v_rep = v_rep.reshape(-1, num_atoms, 3, hidden_channels)
                atom_rep = torch.cat([x_rep, v_rep], dim=-2)

                if atom_mask is not None:
                    atom_rep = atom_rep[:, atom_mask, :]

                if cg_mapping is not None:
                    if sum_or_mean == "mean":
                        cg_rep = scatter(atom_rep, cg_map, dim=1, reduce="mean")
                    else:
                        cg_rep = scatter(atom_rep, cg_map, dim=1, reduce="add")
                    if cg_mask is not None:
                        cg_rep = cg_rep[:, cg_mask, :, :]

                    cg_rep.detach().cpu()
                    out_list.append(cg_rep)
                    continue

                atom_rep = atom_rep.detach().cpu()
                if sum_or_mean == "mean":
                    out_list.append(atom_rep.mean(dim=1))
                else:
                    out_list.append(atom_rep.sum(dim=1))
                torch.cuda.empty_cache()

            # concatenate the output batches
            traj_rep = torch.cat(out_list, dim=0)
            # if torch_or_numpy == "numpy":
            #     traj_rep = traj_rep.clone().detach().cpu().numpy()
            #     saving_file_name = (
            #         file_name_list[i] if file_name_list is not None else f"traj_{i}"
            #     )
            #     saving_filepath = os.path.join(saving_path, f"{saving_file_name}.npz")
            #     np.savez(saving_filepath, traj_rep)
            #     print(f"Trajectory {i} has been saved to {saving_path} using numpy.")
            # elif torch_or_numpy == "torch":
            saving_file_name = (
                file_name_list[i] if file_name_list is not None else f"traj_{i}"
            )
            saving_filepath = os.path.join(saving_path, f"{saving_file_name}.pt")
            torch.save(traj_rep, saving_filepath)
            print(f"Trajectory {i} has been saved to {saving_path} using torch.")
            # else:
            #     print(
            #         "Invalid option for torch_or_numpy. Please choose either 'torch' or 'numpy'."
            #     )

    return None


def count_segments(numbers):
    segments = []
    current_segment = [numbers[0]]

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1]:
            current_segment.append(numbers[i])
        else:
            segments.append(current_segment)
            current_segment = [numbers[i]]
    segments.append(current_segment)  # Add the last segment

    # Count elements in each segment
    segment_counts = [Counter(segment) for segment in segments]
    segment_counts_array = np.array(
        [list(segment_count.values()) for segment_count in segment_counts]
    )
    segment_counts_array = np.concatenate(segment_counts_array, axis=0)
    return segment_counts_array


def extract_mda_info(protein, stride=1, selection=None):
    # input: MDA Universe object with selection, output: positions, atomic_numbers, segment_counts
    protein_residues = protein.select_atoms("prop mass > 1.5 ")  # remove hydrogens
    if selection is not None:
        protein_residues = protein.select_atoms(selection)
    # Get all residues in the protein selection
    # protein_residues = protein.residues
    atomic_masses = protein_residues.masses
    atomic_masses = np.round(atomic_masses, 3)

    atomic_types = [
        list(mass_mapping.keys())[list(mass_mapping.values()).index(mass)]
        for mass in atomic_masses
    ]
    atomic_numbers = [atomic_mapping[atom] for atom in atomic_types]

    positions = []
    for ts in protein.trajectory:
        positions.append(protein_residues.positions.copy())

    positions = np.array(positions)[::stride]

    segment_counts = count_segments(protein_residues.resids)

    return positions, np.array(atomic_numbers), np.array(segment_counts)


def extract_mda_info_folder(
        folder, top_file, stride=1, selection=None, file_postfix=".dcd", sorting=True
):
    r"""
    do the extraction for all the files in the folder

    Args:
    - folder: str
        The folder containing the .dcd files
    - top_file: str
        The topology file
    - stride: int, default = 1
        The stride to use when extracting the data
    - selection: str, default = None
        The selection to use when extracting the data in MDAnalysis. If None, all atoms are selected except hydrogens.
    - file_postfix: str, default = '.dcd'
        The postfix of the files to be extracted

    Returns:
    - position_list: list
        The list of positions for each trajectory
    - atomic_numbers: ndarray
        The atomic numbers of the atoms
    - segment_counts: ndarray
        The number counts for each segment
    - dcd_files: list
        The list of dcd files in the folder (important for the order of the trajectories)
    """

    # Get all the .dcd files in the folder
    dcd_files = [f for f in os.listdir(folder) if f.endswith(file_postfix)]
    if sorting:
        # Sort the files
        dcd_files.sort()

    position_list = []
    mda_objects = []
    file_paths = []
    for traj in dcd_files:
        print(f"Processing {traj}")
        u = mda.Universe(top_file, os.path.join(folder, traj))
        file_paths.append(os.path.join(folder, traj))
        positions, atomic_numbers, segment_counts = extract_mda_info(
            u, stride=stride, selection=selection
        )
        position_list.append(positions)
        mda_objects.append(u)

    return position_list, atomic_numbers, segment_counts, file_paths, mda_objects


def extract_mdtraj_info(md_traj_object, exclude_hydrogens=True):
    '''
    Extracts the positions, atomic numbers, and segment counts from a mdtraj object
    Args:
        md_traj_object: mdtraj trajectory object after selection
        exclude_hydrogens: whether to exclude hydrogens from the data

    Returns:
        positions: positions of the atoms x,y,z
        atomic_numbers: atomic numbers of the atoms
        segment_counts: CG mapping using the residue indices

    '''
    atomic_numbers = [atom.element.atomic_number for atom in md_traj_object.top.atoms]
    atomic_numbers = np.array(atomic_numbers)
    residue_indices = [atom.residue.index for atom in md_traj_object.top.atoms]
    residue_indices = np.array(residue_indices)
    positions = md_traj_object.xyz * 10  # Convert to angstroms

    hydrogen_mask = np.array(atomic_numbers) == 1

    if exclude_hydrogens:
        positions = positions[:, ~hydrogen_mask]
        atomic_numbers = atomic_numbers[~hydrogen_mask]
        segment_counts = count_segments(residue_indices[~hydrogen_mask])
    else:
        segment_counts = count_segments(residue_indices)

    return positions, atomic_numbers, segment_counts


def extract_mdtraj_info_folder(folder, top_file, stride=1,
                               selection='protein', file_postfix=".dcd",
                               num_trajs=None, exclude_hydrogens=True):
    r"""
    do the extraction for all the files in the folder

    Args:
    - folder: str
        The folder containing the .dcd files
    - top_file: str
        The topology file
    - stride: int, default = 1
        The stride to use when extracting the data
    - selection: str, default = None
        The selection to use when extracting the data in MDAnalysis. If None, all atoms are selected except hydrogens.
    - file_postfix: str, default = '.dcd'
        The postfix of the files to be extracted

    Returns:
    - position_list: list
        The list of positions for each trajectory
    - atomic_numbers: ndarray
        The atomic numbers of the atoms
    - segment_counts: ndarray
        The number counts for each segment
    - dcd_files: list
        The list of dcd files in the folder (important for the order of the trajectories)
    """

    # Get all the .dcd files in the folder
    dcd_files = [f for f in os.listdir(folder) if f.endswith(file_postfix)]
    # Sort the files
    dcd_files.sort()

    if num_trajs is not None:
        dcd_files = dcd_files[:num_trajs]

    position_list = []
    file_paths = []
    for traj in dcd_files:
        print(f"Processing {traj}")
        try:
            mdtraj_object = md.load(os.path.join(folder, traj), top=top_file, stride=stride)
            file_paths.append(os.path.join(folder, traj))
        except Exception as e:
            print(f"Error loading file {traj}")
            print(e)
            continue

        if selection == 'protein':
            mdtraj_object = mdtraj_object.atom_slice(mdtraj_object.top.select('protein'))
        elif selection == 'backbone':
            mdtraj_object = mdtraj_object.atom_slice(mdtraj_object.top.select('backbone'))
        elif selection == 'heavy':
            mdtraj_object = mdtraj_object.atom_slice(mdtraj_object.top.select('not water and not hydrogen'))
        elif selection == 'all':
            pass
        else:
            raise ValueError('Invalid selection type')

        positions, atomic_numbers, segment_counts = extract_mdtraj_info(mdtraj_object,
                                                                        exclude_hydrogens=exclude_hydrogens)
        position_list.append(positions)

    return position_list, atomic_numbers, segment_counts, file_paths, mdtraj_object
