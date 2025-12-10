import numpy as np
import torch
from ase.data import covalent_radii
from ase.io import read
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class DenaliDataset(InMemoryDataset):
    def __init__(self, root, bond_threshold=0.5, transform=None, pre_transform=None, pre_filter=None):
        super(DenaliDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.bond_threshold = bond_threshold

    @property
    def raw_file_names(self):
        return ['combined.xyz']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def get_edge_index_from_ase(self, atoms, thresholds=0.5):
        """
        Generate edge indices using ASE by calculating distances and comparing
        to covalent radii to determine bonding.

        Parameters:
        - atoms (ase.Atoms): An ASE Atoms object representing the molecule.

        Returns:
        - edge_index_tensor (torch.LongTensor): Edge index tensor for PyG.
        """
        # Calculate distance matrix
        distances = atoms.get_all_distances(mic=True)

        # Determine potential bonding using covalent radii
        radii = covalent_radii[atoms.numbers]
        radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]

        # A simple heuristic to determine bonding: distance less than sum of covalent radii
        is_bonded = distances < radii_matrix + thresholds
        np.fill_diagonal(is_bonded, False)  # Remove self-bonding

        edge_indices = np.vstack(np.where(is_bonded))

        # Convert to PyTorch tensor
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long)

        return edge_index_tensor

    def process(self):
        # Read the combined XYZ file using ASE
        molecules = read(self.raw_paths[0], index=':')

        data_list = []
        for molecule in tqdm(molecules):
            # Get the atomic numbers and positions
            atomic_numbers = molecule.get_atomic_numbers()
            positions = molecule.get_positions()
            edge_index = self.get_edge_index_from_ase(molecule, thresholds=0.5)

            # Create a Data object
            data = Data(pos=torch.from_numpy(positions).float(),
                        z=torch.from_numpy(atomic_numbers).long(),
                        edge_index=edge_index.long())
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])