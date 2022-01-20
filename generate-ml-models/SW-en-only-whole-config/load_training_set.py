import torch
from kliff.dataset import Dataset
from kliff.neighbor import NeighborList
import numpy as np


def load_kliff_configuration(kliff_config, cutoff, torch_tensor_output=False):
    energy = kliff_config.energy
    forces = np.array(kliff_config.forces, dtype=np.float32)

    # Generate neighbour list command and get padded indices
    # to be folded back into 0:num_atoms format
    kliff_neighbor_list = NeighborList(kliff_config, cutoff, padding_need_neigh=False)

    num_contributing_atoms = kliff_config.get_num_atoms()
    atom_indices = np.arange(0, num_contributing_atoms)
    particle_contributing = np.ones(num_contributing_atoms, dtype=np.int32).reshape(
        (1, -1)
    )

    xyz_tensor = []
    num_neighbors = []
    neighbor_list = []

    # Create a list of all indices, padded or otherwise for later use
    neighbor_padding_image = kliff_neighbor_list.padding_image
    num_padding_atoms = len(kliff_neighbor_list.coords) - num_contributing_atoms
    if num_padding_atoms > 0:
        atom_indices = np.append(atom_indices, neighbor_padding_image)
        particle_contributing = np.append(
            particle_contributing, np.zeros(num_padding_atoms, dtype=np.int32)
        )

    # collect neighbours for each atom in kliff_config
    for i in range(num_contributing_atoms):
        atom_i_neighbor_indices_unwrapped, _, _ = kliff_neighbor_list.get_neigh(i)
        neighbor_list.extend(atom_i_neighbor_indices_unwrapped)
        num_neighbors.append(len(atom_i_neighbor_indices_unwrapped))

    xyz_tensor = kliff_neighbor_list.coords.astype(np.float32)
    xyz_tensor = xyz_tensor.reshape((1, -1))
    neighbor_list = np.array(neighbor_list, dtype=np.int32).reshape((1, -1))

    if torch_tensor_output:
        energy = torch.tensor(energy, dtype=torch.float32)
        forces = torch.from_numpy(forces)
        particle_contributing = torch.from_numpy(particle_contributing)
        xyz_tensor = torch.from_numpy(xyz_tensor)
        num_neighbors = torch.tensor(num_neighbors, dtype=torch.int32)
        neighbor_list = torch.from_numpy(neighbor_list)
        neighbor_padding_image = torch.from_numpy(neighbor_padding_image)

    config_info = [
        particle_contributing,
        xyz_tensor,
        num_neighbors,
        neighbor_list,
        neighbor_padding_image,
    ]

    return config_info, energy, forces


def load_kliff_training_set(path, cutoff, torch_tensor_output=False):
    """
    Takes in the path of the dataset and cutoff/ distance of influence
    And returns a List containing following structure
    num_configs x num_atoms_in_config x 3 x num_neighbours

      conf1              conf2                                            conf3          ...
    [[atom1,atom2,...],[[[neigh_coords], [neig_idx], [symbols]],[] ...],        ,                ,...]

    Therefore Descriptor[i]         = Configuration
              Descriptor[i][j]      = Atom in configuration
              Descriptor[i][j][0:3] = [neigh_coords, neigh_idx, neigh_symbol]
    Depends on Kliff Dataset object and NeighbourList object to do the heavylifting
    :return: List
    """
    # Load the dataset from path
    data_obj = Dataset(path)
    configurations = []
    energies = []
    forces = []
    # For each xyz file in config path folder
    for conf in data_obj.get_configs():
        config_info, config_energy, config_forces = load_kliff_configuration(
            kliff_config=conf, cutoff=cutoff, torch_tensor_output=torch_tensor_output
        )
        configurations.append(config_info)
        energies.append(config_energy)
        forces.append(config_forces)

    return configurations, energies, forces
