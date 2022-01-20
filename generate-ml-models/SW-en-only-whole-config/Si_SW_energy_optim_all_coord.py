import sys

import torch
from torch import nn

from load_training_set import load_kliff_training_set

torch.set_default_dtype(torch.float)

# =============================================================================
# Torch StillingerWeber Layer
# =============================================================================
@torch.jit.script
def calc_sw2(A, B, p, q, sigma, cutoff, rij):
    """
    Two body energy for SW
    """
    if rij < cutoff:
        sig_r = sigma / rij
        E2 = A * (B * sig_r ** p - sig_r ** q) * torch.exp(sigma / (rij - cutoff))
    else:
        E2 = torch.tensor(0.0, dtype=torch.float32)
    return E2


@torch.jit.script
def calc_sw3(
    lam, cos_beta0, gamma_ij, gamma_ik, cutoff_ij, cutoff_ik, cutoff_jk, rij, rik, rjk
):
    """
    Three body energy for SW
    """
    cos_beta_ikj = (rij ** 2 + rik ** 2 - rjk ** 2) / (2 * rij * rik)
    cos_diff = cos_beta_ikj - cos_beta0
    exp_ij_ik = torch.exp(gamma_ij / (rij - cutoff_ij) + gamma_ik / (rik - cutoff_ik))
    E3 = lam * exp_ij_ik * cos_diff ** 2
    return E3


@torch.jit.script
def energy(
    particle_contributing: torch.IntTensor,
    coords: torch.Tensor,
    num_neighbors: torch.IntTensor,
    neighbor_list: torch.IntTensor,
    A: torch.Tensor,
    B: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    sigma: torch.Tensor,
    gamma: torch.Tensor,
    cutoff: torch.Tensor,
    lam: torch.Tensor,
    cos_beta0: torch.Tensor,
):
    """
    Calculate Energy for a given list of coordiates, assuming first coordinate
    to be of query atom i, and remaining in the list to be neighbours.
    """
    energy_conf = torch.tensor(0.0, dtype=torch.float32)

    # Store index to be able to traverse the neighbor list in sequence
    neigh_list_cursor = 0

    num_atoms = particle_contributing.shape[0]

    for atom_i in range(num_atoms):

        if particle_contributing[atom_i] != 1:
            continue

        # Coordinates of atom i
        xyz_i = coords[0, atom_i * 3 : (atom_i + 1) * 3]
        num_neigh_i = int(num_neighbors[atom_i])

        neigh_i_begin = neigh_list_cursor
        neigh_i_end = neigh_i_begin + num_neigh_i

        nli = neighbor_list[0, torch.arange(neigh_i_begin, neigh_i_end)]

        neigh_list_cursor = neigh_i_end

        for atom_j in range(num_neigh_i):
            xyz_j = coords[0, (nli[atom_j]) * 3 : ((nli[atom_j]) + 1) * 3]
            rij = xyz_j - xyz_i
            norm_rij = (rij[0] ** 2 + rij[1] ** 2 + rij[2] ** 2) ** 0.5
            E2 = calc_sw2(A, B, p, q, sigma, cutoff, norm_rij)
            energy_conf = energy_conf + 0.5 * E2
            for atom_k in range(atom_j + 1, num_neigh_i):
                xyz_k = coords[0, (nli[atom_k]) * 3 : ((nli[atom_k]) + 1) * 3]
                rik = xyz_k - xyz_i
                norm_rik = (rik[0] ** 2 + rik[1] ** 2 + rik[2] ** 2) ** 0.5
                rjk = xyz_k - xyz_j
                norm_rjk = (rjk[0] ** 2 + rjk[1] ** 2 + rjk[2] ** 2) ** 0.5
                E3 = calc_sw3(
                    lam,
                    cos_beta0,
                    gamma,
                    gamma,
                    cutoff,
                    cutoff,
                    cutoff,
                    norm_rij,
                    norm_rik,
                    norm_rjk,
                )
                energy_conf = energy_conf + E3

    return energy_conf


# ==============================================================================
class StillingerWeberLayer(nn.Module):
    """
    Stillinger-Weber single species layer for Si atom for use in PyTorch model
    Before optimization, the parameter to be optimized need to be set using
    set_optim function. Forward method returns energy of the configuration
     and force array.
    """

    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(15.2848479197914, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(0.6022245584, dtype=torch.float32))
        self.p = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.q = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.sigma = nn.Parameter(torch.tensor(2.0951, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(2.51412, dtype=torch.float32))
        self.cutoff = nn.Parameter(torch.tensor(3.77118, dtype=torch.float32))
        self.lam = nn.Parameter(torch.tensor(45.5322, dtype=torch.float32))
        self.cos_beta0 = nn.Parameter(
            torch.tensor(-0.333333333333333, dtype=torch.float32)
        )

    def set_optim(self, to_optim):
        """
        Makes initialized parameters as Pytorch Parameter, for torch model to
        optimize.
        #TODO Make feature compatible with Kliff parameter class, with functionality
        to set bounds
        """
        for member in to_optim:
            setattr(self, member, nn.Parameter(getattr(self, member)))

    def forward(
        self,
        particle_contributing: torch.Tensor,
        coords: torch.Tensor,
        num_neighbors: torch.Tensor,
        neighbor_list: torch.Tensor,
    ):  # desc:Tuple[torch.Tensor, torch.Tensor]):
        total_conf_energy = energy(
            particle_contributing,
            coords,
            num_neighbors,
            neighbor_list,
            self.A,
            self.B,
            self.p,
            self.q,
            self.sigma,
            self.gamma,
            self.cutoff,
            self.lam,
            self.cos_beta0,
        )
        return total_conf_energy

    @torch.jit.export
    def forces(self, energy: torch.Tensor, coords: torch.Tensor):
        forces = torch.autograd.grad([energy], [coords], create_graph=True)[0]
        if forces is not None:
            return forces
        else:
            print("ERROR Forces are NONE")
            return torch.tensor(-1)


# ==========================================================================================================
# ==========================================================================================================
# Example implementation

if __name__ == "__main__":

    # Stillinger weber layer and set A as parameter to be optimized
    SWL = StillingerWeberLayer()
    SWL.set_optim(["A", "B", "gamma"])
    model = torch.jit.script(SWL)

    export_to_torchscript = True
    export_to_onnx = False

    if export_to_torchscript:
        model.save("SW_en_only.pt")

    if export_to_onnx:

        filename = sys.argv[1]

        # Load inputs for all configurations from training set
        desc, energies, forces = load_kliff_training_set(
            filename, cutoff=3.77118, torch_tensor_output=True
        )

        # Pull out inputs for the first configuration
        particle_contributing, coords, num_neighbors, neighbor_list, *_ = desc[0][:]

        model_inputs = particle_contributing, coords, num_neighbors, neighbor_list

        model(*model_inputs)

        torch.onnx.export(
            model,
            args=model_inputs,
            f="SW_en.onnx",
            opset_version=14,
            input_names=[
                "particle_contributing",
                "coords",
                "num_neighbors",
                "neighbor_list",
            ],
        )
