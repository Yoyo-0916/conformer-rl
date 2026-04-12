"""
Chemistry Utilities
===================

Chemistry and molecule utility functions.
"""
import numpy as np
import bisect
import matplotlib.pyplot as plt
from rdkit.Chem import TorsionFingerprints
import rdkit.Chem.AllChem as Chem
from rdkit.Chem.Draw import rdMolDraw2D

from pathlib import Path
from typing import Tuple, List, Union
import logging


def get_conformer_energies(mol: Chem.Mol) -> List[float]:
    """Returns a list of energies for each conformer in `mol`.
    """
    energies = []
    Chem.MMFFSanitizeMolecule(mol)
    mmff_props = Chem.MMFFGetMoleculeProperties(mol)
    for conf in mol.GetConformers():
        ff = Chem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf.GetId())
        energy = ff.CalcEnergy()
        energies.append(energy)
    
    return np.asarray(energies, dtype=float)

def get_conformer_energy(mol: Chem.Mol, confId: int = None) -> float:
    """Returns the energy of the conformer with `confId` in `mol`.
    """
    if confId is None:
        confId = mol.GetNumConformers() - 1
    Chem.MMFFSanitizeMolecule(mol)
    mmff_props = Chem.MMFFGetMoleculeProperties(mol)
    ff = Chem.MMFFGetMoleculeForceField(mol, mmff_props, confId=confId)
    energy = ff.CalcEnergy()

    return energy

def prune_last_conformer(mol: Chem.Mol, tfd_thresh: float, energies: List[float]) -> Tuple[Chem.Mol, List[float]]:
    """Prunes the last conformer of the molecule.

    If no conformers in `mol` have a TFD (Torsional Fingerprint Deviation) with the last conformer of less than `tfd_thresh`,
    the last conformer is kept. Otherwise, the lowest energy conformer with TFD less than `tfd_thresh` is kept and all other conformers
    are discarded.

    Parameters
    ----------
    mol : RDKit Mol
        The molecule to be pruned. The conformers in the molecule should be ordered by ascending energy.
    tfd_thresh : float
        The minimum threshold for TFD between conformers.
    energies : list of float
        A list of all the energies of the conformers in `mol`.

    Returns
    -------
    mol : RDKit Mol
        The updated molecule after pruning, with conformers sorted by ascending energy.
    energies : list of float
        A list of all the energies of the conformers in `mol` after pruning and sorting by ascending energy.
    """
    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol, energies

    idx = bisect.bisect(energies[:-1], energies[-1])
    tfd = TorsionFingerprints.GetTFDBetweenConformers(mol, range(0, mol.GetNumConformers() - 1), [mol.GetNumConformers() - 1], useWeights=False)
    tfd = np.array(tfd)

    # if lower energy conformer is within threshold, drop new conf
    if not np.all(tfd[:idx] >= tfd_thresh):
        energies = energies[:-1]
        mol.RemoveConformer(mol.GetNumConformers() - 1)
        return mol, energies
    else:
        keep = list(range(0, idx))
        keep.append(mol.GetNumConformers() - 1)
        keep += [x for x in range(idx, mol.GetNumConformers() - 1) if tfd[x] >= tfd_thresh]

        new = Chem.Mol(mol)
        new.RemoveAllConformers()
        for i in keep:
            conf = mol.GetConformer(i)
            new.AddConformer(conf, assignId=True)

        return new, [energies[i] for i in keep]

def prune_conformers(mol: Chem.Mol, tfd_thresh: float) -> Chem.Mol:
    """Prunes all the conformers in the molecule.

    Removes conformers that have a TFD (torsional fingerprint deviation) lower than
    `tfd_thresh` with other conformers. Lowest energy conformers are kept.

    Parameters
    ----------
    mol : RDKit Mol
        The molecule to be pruned.
    tfd_thresh : float
        The minimum threshold for TFD between conformers.

    Returns
    -------
    mol : RDKit Mol
        The updated molecule after pruning.
    """
    if tfd_thresh < 0 or mol.GetNumConformers() <= 1:
        return mol

    energies = get_conformer_energies(mol)
    tfd = tfd_matrix(mol)
    sort = np.argsort(energies)  # sort by increasing energy
    keep = []  # always keep lowest-energy conformer
    discard = []

    for i in sort:
        this_tfd = tfd[i][np.asarray(keep, dtype=int)]
        # discard conformers within the tfd threshold
        if np.all(this_tfd >= tfd_thresh):
            keep.append(i)
        else:
            discard.append(i)

    # create a new molecule to hold the chosen conformers
    # this ensures proper conformer IDs and energy-based ordering
    new = Chem.Mol(mol)
    new.RemoveAllConformers()
    for i in keep:
        conf = mol.GetConformer(int(i))
        new.AddConformer(conf, assignId=True)

    return new

def tfd_matrix(mol: Chem.Mol) -> np.array:
    """Calculates the TFD matrix for all conformers in a molecule.
    """
    tfd = TorsionFingerprints.GetTFDMatrix(mol, useWeights=False)
    n = int(np.sqrt(len(tfd)*2))+1
    idx = np.tril_indices(n, k=-1, m=n)
    matrix = np.zeros((n,n))
    matrix[idx] = tfd
    matrix += np.transpose(matrix)
    return matrix

def calculate_normalizers(mol: Chem.Mol, num_confs: int = 200, pruning_thresh: float = 0.05) -> Tuple[float, float]:
    """Calculates the :math:`E_0` and :math:`Z_0` normalizing constants for a molecule used in the TorsionNet [1]_ paper.

    Parameters
    ----------
    mol : RDKit Mol
        The molecule of interest.
    num_confs : int
        The number of conformers to generate when calculating the constants. Should equal
        the number of steps for each episode of the environment containing this molecule.
    pruning_thresh : float
        TFD threshold for pruning the conformers of `mol`.

    References
    ----------
    .. [1] `TorsionNet paper <https://arxiv.org/abs/2006.07078>`_
    """
    Chem.MMFFSanitizeMolecule(mol)
    confslist = Chem.EmbedMultipleConfs(mol, numConfs = num_confs, useRandomCoords=True)
    if (len(confslist) < 1):
        raise Exception('Unable to embed molecule with conformer using rdkit')
    Chem.MMFFOptimizeMoleculeConfs(mol, nonBondedThresh=10.)
    mol = prune_conformers(mol, pruning_thresh)
    energys = get_conformer_energies(mol)
    E0 = energys.min()
    Z0 = np.sum(np.exp(-(energys - E0)))

    mol.RemoveAllConformers()

    return E0, Z0


def save_molecule_png(
    mol: Chem.Mol,
    output_path: Union[str, Path],
    size: Tuple[int, int] = (500, 400),
    conf_id: int = 0,
    add_atom_indices: Union[bool, None] = None,
    mode: str = "2d",
) -> Path:
    """Save an RDKit molecule conformer as a PNG image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if add_atom_indices is None:
        add_atom_indices = mode == "2d"

    if mode == "2d":
        return _save_molecule_png_2d(
            mol,
            output_path,
            size=size,
            conf_id=conf_id,
            add_atom_indices=add_atom_indices,
        )
    if mode == "3d":
        return _save_molecule_png_3d(
            mol,
            output_path,
            size=size,
            conf_id=conf_id,
            add_atom_indices=add_atom_indices,
        )
    raise ValueError("mode must be either '2d' or '3d'.")


def _save_molecule_png_2d(
    mol: Chem.Mol,
    output_path: Path,
    size: Tuple[int, int],
    conf_id: int,
    add_atom_indices: bool,
) -> Path:
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    draw_options = drawer.drawOptions()
    draw_options.addAtomIndices = add_atom_indices
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, Chem.Mol(mol), confId=conf_id)
    drawer.FinishDrawing()
    output_path.write_bytes(drawer.GetDrawingText())
    return output_path


def _save_molecule_png_3d(
    mol: Chem.Mol,
    output_path: Path,
    size: Tuple[int, int],
    conf_id: int,
    add_atom_indices: bool,
) -> Path:
    conf = mol.GetConformer(conf_id)
    positions = np.array(conf.GetPositions(), dtype=float)
    positions = _orient_positions_for_display(positions)
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    colors = {
        1: "#d9d9d9",
        6: "#4d4d4d",
        7: "#1f77b4",
        8: "#d62728",
        9: "#2ca02c",
    }
    sizes = {
        1: 80,
        6: 180,
        7: 200,
        8: 200,
        9: 220,
    }

    fig = plt.figure(figsize=(size[0] / 100.0, size[1] / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("persp")
    ax.set_facecolor("white")

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        xyz = positions[[begin, end]]
        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color="#606060",
            linewidth=3.0,
            zorder=1,
        )

    for atom_index, (xyz, atomic_number) in enumerate(zip(positions, atomic_numbers)):
        ax.scatter(
            xyz[0],
            xyz[1],
            xyz[2],
            color=colors.get(atomic_number, "#9467bd"),
            s=sizes.get(atomic_number, 160),
            edgecolors="black",
            linewidths=0.6,
            depthshade=True,
            zorder=2,
        )
        if add_atom_indices:
            ax.text(xyz[0], xyz[1], xyz[2], str(atom_index), fontsize=8, zorder=3)

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(np.max(maxs - mins) / 2.0, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=32)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, transparent=False)
    plt.close(fig)
    return output_path


def _orient_positions_for_display(positions: np.ndarray) -> np.ndarray:
    """Rotate coordinates to a stable presentation using PCA axes."""
    centered = positions - positions.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return centered

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vh.T
    return rotated
