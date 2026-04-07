"""
QM9 dataset helpers.
"""
from pathlib import Path
import re
from typing import Any, Callable, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:  # pragma: no cover - compatibility with older PyG versions.
    try:
        from torch_geometric.data import Data
        from torch_geometric.data import DataLoader as PyGDataLoader
    except ImportError:  # pragma: no cover - allows dict loading without PyG.
        Data = None
        PyGDataLoader = None


QM9_PROPERTIES = (
    "A",
    "B",
    "C",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
)

_ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
}


class QM9XYZDataset(Dataset):
    """Lazy PyTorch Geometric dataset for QM9 ``.xyz`` files.

    Parameters
    ----------
    root : str
        QM9 dataset directory. If ``root/dsgdb9nsd.xyz`` exists, files are loaded
        from that subdirectory to avoid double-counting extracted files.
    target : str or list of str, optional
        QM9 property/properties to expose as ``data.y``. If omitted, all QM9
        scalar properties are returned.
    edge_strategy : str
        ``"complete"`` creates a directed complete graph with distance
        ``edge_attr``; ``"none"`` leaves edges empty.
    node_features : str
        ``"one_hot"`` uses H/C/N/O/F one-hot features; ``"atomic_number"``
        uses a single atomic-number feature.
    return_type : str
        ``"auto"`` returns PyG ``Data`` when torch-geometric is installed and
        dicts otherwise; ``"pyg"`` requires torch-geometric; ``"dict"`` always
        returns dictionaries.
    transform : callable, optional
        Optional transform called on each parsed ``Data`` object.
    limit : int, optional
        Optional cap on the number of files, useful for smoke tests.
    """

    atom_types = ("H", "C", "N", "O", "F")

    def __init__(
        self,
        root: str = "dataset",
        target: Optional[Union[str, Sequence[str]]] = None,
        edge_strategy: str = "complete",
        node_features: str = "one_hot",
        return_type: str = "auto",
        transform: Optional[Callable[[Any], Any]] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.data_dir = self._resolve_data_dir(self.root)
        self.edge_strategy = edge_strategy
        self.node_features = node_features
        self.return_type = return_type
        self.transform = transform
        self.target_names = self._normalize_targets(target)

        if self.edge_strategy not in ("complete", "none"):
            raise ValueError("edge_strategy must be either 'complete' or 'none'.")
        if self.node_features not in ("one_hot", "atomic_number"):
            raise ValueError("node_features must be either 'one_hot' or 'atomic_number'.")
        if self.return_type not in ("auto", "pyg", "dict"):
            raise ValueError("return_type must be 'auto', 'pyg', or 'dict'.")
        if self.return_type == "pyg" and Data is None:
            raise ImportError("torch-geometric is required when return_type='pyg'.")
        if self.return_type == "auto":
            self.return_type = "pyg" if Data is not None else "dict"

        self.paths = sorted(self.data_dir.glob("dsgdb9nsd_*.xyz"), key=self._path_sort_key)
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError("No QM9 .xyz files were found in {}.".format(str(self.data_dir)))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Data:
        data = self._parse_xyz(self.paths[index])
        if self.transform is not None:
            data = self.transform(data)
        return data

    @staticmethod
    def _resolve_data_dir(root: Path) -> Path:
        nested_xyz_dir = root / "dsgdb9nsd.xyz"
        if nested_xyz_dir.is_dir():
            return nested_xyz_dir
        return root

    @staticmethod
    def _path_sort_key(path: Path) -> int:
        match = re.search(r"(\d+)$", path.stem)
        if match is None:
            return 0
        return int(match.group(1))

    @staticmethod
    def _normalize_targets(target: Optional[Union[str, Sequence[str]]]) -> List[str]:
        if target is None:
            return list(QM9_PROPERTIES)
        if isinstance(target, str):
            targets = [target]
        else:
            targets = list(target)

        unknown = [name for name in targets if name not in QM9_PROPERTIES]
        if unknown:
            raise ValueError("Unknown QM9 target(s): {}.".format(", ".join(unknown)))
        return targets

    def _parse_xyz(self, path: Path) -> Any:
        lines = path.read_text().splitlines()
        num_atoms = int(lines[0].strip())

        metadata = lines[1].split()
        molecule_id = metadata[0]
        qm9_index = int(metadata[1])
        property_values = torch.tensor([float(value) for value in metadata[2:]], dtype=torch.float)

        symbols = []
        positions = []
        charges = []
        for line in lines[2 : 2 + num_atoms]:
            parts = line.split()
            symbols.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            charges.append(float(parts[4]))

        z = torch.tensor([_ATOMIC_NUMBERS[symbol] for symbol in symbols], dtype=torch.long)
        pos = torch.tensor(positions, dtype=torch.float)
        x = self._build_node_features(symbols, z)
        edge_index, edge_attr = self._build_edges(pos)

        property_dict = {
            name: property_values[i].item()
            for i, name in enumerate(QM9_PROPERTIES)
        }
        target_indices = [QM9_PROPERTIES.index(name) for name in self.target_names]

        payload = {
            "x": x,
            "z": z,
            "pos": pos,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": property_values[target_indices].view(1, -1),
            "properties": property_values.view(1, -1),
            "charges": torch.tensor(charges, dtype=torch.float).view(-1, 1),
            "molecule_id": molecule_id,
            "qm9_index": qm9_index,
            "property_names": list(QM9_PROPERTIES),
            "target_names": list(self.target_names),
            "property_dict": property_dict,
        }

        if len(lines) > 2 + num_atoms:
            payload["frequencies"] = torch.tensor(
                [float(value) for value in lines[2 + num_atoms].split()],
                dtype=torch.float,
            )
        if len(lines) > 3 + num_atoms:
            payload["smiles"] = lines[3 + num_atoms].split()
        if len(lines) > 4 + num_atoms:
            payload["inchi"] = lines[4 + num_atoms].split()

        if self.return_type == "dict":
            return payload

        data = Data(**{
            key: value
            for key, value in payload.items()
            if torch.is_tensor(value)
        })
        data.molecule_id = molecule_id
        data.qm9_index = qm9_index
        data.property_names = payload["property_names"]
        data.target_names = payload["target_names"]
        data.property_dict = property_dict
        if "frequencies" in payload:
            data.frequencies = payload["frequencies"]
        if "smiles" in payload:
            data.smiles = payload["smiles"]
        if "inchi" in payload:
            data.inchi = payload["inchi"]
        return data

    def _build_node_features(self, symbols: Sequence[str], z: torch.Tensor) -> torch.Tensor:
        if self.node_features == "atomic_number":
            return z.float().view(-1, 1)

        rows = []
        for symbol in symbols:
            row = [0.0] * len(self.atom_types)
            row[self.atom_types.index(symbol)] = 1.0
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float)

    def _build_edges(self, pos: torch.Tensor):
        if self.edge_strategy == "none" or pos.size(0) <= 1:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 1), dtype=torch.float),
            )

        sources = []
        destinations = []
        distances = []
        for i in range(pos.size(0)):
            for j in range(pos.size(0)):
                if i == j:
                    continue
                sources.append(i)
                destinations.append(j)
                distances.append(torch.norm(pos[i] - pos[j]).item())

        edge_index = torch.tensor([sources, destinations], dtype=torch.long)
        edge_attr = torch.tensor(distances, dtype=torch.float).view(-1, 1)
        return edge_index, edge_attr


def create_qm9_dataloader(
    root: str = "dataset",
    batch_size: int = 32,
    shuffle: bool = True,
    target: Optional[Union[str, Sequence[str]]] = None,
    edge_strategy: str = "complete",
    node_features: str = "one_hot",
    return_type: str = "auto",
    transform: Optional[Callable[[Any], Any]] = None,
    limit: Optional[int] = None,
    **loader_kwargs
):
    """Create a dataloader for local QM9 ``.xyz`` files.

    When torch-geometric is installed, this returns a PyG ``DataLoader`` by
    default. Otherwise it returns a PyTorch ``DataLoader`` that yields lists of
    dict samples.
    """
    dataset = QM9XYZDataset(
        root=root,
        target=target,
        edge_strategy=edge_strategy,
        node_features=node_features,
        return_type=return_type,
        transform=transform,
        limit=limit,
    )
    if dataset.return_type == "pyg":
        return PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)

    loader_kwargs.setdefault("collate_fn", lambda batch: batch)
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
