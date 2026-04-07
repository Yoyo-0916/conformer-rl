import torch

from conformer_rl.datasets import QM9XYZDataset, create_qm9_dataloader


def test_qm9_xyz_dataset_reads_sample_file():
    dataset = QM9XYZDataset(root="dataset", target="mu", return_type="dict", limit=1)

    data = dataset[0]

    assert len(dataset) == 1
    assert data["molecule_id"] == "gdb"
    assert data["qm9_index"] == 1
    assert data["x"].shape == (5, 5)
    assert data["z"].tolist() == [6, 1, 1, 1, 1]
    assert data["pos"].shape == (5, 3)
    assert data["edge_index"].shape == (2, 20)
    assert data["edge_attr"].shape == (20, 1)
    assert data["y"].shape == (1, 1)
    assert torch.isclose(data["y"][0, 0], torch.tensor(0.0))
    assert data["target_names"] == ["mu"]


def test_create_qm9_dataloader_batches_graphs():
    loader = create_qm9_dataloader(
        root="dataset",
        batch_size=2,
        shuffle=False,
        target=["homo", "lumo"],
        return_type="dict",
        limit=2,
    )

    batch = next(iter(loader))

    assert len(batch) == 2
    assert batch[0]["y"].shape == (1, 2)
    assert batch[0]["pos"].shape[1] == 3
