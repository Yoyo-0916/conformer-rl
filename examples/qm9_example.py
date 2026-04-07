import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from rdkit.Chem import TorsionFingerprints

from conformer_rl import utils
from conformer_rl.agents import PPOAgent
from conformer_rl.config import Config
from conformer_rl.datasets import QM9XYZDataset, qm9_data_to_mol_config, qm9_data_to_rdkit_mol
from conformer_rl.environments import Task
from conformer_rl.models import RTGN


logging.basicConfig(level=logging.INFO)


def resolve_dataset_root(dataset_root: str) -> Path:
    root = Path(dataset_root)
    if (root / "dsgdb9nsd.xyz").is_dir() or any(root.glob("dsgdb9nsd_*.xyz")):
        return root

    parent_root = Path("..") / root
    if (parent_root / "dsgdb9nsd.xyz").is_dir() or any(parent_root.glob("dsgdb9nsd_*.xyz")):
        return parent_root

    return root


def select_qm9_sample(dataset: QM9XYZDataset, qm9_index: int = None):
    for idx in range(len(dataset)):
        data = dataset[idx]
        if qm9_index is not None and data["qm9_index"] != qm9_index:
            continue

        try:
            mol = qm9_data_to_rdkit_mol(data)
        except ValueError:
            if qm9_index is not None:
                raise
            continue

        nonring, _ = TorsionFingerprints.CalculateTorsionLists(mol)
        if nonring:
            return idx, data, len(nonring)

        if qm9_index is not None:
            raise ValueError("QM9 index {} has no non-ring torsions.".format(qm9_index))

    raise ValueError("No QM9 sample with non-ring torsions found in the scanned range.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train conformer-rl from a real QM9 conformer.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--scan-limit", type=int, default=1000)
    parser.add_argument("--qm9-index", type=int, default=None)
    parser.add_argument("--num-conformers", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--skip-normalizers", action="store_true")
    args = parser.parse_args()

    utils.set_one_thread()

    dataset_root = resolve_dataset_root(args.dataset_root)
    logging.info("Using QM9 dataset root: %s", dataset_root.resolve())
    dataset = QM9XYZDataset(root=str(dataset_root), return_type="dict", limit=args.scan_limit)
    idx, data, num_torsions = select_qm9_sample(dataset, qm9_index=args.qm9_index)
    logging.info(
        "Using QM9 dataset index %s, qm9_index=%s, smiles=%s, torsions=%s",
        idx,
        data["qm9_index"],
        data["smiles"],
        num_torsions,
    )

    mol_config = qm9_data_to_mol_config(
        data,
        num_conformers=args.num_conformers,
        calc_normalizers=not args.skip_normalizers,
    )

    config = Config(gpu_id=args.gpu_id)
    config.tag = "qm9_example"
    config.network = RTGN(6, 128, edge_dim=6, node_dim=5).to(config.device)
    config.train_env = Task(
        "GibbsScorePruningEnv-v0",
        concurrency=args.num_envs > 1,
        num_envs=args.num_envs,
        seed=np.random.randint(0, 1e5),
        mol_config=mol_config,
    )
    config.eval_env = Task(
        "GibbsScorePruningEnv-v0",
        num_envs=1,
        seed=np.random.randint(0, 1e5),
        mol_config=mol_config,
    )

    config.max_steps = args.max_steps
    config.rollout_length = min(20, args.num_conformers)
    config.mini_batch_size = min(20, config.rollout_length * args.num_envs)
    config.save_interval = 0
    config.eval_interval = 0
    config.eval_episodes = 1
    config.use_tensorboard = False
    config.exp_tag = ""
    config.eval = False
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-5, eps=1e-5)

    agent = PPOAgent(config)
    agent.run_steps()
