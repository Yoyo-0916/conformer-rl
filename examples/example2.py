import numpy as np
import torch
import argparse

from conformer_rl import utils
from conformer_rl.agents import PPOAgent
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGN

from conformer_rl.molecule_generation.generate_lignin import generate_lignin
from conformer_rl.molecule_generation.generate_molecule_config import config_from_rdkit

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=0, help='CUDA GPU index to use when available')
    args = parser.parse_args()

    utils.set_one_thread()

    # configure molecule
    mol = generate_lignin(3)
    mol_config = config_from_rdkit(mol, num_conformers=200, calc_normalizers=True, save_file='lignin')

    # create agent config and set environment
    config = Config(gpu_id=args.gpu_id)
    config.tag = 'example2'
    config.train_env = Task('GibbsScorePruningEnv-v0', concurrency=True, num_envs=10, mol_config=mol_config)

    # Neural Network
    config.network = RTGN(6, 128, edge_dim=6, node_dim=5).to(config.device)

    # Logging Parameters
    config.save_interval = 20000
    config.data_dir = 'data'
    config.use_tensorboard = True

    # Set up evaluation
    eval_mol = generate_lignin(4)
    eval_mol_config = config_from_rdkit(mol, num_conformers=200, calc_normalizers=True, save_file='lignin_eval')
    config.eval_env = Task('GibbsScorePruningEnv-v0', num_envs=1, mol_config=eval_mol_config)
    # config.eval_interval = 20000
    config.eval_interval = 5000
    config.eval_episodes = 10

    # Batch Hyperparameters
    config.rollout_length = 20
    config.recurrence = 5
    config.optimization_epochs = 4
    config.max_steps = 80001
    config.mini_batch_size = 50

    # Training Hyperparameters
    lr = 5e-6 * np.sqrt(10)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_lambda = 0.95
    config.entropy_weight = 0.001
    config.value_loss_weight = 0.25
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2
    config.exp_tag = f'opt_{config.optimization_epochs}_eval_{config.eval_interval}'
    config.eval = False
    
    if config.eval:
        agent = PPOAgent(config)
        model_path = 'data/models/example2_23-03-2026_07:41:56opt_10/60000.model' # specify path to model to load
        agent.load(model_path)
        print(f"Evaluate {agent.unique_tag}")
        agent.evaluate()
    else:
        agent = PPOAgent(config)
        print(f"Train {agent.unique_tag}")
        agent.run_steps()
        print(f"Done {agent.unique_tag}")