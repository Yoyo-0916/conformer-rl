"""
Env_logger
==========
"""

import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from conformer_rl.utils import mkdir
from conformer_rl.utils import tfd_matrix
from typing import Any

class EnvLogger:
    """Used by the agent for logging metrics produced by the environment, for example, observations, rewards, renders, etc.
    Supports saving data to pickle and saving molecules as .mol files.

    Parameters
    ----------
    tag : str
        Unique tag for identifying the logging session.
    dir : str
        Path to root directory for where logging results should be saved.


    Attributes
    ----------
    step_data : dict
        Used for storing data for every step of a single episode. 
        Maps from strings to lists, where each index of the list corresponds
        to the data for that corresponding step within the episode.
    episode_data : dict 
        Used for storing information for a single episode. Used to store the self.step_data
        for that episode and metadata global to the entire episode.
    cache : dict 
        Used for storing data across several episodes. Maps from strings to lists, where each
        index of the list corresponds to an episode.
    """
    def __init__(self, tag: str, dir: str = "data"):
        self.dir = dir
        mkdir(dir)
        self.tag = tag
        self.step_data = {}
        self.episode_data = {}
        self.cache = {}

    def clear_data(self) -> None:
        """Resets the logger.
        """
        self.cache = {}
        self.episode_data = {}
        self.step_data = {}

    def clear_episode(self) -> None:
        """Clears episode and step data.
        """
        self.step_data = {}
        self.episode_data = {}

    def log_step_item(self, key: str, val: Any) -> None:
        """Logs a single key value pair for current step.

        If an existing key is found, the value is appended
        to the list associated with that key.

        Parameters
        ----------
        key : str
            the key for the data to be added.
        val : Any
            the value of the data to be added.
        """
        if key in self.step_data:
            self.step_data[key].append(val)
        else:
            self.step_data[key] = [val]

    def log_step(self, step_data: dict) -> None:
        """Logs each key-value pair for current step.

        If an existing key is found, the value is appended
        to the list associated with that key.

        Parameters
        ----------
        step_data : dict
            Contains key-value pairs to be logged.
        """
        for key, val in step_data.items():
            self.log_step_item(key, val)

    def log_episode_item(self, key: str, value: Any) -> None:
        """Logs a single key-value pair to the per-episode data.
        """
        self.episode_data[key] = value

    def log_episode(self, episode_data: dict) -> None:
        """Logs each key-value pair to the per-episode data.

        Also adds `step_data` to the per-episode data with corresponding key
        'step_data'. Existing keys will be overwritten.

        Parameters
        ----------
        episode_data : dict
            Contains key-value pairs to be logged.
        """
        self.episode_data.update(episode_data)
        self.episode_data["step_data"] = self.step_data

    def save_episode(self, subdir: str, save_pickle: bool = True, save_molecules: bool = False, save_cache: bool = False) -> None:
        """Saves current episode_data with options for dumping to pickle file,
        saving data to a cache dict, and saving the rdkit molecules as .mol files.
        Clears the current episode and step data.

        Parameters
        ----------
        subdir : str
            The directory for episode data to be saved (relative to self.dir)
        save_pickle : bool
            If True, dumps episode_data as a .pickle file.
        save_molecules : bool
            If True, and 'molecule' key exists in step_data, dumps each molecule generated 
            throughout the episode as a .mol file uniquely named by the step number.
        save_cache : bool
            If True, episode data is cached to self.data.
        """
        path = self.dir + '/' +  'env_data' + '/' + self.tag + '/' + subdir
        mkdir(path)
        filename = path + '/' +  'data.pickle'

        if save_pickle:
            outfile = open(filename, 'w+b')
            pickle.dump(self.episode_data, outfile)
            outfile.close()

        if save_molecules and 'mol' in self.episode_data:
            mol = self.episode_data['mol']
            for i in range(mol.GetNumConformers()):
                Chem.MolToMolFile(mol, filename=path + '/' +  f'step_{i}.mol', confId=i)

        if save_cache:
            self._add_to_cache(self.episode_data)

        self.clear_episode()
        
    def save_tfd_summary(self, eval_subdir: str, history_filename: str = 'tfd_total_history.png', summary_filename: str = 'tfd_summary.txt') -> dict:
        """Computes mean/std of TFD totals across all saved episodes in an evaluation directory,
        saves a per-evaluation summary .txt file, and updates a history line plot.

        Parameters
        ----------
        eval_subdir : str
            Evaluation subdirectory relative to ``env_data/<tag>``. For example,
            ``agent_step_20000``.
        history_filename : str
            Filename for the trend plot saved under ``env_data/<tag>``.
        summary_filename : str
            Filename for the summary text file saved in the evaluation directory.

        Returns
        -------
        dict
            Dict containing ``step``, ``tfd_total_mean``, and ``tfd_total_std``.
        """
        eval_path = Path(self.dir) / 'env_data' / self.tag / eval_subdir
        pickle_paths = sorted(path for path in eval_path.glob('ep_*/data.pickle') if path.is_file())
        if not pickle_paths:
            raise ValueError(f'No evaluation pickle files were found in {str(eval_path)!r}.')

        tfd_totals = []
        for pickle_path in pickle_paths:
            infile = open(pickle_path, 'rb')
            data = pickle.load(infile)
            infile.close()

            if 'mol' not in data:
                raise ValueError(f"Saved episode {str(pickle_path)!r} does not contain 'mol' data needed for TFD calculation.")
            tfd_totals.append(float(np.sum(tfd_matrix(data['mol']))))

        summary = {
            'step': self._extract_step_from_subdir(eval_subdir),
            'tfd_total_mean': float(np.mean(tfd_totals)),
            'tfd_total_std': float(np.std(tfd_totals)),
            'num_episodes': len(tfd_totals)
        }

        self._write_tfd_summary(eval_path / summary_filename, summary)
        self._save_tfd_history_plot(Path(self.dir) / 'env_data' / self.tag / history_filename, summary_filename)

        return summary

    def save_tfd_history_plot_v2(self, history_filename: str = 'tfd_total_history_v2.png', summary_filename: str = 'tfd_summary.txt') -> None:
        """Saves a second TFD history plot using a line-plus-band style.

        This method does not modify the existing ``tfd_total_history.png`` behavior.
        It reuses the saved ``tfd_summary.txt`` files under ``env_data/<tag>/agent_step_*``
        and renders the mean as a line with a shaded band for plus/minus one standard deviation.

        Parameters
        ----------
        history_filename : str
            Filename for the new trend plot saved under ``env_data/<tag>``.
        summary_filename : str
            Filename of the summary text files stored in each evaluation directory.
        """
        self._save_tfd_history_plot_v2(Path(self.dir) / 'env_data' / self.tag / history_filename, summary_filename)

    def _add_to_cache(self, data:dict) -> None:
        """Logs each key-value pair in data to self.cache.
        If an existing key is found, the value is appended
        to the list associated with that key.

        Parameters
        ----------
        data : dict
            contains key-value pairs to be logged.
        """
        for key, val in data.items():    
            if key in self.cache:
                self.cache[key].append(val)
            else:
                self.cache[key] = [val]
                
    def _extract_step_from_subdir(self, eval_subdir: str) -> int:
        match = re.search(r'agent_step_(\d+)', eval_subdir)
        if match is None:
            raise ValueError(f"Could not extract training step from evaluation subdir {eval_subdir!r}.")
        return int(match.group(1))

    def _write_tfd_summary(self, filename: Path, summary: dict) -> None:
        outfile = open(filename, 'w')
        outfile.write(f"step: {summary['step']}\n")
        outfile.write(f"num_episodes: {summary['num_episodes']}\n")
        outfile.write(f"tfd_total_mean: {summary['tfd_total_mean']}\n")
        outfile.write(f"tfd_total_std: {summary['tfd_total_std']}\n")
        outfile.write(f"tfd_total_upper: {summary['tfd_total_mean'] + summary['tfd_total_std']}\n")
        outfile.write(f"tfd_total_lower: {summary['tfd_total_mean'] - summary['tfd_total_std']}\n")
        outfile.close()

    def _load_tfd_summary(self, filename: Path) -> dict:
        summary = {}
        infile = open(filename, 'r')
        for line in infile:
            key, value = line.strip().split(': ', 1)
            summary[key] = value
        infile.close()
        return {
            'step': int(summary['step']),
            'num_episodes': int(summary['num_episodes']),
            'tfd_total_mean': float(summary['tfd_total_mean']),
            'tfd_total_std': float(summary['tfd_total_std'])
        }

    def _collect_tfd_summaries(self, summary_filename: str) -> list:
        root = Path(self.dir) / 'env_data' / self.tag
        summaries = []
        for eval_dir in sorted(path for path in root.glob('agent_step_*') if path.is_dir()):
            summary_path = eval_dir / summary_filename
            if summary_path.is_file():
                summaries.append(self._load_tfd_summary(summary_path))

        summaries.sort(key=lambda summary: summary['step'])
        return summaries

    def _save_tfd_history_plot(self, filename: Path, summary_filename: str) -> None:
        summaries = self._collect_tfd_summaries(summary_filename)
        if not summaries:
            return

        steps = [summary['step'] for summary in summaries]
        means = [summary['tfd_total_mean'] for summary in summaries]
        stds = [summary['tfd_total_std'] for summary in summaries]

        fig, ax = plt.subplots(figsize=(8., 6.))
        ax.errorbar(steps, means, yerr=stds, fmt='-o', color='C0', ecolor='black', elinewidth=1.5, capsize=5)
        ax.set_xlabel('training_step')
        ax.set_ylabel('tfd_total_mean')
        ax.set_title('Evaluation TFD Over Training')
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)

    def _save_tfd_history_plot_v2(self, filename: Path, summary_filename: str) -> None:
        summaries = self._collect_tfd_summaries(summary_filename)
        if not summaries:
            return

        steps = np.array([summary['step'] for summary in summaries])
        means = np.array([summary['tfd_total_mean'] for summary in summaries])
        stds = np.array([summary['tfd_total_std'] for summary in summaries])

        fig, ax = plt.subplots(figsize=(8., 6.))
        ax.plot(steps, means, marker='o', linewidth=2.0, color='C0')
        ax.fill_between(steps, means - stds, means + stds, color='C0', alpha=0.2)
        ax.set_xlabel('training_step')
        ax.set_ylabel('tfd_total_mean')
        ax.set_title('Evaluation TFD Over Training')
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
