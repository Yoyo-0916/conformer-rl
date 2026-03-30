"""
Analysis
========

Functions for analyzing and visualizing (in Jupyter/IPython notebook) logged
environment data. The functions for visualizations here provide a basic set of functionality
to guide users in understanding the format of the logged environment data. 
Users are encouraged to generate their own plots and visualizations based on their specific needs.
"""

import pickle
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem
from conformer_rl.utils import tfd_matrix
import py3Dmol
import logging
from pprint import pprint
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from typing import Any, List, Optional, Tuple


def _load_tfd_summary_file(filename: str) -> dict:
    """Loads a TFD summary text file created by ``EnvLogger.save_tfd_summary``."""
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

def _load_from_pickle(filename: str) -> Any:
    """Loads an object from a .pickle file.
    """
    infile = open(filename, 'rb') # 以二進位讀取
    data = pickle.load(infile)
    infile.close()
    return data

def load_data_from_pickle(paths: List[str], indices: Optional[List[str]]=None) -> dict:
    """Loads saved pickled environment data from multiple runs into a combined data dict.

    Parameters
    ----------
    paths : list of str
        List of paths to .pickle files corresponding to the environment data from the runs of interest.
    indices : list of str, optional
        Specifies custom indices/labels to be displayed in generated Seaborn graphs for each run. Should be the
        same length as `paths`. If not specified, the labels default to ``test0, test1, test2, ...``.

    Returns
    -------
    dict mapping from str to list
        The str corresponds to the key for the data in the original pickled dict object. The list contains
        the data for each of the environment data sets specified in `paths`, in the same order they were
        given in `paths`.

    Notes
    -----
    The ``.pickle`` files specified by `paths` should be dumped directly by :class:`~conformer_rl.logging.env_logger.EnvLogger`,
    and should correspond to a single evaluation episode.
    See :meth:`conformer_rl.logging.env_logger.EnvLogger.save_episode` for more details on the dumped format.

    An example of how the function operates: Suppose that our paths are::

        ['data1.pickle', 'data2.pickle', 'data3.pickle']

    And each pickle object contains corresponding data::

        data1 = {
            'total_rewards': data1_total_rewards,
            'mol': data1_molecule,
            'rewards': [data1_step1_rewards, data1_step2_rewards, data1_step3_rewards, data1_step4_rewards]
        }
        data2 = {
            'total_rewards': data2_total_rewards,
            'mol': data2_molecule,
            'rewards': [data2_step1_rewards, data2_step2_rewards, data2_step3_rewards, data2_step4_rewards]
        }
        data3 = {
            'total_rewards': data3_total_rewards,
            'mol': data3_molecule,
            'rewards': [data3_step1_rewards, data3_step2_rewards, data3_step3_rewards, data3_step4_rewards]
        }
    
    Suppose that data1 corresponds to some eval data obtained from training with the PPO agent, data2 was obtained
    from the PPORecurrent agent, and data3 was obtained from training with the A2C agent. Then we can input custom `indices`
    to help us understand each dataset better::

        indices = ['PPO', 'PPO_recurrent', 'A2C']

    Given these data and indices, :func:`load_data_from_pickle` would return the following dict::

        {
            'indices': ['PPO', 'PPO_recurrent', 'A2C'],
            'total_rewards': [
                data1_total_rewards,
                data2_total_rewards,
                data3_total_rewards
            ],
            'mol': [
                data1_molecule,
                data2_molecule,
                data3_molecule
            ],
            'rewards': [
                [data1_step1_rewards, data1_step2_rewards, data1_step3_rewards, data1_step4_rewards],
                [data2_step1_rewards, data2_step2_rewards, data2_step3_rewards, data2_step4_rewards],
                [data3_step1_rewards, data3_step2_rewards, data3_step3_rewards, data3_step4_rewards]
            ],
        }
    
    This format consolidates all the data into a single dict and is compatible with the other visualization
    functions in this module. Furthermore, it is also easy to convert a dict of this format into a Pandas dataframe
    or other tabular formats if needed.
    """

    if not isinstance(paths, list): # 如果 paths 不是列表
        paths = [paths]             # 轉換為列表

    if indices is None:
        indices = [f'test{i}' for i, x in enumerate(paths)] # 產生預設 indices 如 test0, test1 等

    data = map(_load_from_pickle, paths) # 使用 map 載入每個 paths 的資料
    data =  list(data)
    pprint(data[0])
    pprint(data[0]['total_rewards'])
    # print(data[0])                                # output: {'mol': <rdkit.Chem.rdchem.Mol object at 0x7f8c8c3e2d60>, 'total_rewards': 23.456, 'step_data': {'rewards': [1.2, 2.3, 3.4, ...]}}
    print(len(data), data[0].keys())              # output: 3 dict_keys(['mol', 'total_rewards', 'step_data'])


    final_data = {"indices": indices}
    # print(final_data)                             # output: {'indices': ['test0', 'test1', 'test2']}
    for datum in data:
        if 'step_data' in datum:                # 如果有 step_data 這個 key
            datum.update(datum['step_data'])
            del datum['step_data']
        for key, val in datum.items():
            final_data.setdefault(key, []).append(val)
    
    # print(final_data.keys())  # output: dict_keys(['indices', 'mol', 'total_rewards', 'rewards'])
    return final_data

def load_tfd_data_from_folders(folder_paths: List[str], indices: Optional[List[str]]=None, pattern: str='*.pickle') -> dict:
    """Loads pickle files from each folder, calculates TFD totals per pickle, and
    aggregates per-folder summary statistics.

    Parameters
    ----------
    folder_paths : list of str
        List of folder paths. Each folder is expected to contain one or more pickle files
        matching `pattern`.
    indices : list of str, optional
        Specifies custom labels for each folder. If not specified, folder names are used.
    pattern : str
        Glob pattern used to discover pickle files inside each folder.

    Returns
    -------
    dict mapping from str to list
        A dict containing folder labels, discovered pickle paths, the `tfd_total` values
        for each pickle in each folder, the average `tfd_total` for each folder, and
        the standard deviation of `tfd_total` for each folder.
    """
    if not isinstance(folder_paths, list):
        folder_paths = [folder_paths]

    resolved_folders = [Path(folder) for folder in folder_paths]

    if indices is None:
        indices = [folder.name for folder in resolved_folders]

    if len(indices) != len(resolved_folders):
        raise ValueError('indices must have the same length as folder_paths.')

    final_data = {
        'indices': indices,
        'pickle_paths': [],
        'tfd_total': [],
        'tfd_total_average': [],
        'tfd_total_std': []
    }

    for folder in resolved_folders:
        pickle_paths = sorted(str(path) for path in folder.glob(pattern) if path.is_file())
        if not pickle_paths:
            raise ValueError(f'No pickle files matching {pattern!r} were found in folder {str(folder)!r}.')

        folder_data = load_data_from_pickle(pickle_paths)
        calculate_tfd(folder_data)

        folder_tfd_total = folder_data['tfd_total']
        final_data['pickle_paths'].append(pickle_paths)
        final_data['tfd_total'].append(folder_tfd_total)
        final_data['tfd_total_average'].append(float(np.mean(folder_tfd_total)))
        final_data['tfd_total_std'].append(float(np.std(folder_tfd_total)))

    return final_data


def load_tfd_summary_data_from_runs(run_paths: List[str], indices: Optional[List[str]]=None, summary_filename: str='tfd_summary.txt') -> dict:
    """Loads per-evaluation TFD summaries from one or more training run directories.

    Parameters
    ----------
    run_paths : list of str
        List of directories, where each directory contains ``agent_step_*`` folders
        produced by a training run.
    indices : list of str, optional
        Custom labels for each run. If not specified, the directory names are used.
    summary_filename : str
        Name of the summary text file inside each ``agent_step_*`` folder.

    Returns
    -------
    dict
        A dict containing run labels and, for each run, the loaded summary paths,
        training steps, per-step mean TFD totals, per-step std TFD totals, and the
        number of evaluation episodes.
    """
    if not isinstance(run_paths, list):
        run_paths = [run_paths]

    resolved_runs = [Path(run_path) for run_path in run_paths]

    if indices is None:
        indices = [run_path.name for run_path in resolved_runs]

    if len(indices) != len(resolved_runs):
        raise ValueError('indices must have the same length as run_paths.')

    final_data = {
        'indices': indices,
        'summary_paths': [],
        'steps': [],
        'tfd_total_mean': [],
        'tfd_total_std': [],
        'num_episodes': []
    }

    for run_path in resolved_runs:
        summaries = []
        summary_paths = sorted(
            path for path in run_path.glob(f'agent_step_*/{summary_filename}')
            if path.is_file()
        )
        if not summary_paths:
            raise ValueError(
                f'No summary files named {summary_filename!r} were found in run folder {str(run_path)!r}.'
            )

        for summary_path in summary_paths:
            summary = _load_tfd_summary_file(str(summary_path))
            summaries.append(summary)

        summaries.sort(key=lambda summary: summary['step'])
        final_data['summary_paths'].append([str(path) for path in summary_paths])
        final_data['steps'].append([summary['step'] for summary in summaries])
        final_data['tfd_total_mean'].append([summary['tfd_total_mean'] for summary in summaries])
        final_data['tfd_total_std'].append([summary['tfd_total_std'] for summary in summaries])
        final_data['num_episodes'].append([summary['num_episodes'] for summary in summaries])

    return final_data


def line_plot_tfd_summary_from_runs(
    run_paths: List[str],
    indices: Optional[List[str]]=None,
    summary_filename: str='tfd_summary.txt',
    figsize: Tuple[float, float]=(8., 6.)
) -> matplotlib.axes.Axes:
    """Plots TFD history curves from multiple training runs on the same axes.

    Each line corresponds to one run. The line shows ``tfd_total_mean`` over
    training steps, and the shaded band shows plus/minus one standard deviation.
    """
    data = load_tfd_summary_data_from_runs(
        run_paths=run_paths,
        indices=indices,
        summary_filename=summary_filename
    )

    fig, axes = plt.subplots(figsize=figsize)
    for i, label in enumerate(data['indices']):
        steps = np.array(data['steps'][i])
        means = np.array(data['tfd_total_mean'][i])
        stds = np.array(data['tfd_total_std'][i])
        axes.plot(steps, means, marker='o', label=label)
        axes.fill_between(steps, means - stds, means + stds, alpha=0.2)

    axes.set(xlabel='training_step', ylabel='tfd_total_mean')
    axes.set_title('Evaluation TFD Over Training')
    axes.legend()

    return fig, axes

def list_keys(data: dict) -> List[str]:
    """Return a list of all keys in a dict.

    Parameters
    ----------
    data : dict
        The dictionary to retrieve keys from.
    """
    return list(key for key, val in data.items())

def bar_plot_episodic(key: str, data: dict) -> matplotlib.axes.Axes:
    """Plots a bar plot comparing a scalar value across all episodes loaded in `data`.

    Parameters
    ----------
    key : str
        The key for the values to be compared across all data sets/episodes.
    data : dict
        Data dictionary generated by :meth:`load_data_from_pickle`.
    """
    ax = sns.barplot(x=data["indices"], y=data[key]) # 使用 Seaborn 繪製柱狀圖
    ax.set(xlabel='run', ylabel=key) # 設定 x 軸和 y 軸標籤
    return ax # 返回軸物件


def histogram_select_episodes(key: str, data: dict, episodes: List[int]=None, binwidth: float=10, figsize: Tuple[float, float]=(8., 6.)) -> matplotlib.axes.Axes:
    """Plots a single histogram where data for each episode in `episodes` are overlayed.

    Parameters
    ----------
    key : str
        The key for the values to be compared across all data sets/episodes.
    data : dict
        Data dictionary generated by :meth:`load_data_from_pickle`.
    episodes : list of int, optional
        Specifies the indices in `data` for the episodes to be shown. If not specified,
        all episodes are shown.
    binwidth: float
        The width of each bin in the histogram.
    figsize: 2-tuple of float
        Specifies the size of the plot.
    """
    if episodes is None:
        episodes = list(range(len(data['indices'])))
    fig, axes = plt.subplots()
    input_data = {data["indices"][i]: data[key][i] for i in episodes}
    sns.histplot(data=input_data, binwidth=binwidth, ax=axes)
    axes.set(xlabel=key)

    return fig, axes

def bar_plot_tfd_average_from_folders(folder_paths: List[str], indices: Optional[List[str]]=None, pattern: str='*.pickle', figsize: Tuple[float, float]=(8., 6.)) -> matplotlib.axes.Axes:
    """Plots a bar plot where each bar is the average `tfd_total` across all pickle
    files in a folder, with error bars showing one standard deviation.

    Parameters
    ----------
    folder_paths : list of str
        List of folder paths. Each folder should contain one or more pickle files.
    indices : list of str, optional
        Specifies custom labels for each folder. If not specified, folder names are used.
    pattern : str
        Glob pattern used to discover pickle files inside each folder.
    figsize: 2-tuple of float
        Specifies the size of the plot.

    Notes
    -----
    The per-folder average `tfd_total` values are available from
    :func:`load_tfd_data_from_folders` under the `tfd_total_average` key, and the
    standard deviations are available under the `tfd_total_std` key.
    """
    data = load_tfd_data_from_folders(folder_paths=folder_paths, indices=indices, pattern=pattern)

    fig, axes = plt.subplots(figsize=figsize)
    sns.barplot(x=data['indices'], y=data['tfd_total_average'], ax=axes)
    axes.errorbar(
        x=np.arange(len(data['indices'])),
        y=data['tfd_total_average'],
        yerr=data['tfd_total_std'],
        fmt='none',
        ecolor='black',
        elinewidth=1.5,
        capsize=5
    )
    axes.set(xlabel='folder', ylabel='tfd_total_average')

    return fig, axes

def histogram_episodic(key: str, data: dict, binwidth: int=10, figsize: Tuple[float, float]=(8., 6.)) -> matplotlib.axes.Axes:
    """Plots histogram on separate axes for each of the episode data sets in `data`.

    Parameters
    ----------
    key : str
        The key for the values to be compared across all data sets/episodes.
    data : dict
        Data dictionary generated by :meth:`load_data_from_pickle`.
    binwidth: float
        The width of each bin in the histogram.
    figsize: 2-tuple of float
        Specifies the size of the plot.
    """
    n = len(data["indices"])
    fig, axes = plt.subplots(nrows=n, figsize=figsize)

    if n == 1:
        sns.histplot(data={data['indices'][0]: data[key][0]}, binwidth=binwidth, ax=axes)
        axes.set(xlabel=key)
    else:
        for i, index in enumerate(data["indices"]):
            sns.histplot(data={index: data[key][i]}, binwidth=binwidth, ax=axes[i])
            axes[i].set(xlabel=key)

    return fig, axes

def heatmap_episodic(key: str, data: dict, figsize: Tuple[float, float]=(8., 6.)) -> matplotlib.axes.Axes:
    """Plots heatmap(s) for matrix data corresponding to `key` across all episodes
    loaded in `data`.

    Parameters
    ----------
    key : str
        The key for the values to be compared across all data sets/episodes.
    data : dict
        Data dictionary generated by :meth:`load_data_from_pickle`.
    figsize: 2-tuple of float
        Specifies the size of the plot.
    """
    n = len(data["indices"])
    fig, axes = plt.subplots(nrows=n, figsize=figsize)
    if n == 1:
        sns.heatmap(data=data[key][0], ax=axes)
        axes.set(xlabel=data['indices'][0])
    else:
        for i, index in enumerate(data["indices"]):
            sns.heatmap(data=data[key][i], ax=axes[i])
            axes[i].set(xlabel=index)
    
    return fig, axes

def get_global_vmin_vmax(data, key):
    all_values = np.concatenate([data[key][i].flatten() for i in range(len(data["indices"]))])
    return all_values.min(), all_values.max()

def cluster_heatmap_episodic(
    key,
    data,
    figsize=(8., 6.),
    method="average"
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, leaves_list

    n = len(data["indices"])
    fig, axes = plt.subplots(nrows=n, figsize=figsize)

    if n == 1:
        axes = [axes]

    vmin, vmax = get_global_vmin_vmax(data, key)

    for i, index in enumerate(data["indices"]):
        matrix = data[key][i]

        condensed = squareform(matrix)
        Z = linkage(condensed, method=method)
        order = leaves_list(Z)

        sorted_matrix = matrix[order][:, order]

        sns.heatmap(
            sorted_matrix,
            ax=axes[i]
        )

        axes[i].set(xlabel=index)

    return fig, axes

def calculate_tfd(data: str) -> None:
    """Updates data with the TFD (Torsion Fingerprint Deviation) matrix (with key 'tfd_matrix') and sum of the TFD matrix
    (with key 'tfd_total') for the molecule conformers across each episode loaded in `data`.

    Parameters
    ----------
    data : dict
        Data dictionary generated by :meth:`load_data_from_pickle`.
    """
    if not 'mol' in data:
        raise Exception('data dict must contain RDKit Mol object with \'mol\' key to generate tfd matrix.')
    if 'tfd_matrix' in data or 'tfd_total' in data:
        logging.info("tfd matrix already exists, recalculating...")
        data.pop('tfd_matrix')
        data.pop('tfd_total')
    for mol in data['mol']:
        matrix = tfd_matrix(mol)
        data.setdefault('tfd_matrix', []).append(matrix)
        data.setdefault('tfd_total', []).append(np.sum(matrix))

def drawConformer(mol: Chem.Mol, confId: int=-1, size: Tuple[int, int]=(300, 300), style: str="stick") -> py3Dmol.view:
    """Displays interactive 3-dimensional representation of specified conformer.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule containing the conformer to be displayed.
    confId : int
        The ID of the conformer to be displayed.
    size : Tuple[int, int]
        The size of the display (width, height).
    style: str
        The drawing style for displaying the molecule. Can be sphere, stick, line, cross, cartoon, and surface.
    """
    print(f'Generating 3D view for conformer ID: {confId}')
    block = Chem.MolToMolBlock(mol, confId=confId)
    view = py3Dmol.view(width=size[0], height=size[1])
    view.addModel(block, 'mol')
    view.setStyle({style : {}})
    view.zoomTo()
    print(f'Displaying conformer ID: {confId}')
    view.png
    return view

def drawConformer_episodic(data: dict, confIds: List[int], size: Tuple[int, int]=(300, 300), style: str="stick") -> py3Dmol.view:
    """Displays a specified conformer for each episode loaded in `data`.

    Parameters
    ----------
    data : dict from string to list
        Contains the loaded episode information. 'mol' must be a key in data and the corresponding list must contain
        RDKit Mol objects.
    confIds : list of int
        The indices for the conformers to be displayed (for each episode loaded in data).
    size : Tuple[int, int]
        The size of the display for each individual molecule (width, height).
    style: str
        The drawing style for displaying the molecule. Can be sphere, stick, line, cross, cartoon, and surface.
    """
    n = len(data['mol'])
    view = py3Dmol.view(width=size[0]*n, height=size[0]*n, linked=False, viewergrid=(n, 1))
    for i in range(n):
        block = Chem.MolToMolBlock(data['mol'][i], confId=confIds[i])
        view.addModel(block, 'mol', viewer=(i, 0))
        view.setStyle({style:{}}, viewer=(i, 0))
    view.zoomTo()
    return view
