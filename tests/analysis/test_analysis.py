from conformer_rl.analysis import analysis
import numpy as np

def test_load_pickle(mocker):
    open = mocker.patch('conformer_rl.analysis.analysis.open')
    load = mocker.patch('conformer_rl.analysis.analysis.pickle.load')
    file = mocker.Mock()


    open.return_value = file

    analysis._load_from_pickle('filename')

    open.assert_called_with('filename', 'rb')
    load.assert_called_with(file)

def test_load_data_single(mocker):
    def load_pickle(path):
        d = {
            'path1': {
                'data1': 1
            }
        }
        return d[path]
    pickle = mocker.patch('conformer_rl.analysis.analysis._load_from_pickle')
    pickle.side_effect = load_pickle

    data = analysis.load_data_from_pickle(paths = ['path1'])
    assert data == {'indices': ['test0'], 'data1': [1]}

def test_load_data_multiple(mocker):
    def load_pickle(path):
        d = {
            'data1': {
            'total_rewards': 'data1_total_rewards',
            'mol': 'data1_molecule',
            'rewards': ['data1_step1_rewards', 'data1_step2_rewards', 'data1_step3_rewards', 'data1_step4_rewards']
            },
            'data2': {
            'total_rewards': 'data2_total_rewards',
            'mol': 'data2_molecule',
            'rewards': ['data2_step1_rewards', 'data2_step2_rewards', 'data2_step3_rewards', 'data2_step4_rewards']
            },
            'data3': {
            'total_rewards': 'data3_total_rewards',
            'mol': 'data3_molecule',
            'rewards': ['data3_step1_rewards', 'data3_step2_rewards', 'data3_step3_rewards', 'data3_step4_rewards']
            },
        }

        return d[path]

    pickle = mocker.patch('conformer_rl.analysis.analysis._load_from_pickle')
    pickle.side_effect = load_pickle
    data = analysis.load_data_from_pickle(paths = ['data1', 'data2', 'data3'], indices = ['PPO', 'PPO_recurrent', 'A2C'])
    assert data == {
        'indices': ['PPO', 'PPO_recurrent', 'A2C'],
        'total_rewards': [
            'data1_total_rewards',
            'data2_total_rewards',
            'data3_total_rewards'
        ],
        'mol': [
            'data1_molecule',
            'data2_molecule',
            'data3_molecule'
        ],
        'rewards': [
            ['data1_step1_rewards', 'data1_step2_rewards', 'data1_step3_rewards', 'data1_step4_rewards'],
            ['data2_step1_rewards', 'data2_step2_rewards', 'data2_step3_rewards', 'data2_step4_rewards'],
            ['data3_step1_rewards', 'data3_step2_rewards', 'data3_step3_rewards', 'data3_step4_rewards']
        ]
    }

    
    

def test_list_keys():
    d = {'test1': 1, 'test2': 2, 'test3': 3}
    assert analysis.list_keys(d) == ['test1', 'test2', 'test3']

def test_load_tfd_data_from_folders(tmp_path, mocker):
    folder1 = tmp_path / 'folder1'
    folder2 = tmp_path / 'folder2'
    folder1.mkdir()
    folder2.mkdir()

    file1 = folder1 / 'a.pickle'
    file2 = folder1 / 'b.pickle'
    file3 = folder2 / 'c.pickle'
    for path in [file1, file2, file3]:
        path.write_bytes(b'')

    def load_pickle(path):
        return {
            str(file1): {'mol': 'mol1'},
            str(file2): {'mol': 'mol2'},
            str(file3): {'mol': 'mol3'},
        }[path]

    def mock_tfd_matrix(mol):
        return {
            'mol1': np.array([[0.0, 1.0], [1.0, 0.0]]),
            'mol2': np.array([[0.0, 2.0], [2.0, 0.0]]),
            'mol3': np.array([[0.0, 3.0], [3.0, 0.0]]),
        }[mol]

    pickle = mocker.patch('conformer_rl.analysis.analysis._load_from_pickle')
    pickle.side_effect = load_pickle
    tfd = mocker.patch('conformer_rl.analysis.analysis.tfd_matrix')
    tfd.side_effect = mock_tfd_matrix

    data = analysis.load_tfd_data_from_folders(
        folder_paths=[str(folder1), str(folder2)],
        indices=['run1', 'run2']
    )

    assert data['indices'] == ['run1', 'run2']
    assert data['pickle_paths'] == [
        [str(file1), str(file2)],
        [str(file3)]
    ]
    assert data['tfd_total'] == [[2.0, 4.0], [6.0]]
    assert data['tfd_total_average'] == [3.0, 6.0]
    assert data['tfd_total_std'] == [1.0, 0.0]

def test_load_tfd_data_from_folders_recursive_root(tmp_path, mocker):
    root = tmp_path / 'evals'
    folder1 = root / 'agent_step_10' / 'ep_0'
    folder2 = root / 'agent_step_10' / 'ep_1'
    folder1.mkdir(parents=True)
    folder2.mkdir(parents=True)

    file1 = folder1 / 'data.pickle'
    file2 = folder2 / 'data.pickle'
    file1.write_bytes(b'')
    file2.write_bytes(b'')

    def load_pickle(path):
        return {
            str(file1): {'mol': 'mol1'},
            str(file2): {'mol': 'mol2'},
        }[path]

    def mock_tfd_matrix(mol):
        return {
            'mol1': np.array([[0.0, 1.0], [1.0, 0.0]]),
            'mol2': np.array([[0.0, 2.0], [2.0, 0.0]]),
        }[mol]

    pickle = mocker.patch('conformer_rl.analysis.analysis._load_from_pickle')
    pickle.side_effect = load_pickle
    tfd = mocker.patch('conformer_rl.analysis.analysis.tfd_matrix')
    tfd.side_effect = mock_tfd_matrix

    data = analysis.load_tfd_data_from_folders(folder_paths=str(root))

    assert data['indices'] == ['agent_step_10/ep_0', 'agent_step_10/ep_1']
    assert data['pickle_paths'] == [[str(file1)], [str(file2)]]
    assert data['tfd_total_average'] == [2.0, 4.0]
    assert data['tfd_total_std'] == [0.0, 0.0]
    
def test_bar_plot_tfd_average_from_folders(mocker):
    mock_data = {
        'indices': ['run1', 'run2'],
        'tfd_total': [[2.0, 4.0], [6.0, 8.0]],
        'tfd_total_average': [3.0, 7.0],
        'tfd_total_std': [1.0, 1.0]
    }

    loader = mocker.patch('conformer_rl.analysis.analysis.load_tfd_data_from_folders')
    loader.return_value = mock_data
    barplot = mocker.patch('conformer_rl.analysis.analysis.sns.barplot')

    fig, axes = analysis.bar_plot_tfd_average_from_folders(
        folder_paths=['folder1', 'folder2'],
        indices=['run1', 'run2']
    )

    loader.assert_called_with(folder_paths=['folder1', 'folder2'], indices=['run1', 'run2'], pattern='*.pickle')
    barplot.assert_called_once_with(x=['run1', 'run2'], y=[3.0, 7.0], ax=axes)
    assert len(axes.lines) == 1
    assert axes.get_xlabel() == 'folder'
    assert axes.get_ylabel() == 'tfd_total_average'


def test_load_tfd_summary_data_from_runs(tmp_path):
    run1 = tmp_path / 'run1'
    run2 = tmp_path / 'run2'
    (run1 / 'agent_step_10').mkdir(parents=True)
    (run1 / 'agent_step_20').mkdir(parents=True)
    (run2 / 'agent_step_5').mkdir(parents=True)

    (run1 / 'agent_step_10' / 'tfd_summary.txt').write_text(
        'step: 10\n'
        'num_episodes: 3\n'
        'tfd_total_mean: 4.0\n'
        'tfd_total_std: 1.5\n'
        'tfd_total_upper: 5.5\n'
        'tfd_total_lower: 2.5\n'
    )
    (run1 / 'agent_step_20' / 'tfd_summary.txt').write_text(
        'step: 20\n'
        'num_episodes: 3\n'
        'tfd_total_mean: 2.0\n'
        'tfd_total_std: 0.5\n'
        'tfd_total_upper: 2.5\n'
        'tfd_total_lower: 1.5\n'
    )
    (run2 / 'agent_step_5' / 'tfd_summary.txt').write_text(
        'step: 5\n'
        'num_episodes: 4\n'
        'tfd_total_mean: 6.0\n'
        'tfd_total_std: 2.0\n'
        'tfd_total_upper: 8.0\n'
        'tfd_total_lower: 4.0\n'
    )

    data = analysis.load_tfd_summary_data_from_runs(
        run_paths=[str(run1), str(run2)],
        indices=['baseline', 'improved']
    )

    assert data['indices'] == ['baseline', 'improved']
    assert data['steps'] == [[10, 20], [5]]
    assert data['tfd_total_mean'] == [[4.0, 2.0], [6.0]]
    assert data['tfd_total_std'] == [[1.5, 0.5], [2.0]]
    assert data['num_episodes'] == [[3, 3], [4]]


def test_line_plot_tfd_summary_from_runs(mocker):
    loader = mocker.patch('conformer_rl.analysis.analysis.load_tfd_summary_data_from_runs')
    loader.return_value = {
        'indices': ['run1', 'run2'],
        'steps': [[10, 20], [10, 20]],
        'tfd_total_mean': [[4.0, 2.0], [5.0, 3.0]],
        'tfd_total_std': [[1.0, 0.5], [1.5, 0.25]],
        'num_episodes': [[3, 3], [3, 3]],
        'summary_paths': [['a', 'b'], ['c', 'd']]
    }

    fig, axes = analysis.line_plot_tfd_summary_from_runs(
        run_paths=['run1_path', 'run2_path'],
        indices=['run1', 'run2']
    )

    loader.assert_called_with(
        run_paths=['run1_path', 'run2_path'],
        indices=['run1', 'run2'],
        summary_filename='tfd_summary.txt'
    )
    assert len(axes.lines) == 2
    assert axes.get_xlabel() == 'training_step'
    assert axes.get_ylabel() == 'tfd_total_mean'
    assert axes.get_legend() is not None
