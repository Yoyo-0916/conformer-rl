import conformer_rl
from conformer_rl.logging.env_logger import EnvLogger
import numpy as np

def test_basic(mocker):
    logger = EnvLogger(tag = "tag")

    assert logger.tag == "tag"
    assert logger.dir == "data"

    for i in range(10):
        logger.log_step_item("item", i)
        logger.log_step({"doubleitem": i*2, "tripleitem": i*3})

    logger.log_episode_item("epitem", 300)

    assert len(logger.step_data) == 3
    assert len(logger.episode_data) == 1

    logger.log_episode({})

    assert len(logger.episode_data) == 2
    assert "step_data" in logger.episode_data

    logger._add_to_cache(logger.episode_data)
    assert logger.cache["epitem"] == [logger.episode_data["epitem"]]
    assert len(logger.cache) == 2
    assert len(logger.cache["step_data"][0]) == 3

    logger.clear_episode()
    assert len(logger.step_data) == 0
    assert len(logger.episode_data) == 0
    assert len(logger.cache) == 2

    logger.clear_data()
    assert len(logger.cache) == 0

def test_save(mocker):
    logger = EnvLogger(tag = "tag")

    mkd = mocker.patch('conformer_rl.logging.env_logger.mkdir')
    pickle = mocker.patch('conformer_rl.logging.env_logger.pickle')
    chem = mocker.patch('conformer_rl.logging.env_logger.Chem')
    open = mocker.patch('conformer_rl.logging.env_logger.open')

    open.return_value = mocker.Mock()

    molecule = mocker.Mock()
    molecule.GetNumConformers.return_value = 3

    logger.log_episode_item('mol', molecule)

    logger.save_episode(subdir = 'subdir', save_pickle = True, save_molecules = True, save_cache = True)
    
    mkd.assert_called_once_with("data/env_data/tag/subdir")
    assert 'mol' in logger.cache
    assert len(logger.episode_data) == 0

    open.assert_called_with("data/env_data/tag/subdir/data.pickle", "w+b")
    pickle.dump.assert_called_once()
    assert chem.MolToMolFile.call_count == 3


def test_save_tfd_summary(tmp_path, mocker):
    logger = EnvLogger(tag="tag", dir=str(tmp_path))
    eval_dir = tmp_path / 'env_data' / 'tag' / 'agent_step_10'
    (eval_dir / 'ep_0').mkdir(parents=True)
    (eval_dir / 'ep_1').mkdir(parents=True)
    (eval_dir / 'ep_0' / 'data.pickle').write_bytes(b'')
    (eval_dir / 'ep_1' / 'data.pickle').write_bytes(b'')

    def load_pickle(file_obj):
        return {
            str(eval_dir / 'ep_0' / 'data.pickle'): {'mol': 'mol1'},
            str(eval_dir / 'ep_1' / 'data.pickle'): {'mol': 'mol2'},
        }[file_obj.name]

    def mock_tfd_matrix(mol):
        return {
            'mol1': np.array([[0.0, 1.0], [1.0, 0.0]]),
            'mol2': np.array([[0.0, 2.0], [2.0, 0.0]]),
        }[mol]

    pickle = mocker.patch('conformer_rl.logging.env_logger.pickle.load')
    pickle.side_effect = load_pickle
    tfd = mocker.patch('conformer_rl.logging.env_logger.tfd_matrix')
    tfd.side_effect = mock_tfd_matrix
    savefig = mocker.patch('matplotlib.figure.Figure.savefig')

    summary = logger.save_tfd_summary('agent_step_10')

    assert summary == {
        'step': 10,
        'tfd_total_mean': 3.0,
        'tfd_total_std': 1.0,
        'num_episodes': 2
    }
    summary_text = (eval_dir / 'tfd_summary.txt').read_text()
    assert 'tfd_total_mean: 3.0' in summary_text
    assert 'tfd_total_std: 1.0' in summary_text
    assert 'tfd_total_upper: 4.0' in summary_text
    assert 'tfd_total_lower: 2.0' in summary_text
    savefig.assert_called_once()


def test_save_tfd_history_plot_v2(tmp_path, mocker):
    logger = EnvLogger(tag="tag", dir=str(tmp_path))
    root = tmp_path / 'env_data' / 'tag'
    (root / 'agent_step_10').mkdir(parents=True)
    (root / 'agent_step_20').mkdir(parents=True)

    (root / 'agent_step_10' / 'tfd_summary.txt').write_text(
        'step: 10\n'
        'num_episodes: 2\n'
        'tfd_total_mean: 3.0\n'
        'tfd_total_std: 1.0\n'
        'tfd_total_upper: 4.0\n'
        'tfd_total_lower: 2.0\n'
    )
    (root / 'agent_step_20' / 'tfd_summary.txt').write_text(
        'step: 20\n'
        'num_episodes: 2\n'
        'tfd_total_mean: 2.5\n'
        'tfd_total_std: 0.5\n'
        'tfd_total_upper: 3.0\n'
        'tfd_total_lower: 2.0\n'
    )

    savefig = mocker.patch('matplotlib.figure.Figure.savefig')

    logger.save_tfd_history_plot_v2()

    savefig.assert_called_once_with(root / 'tfd_total_history_v2.png')

