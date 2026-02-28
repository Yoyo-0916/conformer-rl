# 

## How to Build

```
conda deactivate && conda remove -n mycfmenv --all -y
```

```
conda create -n mycfmenv python=3.9 -y && conda activate mycfmenv
```

```
pip install setuptools==65.5.0 pip==21
```

```
pip install -r requirements.txt
```

```
pip install torch-geometric==2.2.0  # 這會自動拉取依賴
```

```
conda install -c conda-forge mamba -y
```

```
mamba install -c conda-forge rdkit -y  # 用 conda 安裝以避免編譯錯誤
```

```
pip install conformer-rl[dev]
```

```
pip install -e ".[dev]"  # 假設你在專案根目錄執行
```

```
mamba install -c conda-forge libstdcxx-ng -y
```

```
pip install "numpy<1.24" "setuptools<66" "wheel" --force-reinstall
```

```
pip install gym==0.21.0 --no-deps --force-reinstall
```

```
pip install "stable-baselines3==1.7.0" shimmy==0.2.1
```

+ python example1.py

```
ulimit -n 65535
```

+ python example1.py

```
mamba install -c conda-forge py3Dmol -y
```

To run `example2.py`

```
pip install ligninkmc
```

```
pip uninstall common-wrangler -y
```

```
pip install "common-wrangler==0.2.0"
```



```
pip uninstall -y ligninkmc common_wrangler
```

```
pip install "common_wrangler<1.0"
```

```
pip install ligninkmc
```

## Original conformer-rl README
An open-source deep reinforcement learning library for conformer generation.

[![Documentation Status](https://readthedocs.org/projects/conformer-rl/badge/?version=latest)](https://conformer-rl.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/conformer-rl.svg)](https://badge.fury.io/py/conformer-rl)

### Documentation
Documentation can be found at https://conformer-rl.readthedocs.io/.

### Platform Support
Since conformer-rl can be run within a Conda environment, it should work on all platforms (Windows, MacOS, Linux).

### Installation and Quick Start
Please see the documentation for [installation instructions](https://conformer-rl.readthedocs.io/en/latest/tutorial/install.html) and [getting started](https://conformer-rl.readthedocs.io/en/latest/tutorial/getting_started.html).

### Issues and Feature Requests
We are actively adding new features to this project and are open to all suggestions. If you believe you have encountered a bug, or if you have a feature that you would like to see implemented, please feel free to file an [issue](https://github.com/ZimmermanGroup/conformer-rl/issues).

## Developer Documentation
Pull requests are always welcome for suggestions to improve the code or to add additional features. We encourage new developers to document new features and write unit tests (if applicable). For more information on writing documentation and unit tests, see the [developer documentation](https://conformer-rl.readthedocs.io/en/latest/developer.html).




