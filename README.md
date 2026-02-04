# cli-pydantic

Turn a Pydantic model into a CLI. Config files (JSON/YAML) and `--override` flags are merged and validated automatically.

## Install

```
pip install cli-pydantic
```

## Usage

Define your config with nested models, lists, and flags:

```python
# train.py
from pydantic import BaseModel
from cli_pydantic import cli

class Data(BaseModel):
    path: str = "./data"
    splits: list[str] = ["train", "val"]

class Model(BaseModel):
    arch: str = "resnet50"
    lr: float = 1e-3
    layers: list[int] = [64, 128, 256]

class Config(BaseModel):
    data: Data = Data()
    model: Model = Model()
    epochs: int = 10
    verbose: bool = False

cfg = cli(Config, desc="Training pipeline")
```

### Config files

```yaml
# base.yaml
epochs: 50
data:
  path: /datasets/imagenet
model:
  arch: vit_base
  lr: 3e-4
```

```yaml
# fast.yaml — a smaller run for debugging
epochs: 5
model:
  lr: 1e-2
  layers: [32, 64]
data:
  splits: [train]
```

### Layered configs with CLI overrides

Multiple config files are deep-merged left to right, then CLI flags override everything:

```bash
# base config only
python train.py base.yaml

# base + fast overlay (fast.yaml overrides base.yaml)
python train.py base.yaml fast.yaml

# base + fast + CLI overrides on top
python train.py base.yaml fast.yaml --model.lr 0.05 --epochs 3

# lists via comma-separated values
python train.py base.yaml --model.layers 16,32 --data.splits train,val,test

# boolean flags
python train.py base.yaml --verbose
python train.py base.yaml --no-verbose
```

### No config file — pure CLI

```bash
python train.py --model.arch resnet18 --model.lr 0.01 --data.path ./my_data --epochs 20
```
