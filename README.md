# cli-pydantic

Turn a Pydantic model into a CLI. Config files (JSON/YAML) and `--override` flags are merged and validated automatically.

## Install

```
pip install cli-pydantic
```

## Usage

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

```bash
# pure CLI
python train.py --model.arch vit_base --model.lr 3e-4 --epochs 50

# from a config file
python train.py base.yaml

# layer multiple configs, then override with flags
python train.py base.yaml fast.yaml --model.lr 0.05 --epochs 3

# lists and booleans
python train.py --model.layers 16,32 --verbose --no-verbose
```
