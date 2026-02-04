# cli-pydantic

Turn a Pydantic model into a CLI. I dislike every other CLI library so here's yet another one.

- CLI defined by Pydantic
- Use multiple YAML / JSON configs with `--flag` CLI overrides.

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
# CLI, use default from Pydantic
python train.py --model.arch vit_base --model.lr 3e-4 --epochs 50

# From a config file
python train.py base.yaml

# Layer multiple configs, then override with flags
python train.py base.yaml fast.yaml --model.lr 0.05 --epochs 3

# Lists and booleans
python train.py --model.layers 16,32 --verbose
```

## Semantics:

- Use `--foo bar` or `--foo=bar`
- For lists: `--nums=1,2,3` or `--nums 1 --nums=2 --nums 3`
- For bools: `--enable` / `--no-enable`
- Lists will _replace_ previous configs on override -- not append!
