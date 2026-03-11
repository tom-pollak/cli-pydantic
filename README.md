# cli-pydantic

Turn a Pydantic model into a CLI. I dislike every other CLI library so here's another one.

- CLI defined by Pydantic
- Use multiple YAML / JSON configs with `--flag` CLI overrides.

## Install

```
pip install cli-pydantic
```

## Usage

```python
from pydantic import BaseModel, Field
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
    profile: bool = Field(False, description="dump chrome trace")

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
python train.py --model.layers 16,32 --profile
```

## Semantics

- Use `--foo bar` or `--foo=bar`
- For lists: `--nums=1,2,3` or `--nums 1 --nums=2 --nums 3`
- For bools: `--enable` / `--no-enable`
- Lists will _replace_ previous configs on override -- not append!

## Automatic help

```bash
$ python train.py --help
help: Training pipeline

usage: train.py [-h] [configs ...] [--overrides ...]

config arguments:
  --data.path str          (default: ./data)
  --data.splits list[str]  (default: ['train', 'val'])
  --model.arch str         (default: resnet50)
  --model.lr float         (default: 0.001)
  --model.layers list[int] (default: [64, 128, 256])
  --epochs int             (default: 10)
  --verbose bool           dump chrome trace (default: False)
```


## Errors

### Validation

```python
cli(Config, argv=['--epochs', 'not_a_number', '--model.lr', 'also_bad'])
```

```
Validation failed:
  model.lr: Input should be a valid number, unable to parse string as a number
  epochs: Input should be a valid integer, unable to parse string as an integer
```

### Config

```python
cli(Config, argv=['--nonexistent', 'val'])
```

```
Unknown option: --nonexistent
```

## Docstring

```python
cli[T: BaseModel](model_cls: type[T], desc: str = '', argv: list[str] | None = None, raise_on_error: bool = False) -> T
    Build a CLI from a Pydantic model, merging config files and --overrides.

    Positional arguments are paths to JSON/YAML config files (later files
    override earlier ones).  Any remaining ``--key value`` flags are parsed
    as field overrides using dot-notation (e.g. ``--model.lr 0.01``).

    Args:
        model_cls: The Pydantic model class that defines the config schema.
        desc: Optional description shown in ``--help`` output.
        argv: Raw CLI args. Default: ``sys.argv[1:]``.
        raise_on_error: If True, raise ``ConfigError`` on bad input.
            If False (default), print the error to stderr and exit.

    Returns:
        A validated instance of *model_cls*.
```
