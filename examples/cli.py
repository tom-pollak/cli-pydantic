"""
config.yaml
---
foo: 3
nest:
  b: [1, 2]
---

$ uv run examples/cli.py --config examples/config.yaml --bar test1 --bar test2 --nest.a 3.14 --nest.b 69
>>> CLIConfig(foo=3, bar=['test1', 'test2'], nest=Nested(a=3.14, b=[69]))
"""

import argparse
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from cli_pydantic import deep_merge, model_help, parse_unknown_args


class Nested(BaseModel):
    a: float = Field(description="Learning rate")
    b: list[int]


class CLIConfig(BaseModel):
    foo: int = Field(default=2, description="Number of workers")
    bar: list[str]
    nest: Nested


def main():
    parser = argparse.ArgumentParser(
        description="Simple cli",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Config overrides:\n" + "\n".join(model_help(CLIConfig)),
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to YAML config"
    )

    args, unknown = parser.parse_known_args()
    config_path = args.config
    del args

    data = yaml.safe_load(config_path.read_text())
    overrides = parse_unknown_args(unknown, model_cls=CLIConfig)
    config = CLIConfig.model_validate(deep_merge(data, overrides))
    return config


if __name__ == "__main__":
    print(repr(main()))
