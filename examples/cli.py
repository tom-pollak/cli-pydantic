"""
$ uv run examples/cli.py examples/config.yaml --bar test1 --bar test2 --nest.a 3.14 --nest.b 69
>>> CLIConfig(foo=3, bar=['test1', 'test2'], nest=Nested(a=3.14, b=[69]))
"""

from pydantic import BaseModel, Field

from cli_pydantic import cli


class Nested(BaseModel):
    a: float = Field(description="Learning rate")
    b: list[int]


class CLIConfig(BaseModel):
    foo: int = Field(default=2, description="Number of workers")
    bar: list[str]
    nest: Nested


if __name__ == "__main__":
    config = cli(CLIConfig, desc="Simple cli")
    print(repr(config))
