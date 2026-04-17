"""Tests for cli_pydantic."""

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from cli_pydantic import CliError, cli


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


class DynamicConfig(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


class StrictConfig(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


@pytest.fixture()
def yaml_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("epochs: 50\nmodel:\n  lr: 0.01\n")
    return p


@pytest.fixture()
def json_config(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"epochs": 25, "model": {"arch": "vit_large"}}))
    return p


def test_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    assert cli(Config) == Config()

    monkeypatch.setattr("sys.argv", ["prog", "--epochs", "3", "--model.lr", "0.05"])
    cfg = cli(Config)
    assert cfg.epochs == 3
    assert cfg.model.lr == 0.05

    monkeypatch.setattr("sys.argv", ["prog", "--model.arch", "vit_base"])
    assert cli(Config).model.arch == "vit_base"

    monkeypatch.setattr("sys.argv", ["prog", "--epochs=5"])
    assert cli(Config).epochs == 5

    monkeypatch.setattr("sys.argv", ["prog", "--verbose"])
    assert cli(Config).verbose is True

    monkeypatch.setattr("sys.argv", ["prog", "--no-verbose"])
    assert cli(Config).verbose is False

    monkeypatch.setattr("sys.argv", ["prog", "--model.layers", "32,64"])
    assert cli(Config).model.layers == [32, 64]

    monkeypatch.setattr("sys.argv", ["prog", "--data.path", "a,b"])
    assert cli(Config).data.path == "a,b"


def test_configs(monkeypatch, yaml_config, json_config, tmp_path):
    monkeypatch.setattr("sys.argv", ["prog", str(yaml_config)])
    cfg = cli(Config)
    assert cfg.epochs == 50
    assert cfg.model.lr == 0.01

    monkeypatch.setattr("sys.argv", ["prog", str(json_config)])
    cfg = cli(Config)
    assert cfg.epochs == 25
    assert cfg.model.arch == "vit_large"

    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    monkeypatch.setattr("sys.argv", ["prog", str(empty)])
    assert cli(Config) == Config()

    # later config overrides earlier
    monkeypatch.setattr("sys.argv", ["prog", str(yaml_config), str(json_config)])
    cfg = cli(Config)
    assert cfg.epochs == 25
    assert cfg.model.arch == "vit_large"
    assert cfg.model.lr == 0.01  # from yaml, not overridden

    # flags override config
    monkeypatch.setattr("sys.argv", ["prog", str(yaml_config), "--epochs", "3"])
    cfg = cli(Config)
    assert cfg.epochs == 3
    assert cfg.model.lr == 0.01


def test_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["prog", "--help"])
    with pytest.raises(SystemExit, match="0"):
        cli(Config, desc="Training pipeline")
    out = capsys.readouterr().out
    assert "--epochs" in out
    assert "Training pipeline" in out


def test_errors(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["prog", "-x"])
    with pytest.raises(CliError, match="Expected --key"):
        cli(Config, raise_on_error=True)

    monkeypatch.setattr("sys.argv", ["prog", str(tmp_path / "nope.yaml")])
    with pytest.raises(CliError, match="Config file not found"):
        cli(Config, raise_on_error=True)

    toml = tmp_path / "config.toml"
    toml.write_text("x = 1")
    monkeypatch.setattr("sys.argv", ["prog", str(toml)])
    with pytest.raises(CliError, match="Unsupported config file type"):
        cli(Config, raise_on_error=True)

    # pydantic validation errors are wrapped in CliError
    monkeypatch.setattr("sys.argv", ["prog", "--epochs", "not_a_number"])
    with pytest.raises(CliError, match="epochs") as exc_info:
        cli(Config, raise_on_error=True)
    assert "Validation failed" in str(exc_info.value)

    monkeypatch.setattr("sys.argv", ["prog", "--BLOCK_M", "32"])
    with pytest.raises(CliError, match="BLOCK_M"):
        cli(StrictConfig, raise_on_error=True)

    monkeypatch.setattr("sys.argv", ["prog", "--foo=1", "--foo.bar=2"])
    with pytest.raises(CliError, match="Conflicting values for foo"):
        cli(DynamicConfig, raise_on_error=True)


def test_dynamic_flags():
    cfg = cli(DynamicConfig, argv=["--args.BLOCK_M=32"])
    assert cfg.args == {"BLOCK_M": "32"}

    cfg = cli(DynamicConfig, argv=["--BLOCK_M=32"])
    assert cfg.model_extra == {"BLOCK_M": "32"}
    assert cfg.BLOCK_M == "32"

    cfg = cli(DynamicConfig, argv=["--args.BLOCK_N_OUT=128,256"])
    assert cfg.args == {"BLOCK_N_OUT": ["128", "256"]}

    cfg = cli(DynamicConfig, argv=["--BLOCK_N_OUT=128,256"])
    assert cfg.model_extra == {"BLOCK_N_OUT": ["128", "256"]}

    cfg = cli(StrictConfig, argv=["--args.BLOCK_M=32"])
    assert cfg.args == {"BLOCK_M": "32"}

    assert cli(Config, argv=["--nonexistent", "val"]) == Config()
