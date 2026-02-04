"""Tests for cli_pydantic."""

import json

import pytest
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
    monkeypatch.setattr("sys.argv", ["prog", "--nonexistent", "val"])
    with pytest.raises(ValueError, match="Unknown option"):
        cli(Config)

    monkeypatch.setattr("sys.argv", ["prog", "-x"])
    with pytest.raises(ValueError, match="Expected --key"):
        cli(Config)

    monkeypatch.setattr("sys.argv", ["prog", str(tmp_path / "nope.yaml")])
    with pytest.raises(ValueError, match="Config file not found"):
        cli(Config)

    toml = tmp_path / "config.toml"
    toml.write_text("x = 1")
    monkeypatch.setattr("sys.argv", ["prog", str(toml)])
    with pytest.raises(ValueError, match="Unsupported config file type"):
        cli(Config)
