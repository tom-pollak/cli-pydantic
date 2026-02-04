import argparse
import json
from collections import deque
from pathlib import Path
from typing import get_origin

import yaml
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

__all__ = ["cli", "deep_merge", "model_help", "parse_unknown_args"]


def resolve_field_type(model_cls: type[BaseModel], path: list[str]) -> type | None:
    """Walk a dotted path through nested BaseModels, return the leaf annotation."""
    cls = model_cls
    for p in path:
        if p not in cls.model_fields:
            return None
        ann = cls.model_fields[p].annotation
        if p == path[-1]:
            return ann
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            cls = ann
        else:
            return None
    return None


def parse_unknown_args(tokens: list[str], model_cls: type[BaseModel]) -> dict:
    out = {}
    q = deque(tokens)

    def has_value() -> bool:
        return bool(q) and not q[0].startswith("--")

    def route(key: str) -> list[str]:
        parts = key.replace("-", "_").split(".")
        if resolve_field_type(model_cls, parts) is None:
            raise ValueError(f"Unknown option: --{key}")
        return parts

    def put(parts: list[str], val):
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})

        k = parts[-1]
        is_list = get_origin(resolve_field_type(model_cls, parts)) is list

        if is_list:
            vals = val.split(",") if isinstance(val, str) and "," in val else [val]
            cur.setdefault(k, []).extend(vals)
        elif cur.get(k, val) != val:
            raise ValueError(f"Duplicate value for {'.'.join(parts)}")
        else:
            cur[k] = val

    while q:
        t = q.popleft()
        if not t.startswith("--"):
            raise ValueError(f"Expected --key, got: {t}")

        s = t[2:]

        if s.startswith("no-"):  # --no-flag
            if "=" in s or has_value():
                raise ValueError(f"--no-* flags can't take a value: {t}")
            key, val = s[3:], False
        elif "=" in s:  # --k=v
            key, val = s.split("=", 1)
        else:  # --k v / --flag
            key, val = s, (q.popleft() if has_value() else True)

        put(route(key), val)

    return out


def deep_merge(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def model_help(model: type[BaseModel], prefix: str = "") -> list[str]:
    def ty_name(ann) -> str:
        name = getattr(ann, "__name__", None)
        return name if name and "[" not in str(ann) else str(ann)

    def entries(m, pfx):
        out = []
        for name, field in m.model_fields.items():
            key = f"{pfx}{name}"
            ann = field.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                out.extend(entries(ann, f"{key}."))
            else:
                default = (
                    ""
                    if field.default is PydanticUndefined
                    else f" (default: {field.default})"
                )
                desc = field.description or ""
                out.append((f"--{key} {ty_name(ann)}", f"{desc}{default}"))
        return out

    items = entries(model, prefix)
    col = max((len(f) for f, _ in items), default=0) + 2
    return [f"  {f:<{col}}{h}" for f, h in items]


def load_config(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"Config file not found: {path}")

    raw = path.read_text()
    if not raw.strip():
        data = {}
    elif path.suffix == ".json":
        data = json.loads(raw)
    elif path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
    else:
        raise ValueError(f"Unsupported config file type: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file must contain a mapping, got {type(data).__name__}"
        )
    return data


def cli[T: BaseModel](model_cls: type[T], desc: str = "") -> T:
    """Build a CLI from a Pydantic model, merging config files and --overrides.

    Positional arguments are paths to JSON/YAML config files (later files
    override earlier ones).  Any remaining ``--key value`` flags are parsed
    as field overrides using dot-notation (e.g. ``--model.lr 0.01``).

    Args:
        model_cls: The Pydantic model class that defines the config schema.
        desc: Optional description shown in ``--help`` output.

    Returns:
        A validated instance of *model_cls*.
    """
    parser = argparse.ArgumentParser(
        description=desc,
        usage="%(prog)s [-h] [configs ...] [--overrides ...]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="config arguments:\n" + "\n".join(model_help(model_cls)),
    )
    parser.add_argument(
        "config",
        nargs="*",
        default=[],
        type=Path,
        help="Path to config file(s); later files override earlier ones",
    )

    known, unknown = parser.parse_known_args()

    configs = [load_config(p) for p in known.config]
    overrides = parse_unknown_args(unknown, model_cls=model_cls)
    del known, unknown

    data = {}
    for new in configs + [overrides]:
        deep_merge(data, new)

    return model_cls.model_validate(data)
