import json
import sys
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

__all__ = ["cli", "CliError"]


class CliError(Exception):
    """Raised for invalid CLI flags, config files, or validation failures."""


class ResolvedField:
    def __init__(self, annotation: type | object, dynamic: bool = False):
        self.annotation = annotation
        self.dynamic = dynamic


def is_model_type(annotation: object) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def is_mapping_type(annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin is not None:
        return isinstance(origin, type) and issubclass(origin, Mapping)
    return annotation is dict


def mapping_value_type(annotation: object) -> object:
    args = get_args(annotation)
    return args[1] if len(args) == 2 else Any


def parse_scalar(value: str):
    if value == "":
        return value
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def resolve_field_type(
    model_cls: type[BaseModel], path: list[str]
) -> ResolvedField | None:
    """Walk dotted paths through BaseModels and mapping fields."""
    annotation: object = model_cls
    dynamic = False
    for p in path:
        if is_model_type(annotation):
            if p not in annotation.model_fields:
                return None
            annotation = annotation.model_fields[p].annotation
        elif is_mapping_type(annotation):
            annotation = mapping_value_type(annotation)
            dynamic = True
        else:
            return None
    return ResolvedField(annotation, dynamic=dynamic)


def parse_flags(tokens: list[str], model_cls: type[BaseModel]) -> dict:
    out = {}

    def route(key: str) -> tuple[list[str], ResolvedField]:
        parts = key.replace("-", "_").split(".")
        resolved = resolve_field_type(model_cls, parts)
        if resolved is None:
            raise CliError(f"Unknown option: --{key}")
        return parts, resolved

    def put(parts: list[str], resolved: ResolvedField, val):
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})

        k = parts[-1]
        is_list = get_origin(resolved.annotation) is list

        if is_list:
            vals = val.split(",") if isinstance(val, str) and "," in val else [val]
            vals = [parse_scalar(v) if isinstance(v, str) else v for v in vals]
            cur.setdefault(k, []).extend(vals)
        elif resolved.dynamic and isinstance(val, str) and "," in val:
            vals = [parse_scalar(v) for v in val.split(",")]
            cur.setdefault(k, []).extend(vals)
        elif cur.get(k, val) != val:
            raise CliError(f"Duplicate value for {'.'.join(parts)}")
        else:
            cur[k] = parse_scalar(val) if isinstance(val, str) else val

    def has_value() -> bool:
        return bool(q) and not q[0].startswith("--")

    q = deque(tokens)
    while q:
        t = q.popleft()
        if not t.startswith("--"):
            raise CliError(f"Expected --key, got: {t}")

        s = t[2:]

        if s.startswith("no-"):  # --no-flag
            if "=" in s or has_value():
                raise CliError(f"--no-* flags can't take a value: {t}")
            key, val = s[3:], False
        elif "=" in s:  # --k=v
            key, val = s.split("=", 1)
        else:  # --k v / --flag
            key, val = s, (q.popleft() if has_value() else True)

        parts, resolved = route(key)
        put(parts, resolved, val)

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
            if is_model_type(ann):
                out.extend(entries(ann, f"{key}."))
            elif is_mapping_type(ann):
                value_ann = mapping_value_type(ann)
                default = (
                    ""
                    if field.default is PydanticUndefined
                    else f" (default: {field.default})"
                )
                desc = f" {field.description}" if field.description else ""
                out.append((f"--{key}.<key> {ty_name(value_ann)}", f"{desc}{default}"))
            else:
                default = (
                    ""
                    if field.default is PydanticUndefined
                    else f" (default: {field.default})"
                )
                desc = f" {field.description}" if field.description else ""
                out.append((f"--{key} {ty_name(ann)}", f"{desc}{default}"))
        return out

    items = entries(model, prefix)
    col = max((len(f) for f, _ in items), default=0) + 1
    return [f"  {f:<{col}}{h}" for f, h in items]


def load_config(path: Path) -> dict:
    if not path.exists():
        raise CliError(f"Config file not found: {path}")

    raw = path.read_text()
    if not raw.strip():
        data = {}
    elif path.suffix == ".json":
        data = json.loads(raw)
    elif path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
    else:
        raise CliError(f"Unsupported config file type: {path.suffix}")

    if not isinstance(data, dict):
        raise CliError(f"Config file must contain a mapping, got {type(data).__name__}")
    return data


def cli[T: BaseModel](
    model_cls: type[T],
    desc: str = "",
    argv: list[str] | None = None,
    raise_on_error: bool = False,
) -> T:
    """Build a CLI from a Pydantic model, merging config files and --overrides.

    Positional arguments are paths to JSON/YAML config files (later files
    override earlier ones).  Any remaining ``--key value`` flags are parsed
    as field overrides using dot-notation (e.g. ``--model.lr 0.01``).

    Args:
        model_cls: The Pydantic model class that defines the config schema.
        desc: Optional description shown in ``--help`` output.
        argv: Raw CLI args. Default: ``sys.argv[1:]``.
        raise_on_error: If True, raise ``CliError`` on bad input.
            If False (default), print the error to stderr and exit.

    Returns:
        A validated instance of *model_cls*.
    """
    argv = argv or sys.argv[1:]

    def print_help():
        prog = Path(sys.argv[0]).name
        lines = []
        if desc:
            lines.append(f"help: {desc}\n")
        lines.append(f"usage: {prog} [-h] [configs ...] [--overrides ...]")
        lines.append("\nconfig arguments:")
        lines.extend(model_help(model_cls))
        print("\n".join(lines))
        raise SystemExit(0)

    def split_argv() -> tuple[list[Path], list[str]]:
        config_paths: list[Path] = []
        for i, tok in enumerate(argv):
            if tok.startswith("-"):
                return config_paths, argv[i:]
            config_paths.append(Path(tok))
        return config_paths, []

    if "-h" in argv or "--help" in argv:
        print_help()

    config_paths, flag_tokens = split_argv()

    try:
        configs = [load_config(p) for p in config_paths]
        overrides = parse_flags(flag_tokens, model_cls)
    except CliError:
        if raise_on_error:
            raise
        print(f"error: {sys.exc_info()[1]}", file=sys.stderr)
        raise SystemExit(1)

    data = {}
    for new in configs + [overrides]:
        deep_merge(data, new)

    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        items = [
            f"  {'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in e.errors()
        ]
        err = CliError("Validation failed:\n" + "\n".join(items))
        if raise_on_error:
            raise err from e
        print(err, file=sys.stderr)
        raise SystemExit(1)
