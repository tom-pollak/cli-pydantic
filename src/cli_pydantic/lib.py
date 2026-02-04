from collections import deque
from typing import get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined


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

    lines = []
    for name, field in model.model_fields.items():
        key = f"{prefix}{name}"
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            lines.extend(model_help(ann, prefix=f"{key}."))
        else:
            default = (
                "required" if field.default is PydanticUndefined else field.default
            )
            desc = f"  {field.description}" if field.description else ""
            lines.append(f"  --{key}  {ty_name(ann)} (default: {default}){desc}")
    return lines

