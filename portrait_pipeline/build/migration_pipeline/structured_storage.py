from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .structured_errors import PipelineError


def ensure_json_value(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PipelineError(f"{path}: non-finite number")
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            ensure_json_value(item, f"{path}[{index}]")
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PipelineError(f"{path}: mapping keys must be strings")
            ensure_json_value(item, f"{path}.{key}")
        return value
    raise PipelineError(f"{path}: {type(value).__name__} is outside the JSON data model")


def canonical_bytes(value: Any) -> bytes:
    ensure_json_value(value)
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def logical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_document(path: Path) -> Any:
    if not path.is_file():
        raise PipelineError(f"missing file: {path}")
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, yaml.YAMLError, UnicodeError) as exc:
        raise PipelineError(f"invalid document {path}: {exc}") from exc
    return ensure_json_value(value, str(path))


def atomic_write(path: Path, value: Any) -> None:
    ensure_json_value(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        content = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    else:
        content = yaml.safe_dump(value, allow_unicode=True, sort_keys=False, width=120).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def with_hash(document: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(document)
    result.pop(key, None)
    result[key] = logical_hash(result)
    return result


def verify_hash(document: dict[str, Any], key: str) -> bool:
    actual = document.get(key)
    material = dict(document)
    material.pop(key, None)
    return isinstance(actual, str) and actual == logical_hash(material)
