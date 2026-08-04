"""Deterministic parameter extraction inspired by the reference knowledge base."""
from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .structured_errors import PipelineError
from .structured_storage import logical_hash, with_hash


_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return "<computed>"
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value, key=repr)
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return value if value is None or isinstance(value, (str, int, float, bool, list)) else "<computed>"


def _annotation(node: ast.AST | None) -> str:
    if node is None:
        return "unknown"
    try:
        return ast.unparse(node)
    except Exception:
        return "unknown"


def _iter_python(root: Path) -> Iterable[Path]:
    if root.is_dir():
        yield from sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _read_tree(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (SyntaxError, UnicodeError):
        return None


def _source(root: Path, path: Path, line: int, symbol: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    start, end = max(0, line - 5), min(len(lines), line + 4)
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "line": line,
        "symbol": symbol,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "excerpt": "\n".join(f"{index + 1}: {lines[index]}" for index in range(start, end)),
    }


def _category(name: str) -> str:
    value = name.lower().replace("-", "_")
    groups = [
        ("parallelism", ("parallel", "rank", "world_size", "all2all", "all_reduce")),
        ("memory", ("memory", "cache", "offload", "block_size")),
        ("scheduling", ("batch", "prefill", "decode", "scheduler", "num_seqs", "token")),
        ("compilation", ("compile", "cudagraph", "fusion", "kernel", "graph")),
        ("model", ("model", "dtype", "quant", "lora", "speculative")),
        ("communication", ("hccl", "comm", "rdma", "network")),
        ("observability", ("log", "trace", "metric", "profile", "debug")),
    ]
    return next((label for label, tokens in groups if any(token in value for token in tokens)), "other")


def _entry(name: str, kind: str, scope: str, default: Any, value_type: str, source: dict[str, Any], **extra: Any) -> dict[str, Any]:
    identity = f"{kind}:{name}"
    source = dict(source)
    source["repository"] = scope
    result = {
        "id": "param." + hashlib.sha256(identity.encode()).hexdigest()[:16],
        "name": name,
        "type": kind,
        "scope": scope,
        "category": _category(name),
        "default": default,
        "value_type": value_type,
        "description": extra.pop("description", None),
        "source_locations": [source],
    }
    result.update(extra)
    return result


def _production_source(item: dict[str, Any]) -> bool:
    pattern = re.compile(r"(?:^|/)(?:tests?|testing|benchmarks?|examples?)(?:/|$)", re.I)
    return any(not pattern.search(str(source.get("path", ""))) for source in item.get("source_locations", []))


def _entry_score(item: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        1 if _production_source(item) else 0,
        1 if item.get("description") else 0,
        1 if item.get("value_type") not in {None, "unknown", "NoneType"} else 0,
        1 if item.get("default") is not None and item.get("default") != "<computed>" else 0,
    )


def _scope_set(value: str | None) -> set[str]:
    return {"vllm", "vllm-ascend"} if value == "both" else ({value} if value else set())


def _merge(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in entries:
        item = dict(item)
        item["source_variants"] = [{
            key: item.get(key) for key in (
                "scope", "default", "value_type", "choices", "aliases", "action", "nargs", "write_path", "declared_path"
            ) if key in item
        }]
        key = (item["type"], item["name"])
        current = merged.get(key)
        if current is None:
            merged[key] = dict(item)
            continue
        winner = item if _entry_score(item) > _entry_score(current) else current
        scopes = _scope_set(current.get("scope")) | _scope_set(item.get("scope"))
        combined = dict(winner)
        combined["scope"] = next(iter(scopes)) if len(scopes) == 1 else "both"
        combined["source_locations"] = current.get("source_locations", []) + item.get("source_locations", [])
        variants = current.get("source_variants", []) + item.get("source_variants", [])
        combined["source_variants"] = list({logical_hash(value): value for value in variants}.values())
        merged[key] = combined
    for item in merged.values():
        unique = {logical_hash(source): source for source in item["source_locations"]}
        item["source_locations"] = list(unique.values())
    return _enrich_cli_metadata(sorted(merged.values(), key=lambda item: (item["type"], item["name"])))


def _literal_choices(value_type: str) -> list[Any] | None:
    match = re.search(r"Literal\[(.*?)\]", value_type)
    if not match:
        return None
    try:
        values = ast.literal_eval(f"({match.group(1)},)")
    except (SyntaxError, ValueError):
        return None
    return list(values) if isinstance(values, tuple) else None


def _enrich_cli_metadata(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nested: dict[str, list[dict[str, Any]]] = {}
    for item in entries:
        if item.get("type") == "nested":
            nested.setdefault(str(item["name"]).rsplit(".", 1)[-1], []).append(item)
    for item in entries:
        if item.get("type") != "cli":
            continue
        leaf = str(item["name"]).lstrip("-").replace("-", "_")
        # Only infer through the exact **kwargs["field"] pattern emitted by the
        # current argument builder; name similarity alone is insufficient.
        excerpts = "\n".join(str(source.get("excerpt", "")) for source in item.get("source_locations", []))
        if f'["{leaf}"]' not in excerpts and f"['{leaf}']" not in excerpts:
            continue
        candidates = [candidate for candidate in nested.get(leaf, []) if _production_source(candidate)]
        if not candidates:
            continue
        concrete_defaults = {
            repr(candidate.get("default")): candidate.get("default")
            for candidate in candidates
            if candidate.get("default") is not None and candidate.get("default") != "<computed>"
        }
        concrete_types = {
            str(candidate.get("value_type"))
            for candidate in candidates if candidate.get("value_type") not in {None, "unknown", "NoneType"}
        }
        if item.get("default") is None and len(concrete_defaults) == 1:
            item["default"] = next(iter(concrete_defaults.values()))
        if item.get("value_type") in {None, "unknown"} and len(concrete_types) == 1:
            item["value_type"] = next(iter(concrete_types))
        if item.get("choices") is None:
            choices = _literal_choices(str(item.get("value_type", "")))
            if choices:
                item["choices"] = choices
        linked_sources = [source for candidate in candidates for source in candidate.get("source_locations", [])]
        unique = {logical_hash(source): source for source in item.get("source_locations", []) + linked_sources}
        item["source_locations"] = list(unique.values())
    return entries


def _scan_cli(root: Path, scope: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in _iter_python(root):
        tree = _read_tree(path)
        if tree is None:
            continue
        kwargs_classes: dict[str, str] = {}
        group_classes: dict[str, str] = {}
        for assignment in ast.walk(tree):
            if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                continue
            target = (
                assignment.targets[0]
                if isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
                else assignment.target if isinstance(assignment, ast.AnnAssign) else None
            )
            value = assignment.value
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "get_kwargs"
                and value.args
                and isinstance(value.args[0], ast.Name)
            ):
                kwargs_classes[target.id] = value.args[0].id
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "add_argument_group"
            ):
                title = next(
                    (
                        keyword.value.value
                        for keyword in value.keywords
                        if keyword.arg == "title"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ),
                    None,
                )
                if title and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", title):
                    group_classes[target.id] = title
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            option_strings = [
                arg.value for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-")
            ]
            primary = next((value for value in option_strings if value.startswith("--")), None)
            if primary is None:
                continue
            kwargs = {item.arg: item.value for item in node.keywords if item.arg}
            mirror_target = None
            for keyword in node.keywords:
                value = keyword.value
                if keyword.arg is not None or not isinstance(value, ast.Subscript):
                    continue
                if not isinstance(value.value, ast.Name) or value.value.id not in kwargs_classes:
                    continue
                field = (
                    value.slice.value
                    if isinstance(value.slice, ast.Constant) and isinstance(value.slice.value, str)
                    else None
                )
                if field:
                    mirror_target = f"{kwargs_classes[value.value.id]}.{field}"
                    break
            if (
                mirror_target is None
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in group_classes
            ):
                mirror_target = (
                    f"{group_classes[node.func.value.id]}."
                    f"{primary.lstrip('-').replace('-', '_')}"
                )
            results.append(_entry(
                primary, "cli", scope, _literal(kwargs.get("default")), _annotation(kwargs.get("type")),
                _source(root, path, node.lineno, "add_argument"), choices=_literal(kwargs.get("choices")),
                description=_literal(kwargs.get("help")), aliases=[value for value in option_strings if value != primary],
                action=_literal(kwargs.get("action")),
                nargs=_literal(kwargs.get("nargs")),
                mirror_target=mirror_target,
            ))
    return results


def _scan_config_fields(root: Path, scope: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in _iter_python(root):
        if "config" not in path.as_posix().lower() and "arg_utils" not in path.name:
            continue
        tree = _read_tree(path)
        if tree is None:
            continue
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            for node in cls.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    name = f"{cls.name}.{node.target.id}"
                    results.append(_entry(name, "nested", scope, _literal(node.value), _annotation(node.annotation), _source(root, path, node.lineno, name), declared_path=[cls.name, node.target.id], write_path=[]))
    return results


def _scan_declared_nested_fields(root: Path, scope: str, max_depth: int = 1) -> list[dict[str, Any]]:
    """Expand typed config objects such as CompilationConfig.pass_config."""
    classes: dict[str, tuple[Path, ast.ClassDef]] = {}
    fields: dict[str, list[ast.AnnAssign]] = {}
    public_config_keys: set[str] = set()
    for path in _iter_python(root):
        tree = _read_tree(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            for arg in node.args:
                if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                    continue
                flag = arg.value
                if flag.startswith("--") and flag.endswith("-config"):
                    public_config_keys.add(flag[2:].replace("-", "").lower())
        if "config" not in path.as_posix().lower() and "arg_utils" not in path.name:
            continue
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            classes[cls.name] = (path, cls)
            fields[cls.name] = [
                node for node in cls.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            ]

    results: list[dict[str, Any]] = []

    def expand(
        root_class: str, current_class: str, name_parts: list[str], declared_parts: list[str], depth: int,
    ) -> None:
        if depth > max_depth:
            return
        for field in fields.get(current_class, []):
            field_name = field.target.id
            path = classes[current_class][0]
            full_name = ".".join([root_class] + name_parts + [field_name])
            results.append(_entry(
                full_name, "nested", scope, _literal(field.value), _annotation(field.annotation),
                _source(root, path, field.lineno, f"{current_class}.{field_name}"),
                declared_path=declared_parts + [current_class, field_name], write_path=[],
            ))
            referenced = [
                token for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", _annotation(field.annotation))
                if token in classes and token != current_class
            ]
            if len(referenced) == 1:
                expand(
                    root_class, referenced[0], name_parts + [field_name],
                    declared_parts + [current_class, field_name], depth + 1,
                )

    public_roots = {
        class_name for class_name in classes
        if class_name.lower() in public_config_keys
    }
    for class_name in sorted(public_roots):
        for field in fields.get(class_name, []):
            # SkipValidation config objects are injected by the engine or built
            # during post-init.  Their children are runtime state mirrors, not
            # values accepted by the public ``--*-config`` dictionary.
            if "SkipValidation" in _annotation(field.annotation):
                continue
            referenced = [
                token for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", _annotation(field.annotation))
                if token in classes and token != class_name
            ]
            if len(referenced) == 1:
                expand(class_name, referenced[0], [field.target.id], [class_name, field.target.id], 1)
    return results


def _assigned_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _dict_get(node: ast.AST | None) -> tuple[str, str, ast.AST | None] | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "get" or not node.args:
        return None
    key = node.args[0].value if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) else None
    if key is None:
        return None
    return ast.unparse(node.func.value), key, node.args[1] if len(node.args) > 1 else None


def _value_type(annotation: ast.AST | None, value: ast.AST | None, default: Any) -> str:
    declared = _annotation(annotation)
    if declared != "unknown":
        return declared
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in {"bool", "float", "int", "list", "str"}:
        return value.func.id
    return type(default).__name__ if default is not None and default != "<computed>" else "unknown"


def _scan_additional_config_leaves(root: Path, scope: str) -> list[dict[str, Any]]:
    """Expand nested dictionaries that are deterministically rooted at additional_config.

    Ascend builds several typed helpers from dictionaries obtained through
    ``additional_config.get(...)``.  The public knobs live in those helpers'
    constructor arguments, ``dict.get`` calls, and ``_defaults`` dictionaries,
    not in the top-level container itself.
    """
    results: list[dict[str, Any]] = []
    for path in _iter_python(root):
        tree = _read_tree(path)
        if tree is None:
            continue
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        aliases: dict[str, list[str]] = {}
        class_roots: dict[str, tuple[list[str], bool]] = {}

        # Discover local dictionaries obtained from additional_config and the
        # helper classes constructed from those dictionaries.
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            target_node = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else (
                node.target if isinstance(node, ast.AnnAssign) else None
            )
            target = _assigned_name(target_node) if target_node is not None else None
            value = node.value
            if target is None or value is None:
                continue
            access = _dict_get(value)
            if access and access[0].endswith("additional_config"):
                aliases[target] = ["additional_config", access[1]]
                continue
            if not isinstance(value, ast.Call):
                continue
            class_name = value.func.id if isinstance(value.func, ast.Name) else None
            if class_name not in classes:
                continue
            unpacked_alias = next(
                (
                    aliases[item.value.id] for item in value.keywords
                    if item.arg is None and isinstance(item.value, ast.Name) and item.value.id in aliases
                ),
                None,
            )
            positional_alias = next(
                (aliases[item.id] for item in value.args if isinstance(item, ast.Name) and item.id in aliases),
                None,
            )
            if unpacked_alias:
                class_roots[class_name] = (unpacked_alias, True)
            elif positional_alias:
                class_roots[class_name] = (positional_alias, False)

        for class_name, (write_root, unpacked) in class_roots.items():
            cls = classes[class_name]
            init = next(
                (node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"),
                None,
            )
            carrier_names = {"kwargs"} if unpacked else set()
            if init is not None:
                positional = list(init.args.posonlyargs) + list(init.args.args)
                non_self = [item for item in positional if item.arg != "self"]
                if not unpacked and non_self:
                    carrier_names.add(non_self[0].arg)
                defaults = [None] * (len(positional) - len(init.args.defaults)) + list(init.args.defaults)
                for arg, default_node in zip(positional, defaults):
                    if arg.arg == "self" or arg.arg in carrier_names or arg.arg in {"vllm_config"}:
                        continue
                    default = _literal(default_node)
                    name = ".".join(write_root + [arg.arg])
                    results.append(_entry(
                        name, "nested", scope, default, _annotation(arg.annotation),
                        _source(root, path, arg.lineno, f"{class_name}.{arg.arg}"),
                        declared_path=[class_name, arg.arg], write_path=write_root + [arg.arg],
                    ))

            for node in cls.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    name = ".".join(write_root + [node.target.id])
                    results.append(_entry(
                        name, "nested", scope, _literal(node.value), _annotation(node.annotation),
                        _source(root, path, node.lineno, f"{class_name}.{node.target.id}"),
                        declared_path=[class_name, node.target.id], write_path=write_root + [node.target.id],
                    ))
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    target_node = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else (
                        node.target if isinstance(node, ast.AnnAssign) else None
                    )
                    if _assigned_name(target_node) != "_defaults":
                        continue
                    defaults_node = node.value
                    if isinstance(defaults_node, ast.Dict):
                        for key_node, value_node in zip(defaults_node.keys, defaults_node.values):
                            key = key_node.value if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str) else None
                            if key is None:
                                continue
                            default = _literal(value_node)
                            name = ".".join(write_root + [key])
                            results.append(_entry(
                                name, "nested", scope, default, type(default).__name__,
                                _source(root, path, key_node.lineno, f"{class_name}._defaults[{key}]"),
                                declared_path=[class_name, "_defaults", key], write_path=write_root + [key],
                            ))

            for node in ast.walk(cls):
                access = _dict_get(node)
                if access is None:
                    continue
                receiver, key, default_node = access
                if receiver not in carrier_names:
                    continue
                default = _literal(default_node)
                parent = node.parent if hasattr(node, "parent") else None
                annotation = parent.annotation if isinstance(parent, ast.AnnAssign) else None
                name = ".".join(write_root + [key])
                results.append(_entry(
                    name, "nested", scope, default, _value_type(annotation, parent, default),
                    _source(root, path, node.lineno, f"{class_name}.{receiver}.get({key})"),
                    declared_path=[class_name, key], write_path=write_root + [key],
                ))
    return results


def _scan_env_registries(root: Path, scope: str) -> list[dict[str, Any]]:
    """Extract keys from the authoritative lazy environment registries.

    Helper-backed entries may have no direct ``os.getenv("LITERAL")`` call,
    so the registry key itself must be treated as the source of truth.
    """
    results: list[dict[str, Any]] = []
    registry_names = {"environment_variables", "env_variables"}
    for path in _iter_python(root):
        tree = _read_tree(path)
        if tree is None:
            continue
        declarations = {
            node.target.id: node
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and _ENV_KEY.fullmatch(node.target.id)
        }
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            target = (
                node.targets[0]
                if isinstance(node, ast.Assign) and len(node.targets) == 1
                else node.target if isinstance(node, ast.AnnAssign) else None
            )
            if not isinstance(target, ast.Name) or target.id not in registry_names:
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            for key_node, value_node in zip(node.value.keys, node.value.values):
                key = (
                    key_node.value
                    if isinstance(key_node, ast.Constant)
                    and isinstance(key_node.value, str)
                    and _ENV_KEY.fullmatch(key_node.value)
                    else None
                )
                if key is None:
                    continue
                declaration = declarations.get(key)
                value_type = _annotation(declaration.annotation) if declaration else "unknown"
                default = _literal(declaration.value) if declaration else None
                choices = _literal_choices(value_type)
                for call in (item for item in ast.walk(value_node) if isinstance(item, ast.Call)):
                    function_name = (
                        call.func.id if isinstance(call.func, ast.Name)
                        else call.func.attr if isinstance(call.func, ast.Attribute)
                        else ""
                    )
                    if function_name == "env_with_choices" and len(call.args) >= 2:
                        default = _literal(call.args[1])
                        if len(call.args) >= 3:
                            choices = _literal(call.args[2])
                        break
                    if function_name in {"get", "getenv"} and call.args:
                        accessed = (
                            call.args[0].value
                            if isinstance(call.args[0], ast.Constant)
                            and isinstance(call.args[0].value, str)
                            else None
                        )
                        if accessed == key and len(call.args) >= 2:
                            default = _literal(call.args[1])
                            break
                results.append(_entry(
                    key, "env", scope, default, value_type,
                    _source(root, path, key_node.lineno, f"{target.id}[{key}]"),
                    choices=choices,
                ))
    return results


def _scan_runtime_accesses(root: Path, scope: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in _iter_python(root):
        tree = _read_tree(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and ast.unparse(node.value) == "os.environ":
                key = node.slice.value if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) else None
                if key and _ENV_KEY.fullmatch(key):
                    results.append(_entry(
                        key, "env", scope, None, "unknown",
                        _source(root, path, node.lineno, f"environment[{key}]"),
                    ))
                continue
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or not node.args:
                continue
            key = node.args[0].value if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) else None
            if not key:
                continue
            receiver = ast.unparse(node.func.value)
            default = _literal(node.args[1] if len(node.args) > 1 else None)
            if node.func.attr in {"get", "getenv"} and ("environ" in receiver or receiver == "os") and _ENV_KEY.fullmatch(key):
                results.append(_entry(key, "env", scope, default, type(default).__name__, _source(root, path, node.lineno, f"environment({key})")))
            elif node.func.attr == "get" and "additional_config" in receiver:
                results.append(_entry(f"additional_config.{key}", "nested", scope, default, type(default).__name__, _source(root, path, node.lineno, f"additional_config.get({key})"), declared_path=["additional_config", key], write_path=["additional_config", key]))
            elif node.func.attr == "get" and "kv_connector_extra_config" in receiver:
                results.append(_entry(
                    f"KVTransferConfig.kv_connector_extra_config.{key}", "nested", scope, default, type(default).__name__,
                    _source(root, path, node.lineno, f"kv_connector_extra_config.get({key})"),
                    declared_path=["KVTransferConfig", "kv_connector_extra_config", key],
                    write_path=["kv_connector_extra_config", key],
                ))
            elif (
                node.func.attr == "get" and receiver.split(".")[-1] == "extra_config"
                and any(token in path.as_posix().lower() for token in ("kv_transfer", "kv_connector", "kv_offload"))
            ):
                results.append(_entry(
                    f"KVTransferConfig.kv_connector_extra_config.{key}", "nested", scope, default, type(default).__name__,
                    _source(root, path, node.lineno, f"connector.extra_config.get({key})"),
                    declared_path=["KVTransferConfig", "kv_connector_extra_config", key],
                    write_path=["kv_connector_extra_config", key],
                ))
            elif node.func.attr == "get_from_extra_config":
                lowered = receiver.lower()
                if "kv_transfer" in lowered:
                    class_name, field_name = "KVTransferConfig", "kv_connector_extra_config"
                elif "ec_transfer" in lowered:
                    class_name, field_name = "ECTransferConfig", "ec_connector_extra_config"
                else:
                    continue
                results.append(_entry(
                    f"{class_name}.{field_name}.{key}", "nested", scope, default, type(default).__name__,
                    _source(root, path, node.lineno, f"{class_name}.get_from_extra_config({key})"),
                    declared_path=[class_name, field_name, key], write_path=[field_name, key],
                ))
    return results


def _git_head(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PipelineError(f"source is not a readable git checkout: {root}") from exc


def extract_parameters(vllm_root: Path, ascend_root: Path) -> dict[str, Any]:
    vllm_root, ascend_root = vllm_root.resolve(), ascend_root.resolve()
    entries = (
        _scan_cli(vllm_root, "vllm")
        + _scan_config_fields(vllm_root, "vllm")
        + _scan_declared_nested_fields(vllm_root, "vllm")
        + _scan_additional_config_leaves(vllm_root, "vllm")
        + _scan_env_registries(vllm_root, "vllm")
        + _scan_runtime_accesses(vllm_root, "vllm")
        + _scan_cli(ascend_root, "vllm-ascend")
        + _scan_config_fields(ascend_root, "vllm-ascend")
        + _scan_declared_nested_fields(ascend_root, "vllm-ascend")
        + _scan_additional_config_leaves(ascend_root, "vllm-ascend")
        + _scan_env_registries(ascend_root, "vllm-ascend")
        + _scan_runtime_accesses(ascend_root, "vllm-ascend")
    )
    document = {
        "schema_version": "extracted-parameters/v1",
        "sources": {"vllm": {"path": str(vllm_root), "commit": _git_head(vllm_root)}, "vllm_ascend": {"path": str(ascend_root), "commit": _git_head(ascend_root)}},
        "parameters": _merge(entries),
    }
    return with_hash(document, "extraction_hash")
