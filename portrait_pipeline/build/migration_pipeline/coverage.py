"""Independent extraction augmentation and fail-closed coverage checks.

The imported cjx extractor is intentionally conservative.  This module uses a
second AST pass with different matching rules so that a newly introduced
configuration spelling cannot silently disappear before Stage 1.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
_NON_PRODUCTION = re.compile(
    r"(?:^|/)(?:tests?|testing|benchmarks?|examples?)(?:/|$)", re.I
)

# These controls are consumed below vLLM by torch_npu, CANN, HCCL or the
# Mooncake Ascend transport.  Consequently they cannot be discovered by a
# Python-only AST walk, but the pinned vllm-ascend documentation explicitly
# presents them as inference performance/reliability settings.  Keep this list
# narrow: network-interface, rank, IP, library-path and build variables are
# deployment wiring, not tunable axes.
_DOCUMENTED_PERFORMANCE_ENVS = {
    "ASCEND_BUFFER_POOL",
    "ASCEND_CONNECT_TIMEOUT",
    "ASCEND_TRANSFER_TIMEOUT",
    "CPU_AFFINITY_CONF",
    "HCCL_BUFFERSIZE",
    "HCCL_BUFFSIZE",
    "HCCL_CONNECT_TIMEOUT",
    "HCCL_INTRA_PCIE_ENABLE",
    "HCCL_INTRA_ROCE_ENABLE",
    "HCCL_OP_EXPANSION_MODE",
    "HCCL_RDMA_SL",
    "HCCL_RDMA_TC",
    "HCCL_RDMA_TIMEOUT",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "PYTORCH_NPU_ALLOC_CONF",
    "TASK_QUEUE_ENABLE",
}
_DOC_EXCLUDES = re.compile(
    r"(?:^|/)(?:release_notes\.md|community|contribution|support_matrix|_templates)(?:/|$)",
    re.I,
)
_DOC_EXPORT = re.compile(
    r"\bexport\s+([A-Z][A-Z0-9_]{2,})\s*=\s*([^\s#]+)", re.M
)


def _iter_python(root: Path) -> Iterable[Path]:
    yield from sorted(
        path for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and not _NON_PRODUCTION.search(path.relative_to(root).as_posix())
    )


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError):
        return "<computed>"
    if isinstance(value, tuple):
        return list(value)
    return value if value is None or isinstance(value, (str, int, float, bool, list, dict)) else "<computed>"


def _source(root: Path, path: Path, node: ast.AST, symbol: str, scope: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    line = int(getattr(node, "lineno", 1))
    start, end = max(0, line - 5), min(len(lines), line + 4)
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "line": line,
        "symbol": symbol,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "excerpt": "\n".join(f"{index + 1}: {lines[index]}" for index in range(start, end)),
        "repository": scope,
    }


def _category(name: str) -> str:
    value = name.lower().replace("-", "_")
    groups = (
        ("parallelism", ("parallel", "rank", "world_size", "dsa_cp")),
        ("memory", ("memory", "cache", "offload", "block_size", "nz")),
        ("scheduling", ("batch", "prefill", "decode", "scheduler", "token")),
        ("compilation", ("compile", "graph", "fusion", "kernel", "triton")),
        ("communication", ("comm", "hccl", "rdma", "allreduce")),
    )
    return next((label for label, tokens in groups if any(token in value for token in tokens)), "other")


def _entry(
    *, name: str, kind: str, scope: str, default: Any, value_type: str,
    source: dict[str, Any], write_path: list[str] | None = None,
    replaces_deprecated: str | None = None,
) -> dict[str, Any]:
    identity = f"{kind}:{name}"
    result: dict[str, Any] = {
        "id": "param." + hashlib.sha256(identity.encode()).hexdigest()[:16],
        "name": name,
        "type": kind,
        "scope": scope,
        "category": _category(name),
        "default": default,
        "value_type": value_type,
        "description": None,
        "source_locations": [source],
        "source_variants": [{
            "scope": scope,
            "default": default,
            "value_type": value_type,
            "write_path": write_path or [],
            "declared_path": write_path or [],
        }],
    }
    if write_path is not None:
        result["declared_path"] = write_path
        result["write_path"] = write_path
    if replaces_deprecated:
        result["replaces_deprecated"] = replaces_deprecated
    return result


def _documented_envs(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Return source evidence for allow-listed external runtime controls."""
    docs = root / "docs" / "source"
    found: dict[str, list[dict[str, Any]]] = {}
    if not docs.is_dir():
        return found
    for path in sorted(docs.rglob("*.md")):
        relative = path.relative_to(docs).as_posix()
        if _DOC_EXCLUDES.search(relative):
            continue
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeError:
            continue
        lines = content.splitlines()
        # Accept both executable `export X=...` snippets and prose/backtick
        # references.  The allow-list prevents arbitrary uppercase prose from
        # becoming a parameter.
        for name in sorted(_DOCUMENTED_PERFORMANCE_ENVS):
            matches = list(re.finditer(rf"(?<![A-Z0-9_]){re.escape(name)}(?![A-Z0-9_])", content))
            if not matches:
                continue
            match = matches[0]
            line = content.count("\n", 0, match.start()) + 1
            start, end = max(0, line - 4), min(len(lines), line + 3)
            source = {
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
                "line": line,
                "symbol": f"documentation({name})",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "excerpt": "\n".join(
                    f"{index + 1}: {lines[index]}" for index in range(start, end)
                ),
                "repository": "vllm-ascend",
                "evidence_kind": "documentation",
            }
            found.setdefault(name, []).append(source)
    return found


def _documented_default(name: str, sources: list[dict[str, Any]]) -> tuple[Any, str]:
    for source in sources:
        match = re.search(
            rf"\bexport\s+{re.escape(name)}\s*=\s*([^\s#]+)",
            str(source.get("excerpt", "")),
        )
        if match:
            value = match.group(1).strip("\"'")
            if re.fullmatch(r"-?\d+", value):
                return int(value), "int"
            if value.lower() in {"true", "false"}:
                return value.lower() == "true", "bool"
            return value, "str"
    return "<external-default>", "unknown"


def _fallback_default(parameter: dict[str, Any]) -> tuple[Any, str]:
    default = parameter.get("default")
    evidence = "\n".join(
        str(item.get("excerpt", "")) for item in parameter.get("source_locations", [])
    )
    if "bool(int(" in evidence:
        try:
            return bool(int(default)), "bool"
        except (TypeError, ValueError):
            return "<computed>", "bool"
    if "int(os.getenv" in evidence:
        try:
            return int(default), "int"
        except (TypeError, ValueError):
            return "<computed>", "int"
    return default, str(parameter.get("value_type") or "unknown")


def _parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "_coverage_parent", parent)


def _wrapped_type(node: ast.AST) -> str:
    parent = getattr(node, "_coverage_parent", None)
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name):
        if parent.func.id in {"bool", "int", "float", "str", "list"}:
            return parent.func.id
    return "unknown"


def augment_indirect_surfaces(
    extraction: dict[str, Any], vllm_root: Path, ascend_root: Path
) -> dict[str, Any]:
    """Add user-facing surfaces written with forms the base extractor misses."""
    parameters = list(extraction.get("parameters", []))
    existing = {(str(item.get("type")), str(item.get("name"))) for item in parameters}
    by_name = {str(item.get("name")): item for item in parameters}
    additions: list[dict[str, Any]] = []

    for scope, root in (("vllm", vllm_root), ("vllm-ascend", ascend_root)):
        for path in _iter_python(root):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            except (SyntaxError, UnicodeError):
                continue
            _parents(tree)
            for node in ast.walk(tree):
                # Preferred additional_config replacements hidden behind a helper.
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_get_config_value"
                    and len(node.args) >= 3
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                    and isinstance(node.args[2], ast.Constant)
                    and isinstance(node.args[2].value, str)
                ):
                    key, env_key = node.args[1].value, node.args[2].value
                    name = f"additional_config.{key}"
                    if ("nested", name) not in existing:
                        fallback = by_name.get(env_key)
                        default, value_type = (
                            _fallback_default(fallback) if fallback
                            else (_literal(node.args[3]) if len(node.args) > 3 else "<computed>", "unknown")
                        )
                        additions.append(_entry(
                            name=name, kind="nested", scope=scope,
                            default=default, value_type=value_type,
                            source=_source(root, path, node, name, scope),
                            write_path=["additional_config", key],
                            replaces_deprecated=env_key,
                        ))
                        existing.add(("nested", name))
                    continue

                # Read-only subscript access is a real input.  Store access can
                # be an internal CLI projection and must not create a duplicate.
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.ctx, ast.Load)
                    and "additional_config" in ast.unparse(node.value)
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                ):
                    key = node.slice.value
                    name = f"additional_config.{key}"
                    if ("nested", name) not in existing:
                        value_type = _wrapped_type(node)
                        default: Any = False if value_type == "bool" else "<computed>"
                        additions.append(_entry(
                            name=name, kind="nested", scope=scope,
                            default=default, value_type=value_type,
                            source=_source(root, path, node, name, scope),
                            write_path=["additional_config", key],
                        ))
                        existing.add(("nested", name))
                    continue

                # setdefault preserves a user-provided environment value, so it
                # is a public override even though it is not a conventional read.
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setdefault"
                    and ast.unparse(node.func.value) == "os.environ"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and _ENV_KEY.fullmatch(node.args[0].value)
                ):
                    key = node.args[0].value
                    if ("env", key) not in existing:
                        default = _literal(node.args[1] if len(node.args) > 1 else None)
                        additions.append(_entry(
                            name=key, kind="env", scope=scope,
                            default=default,
                            value_type=type(default).__name__ if default != "<computed>" else "unknown",
                            source=_source(root, path, node, key, scope),
                        ))
                        existing.add(("env", key))

    # External runtime controls documented by this exact source snapshot.  Add
    # all locations so Stage 2 can distinguish broad recommendations from
    # model/transport-specific settings instead of inferring from one snippet.
    for name, sources in _documented_envs(ascend_root).items():
        if ("env", name) in existing:
            parameter = next(
                item for item in parameters + additions
                if item.get("type") == "env" and item.get("name") == name
            )
            known = {
                (item.get("path"), item.get("line"))
                for item in parameter.get("source_locations", [])
            }
            parameter.setdefault("source_locations", []).extend(
                item for item in sources
                if (item.get("path"), item.get("line")) not in known
            )
            continue
        default, value_type = _documented_default(name, sources)
        additions.append(_entry(
            name=name,
            kind="env",
            scope="vllm-ascend-external",
            default=default,
            value_type=value_type,
            source=sources[0],
        ))
        additions[-1]["source_locations"] = sources
        additions[-1]["external_consumer"] = "torch_npu/CANN/HCCL/Mooncake"
        existing.add(("env", name))

    document = {key: value for key, value in extraction.items() if key != "extraction_hash"}
    document["parameters"] = sorted(
        parameters + additions,
        key=lambda item: (str(item.get("type")), str(item.get("name"))),
    )
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    document["extraction_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    return document


def audit_extraction_coverage(
    extraction: dict[str, Any], vllm_root: Path, ascend_root: Path
) -> dict[str, Any]:
    """Cross-check source syntax against extracted records and fail closed."""
    existing = {(str(item.get("type")), str(item.get("name"))) for item in extraction.get("parameters", [])}
    expected: dict[str, set[tuple[str, str]]] = {
        "cli": set(), "env": set(), "nested": set(), "config_fields": set(),
    }
    dynamic_cli: list[dict[str, Any]] = []
    parse_failures: list[str] = []

    for scope, root in (("vllm", vllm_root), ("vllm-ascend", ascend_root)):
        for path in _iter_python(root):
            relative = path.relative_to(root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            except (SyntaxError, UnicodeError):
                parse_failures.append(f"{scope}:{relative}")
                continue
            if "config" in relative.lower() or "arg_utils" in path.name:
                for cls in (item for item in ast.walk(tree) if isinstance(item, ast.ClassDef)):
                    for node in cls.body:
                        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                            expected["config_fields"].add(("nested", f"{cls.name}.{node.target.id}"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                ):
                    flags = [
                        arg.value for arg in node.args
                        if isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.startswith("--")
                    ]
                    if flags:
                        expected["cli"].add(("cli", flags[0]))
                    else:
                        dynamic_cli.append({"repository": scope, "path": relative, "line": node.lineno})
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.ctx, ast.Load)
                    and ast.unparse(node.value) == "os.environ"
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                    and _ENV_KEY.fullmatch(node.slice.value)
                ):
                    expected["env"].add(("env", node.slice.value))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    receiver = ast.unparse(node.func.value)
                    key = (
                        node.args[0].value if node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str) else None
                    )
                    if (
                        key and _ENV_KEY.fullmatch(key)
                        and (
                            node.func.attr in {"get", "getenv"}
                            and (receiver == "os" or "environ" in receiver)
                            or node.func.attr == "setdefault" and receiver == "os.environ"
                        )
                    ):
                        expected["env"].add(("env", key))
                    if key and node.func.attr == "get" and "additional_config" in receiver:
                        expected["nested"].add(("nested", f"additional_config.{key}"))
                    if (
                        node.func.attr == "_get_config_value" and len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                    ):
                        expected["nested"].add(("nested", f"additional_config.{node.args[1].value}"))
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.ctx, ast.Load)
                    and "additional_config" in ast.unparse(node.value)
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                ):
                    expected["nested"].add(("nested", f"additional_config.{node.slice.value}"))

    documented_envs = _documented_envs(ascend_root)
    expected["documented_external_env"] = {
        ("env", name) for name in documented_envs
    }

    missing = {
        family: sorted(name for kind, name in values if (kind, name) not in existing)
        for family, values in expected.items()
    }
    errors = []
    for family, names in missing.items():
        if names:
            errors.append(f"{family}: {len(names)} source-backed surfaces are missing")
    if parse_failures:
        errors.append(f"{len(parse_failures)} production Python files could not be parsed")
    if dynamic_cli:
        errors.append(f"{len(dynamic_cli)} dynamic add_argument calls require explicit extraction support")
    return {
        "schema_version": "extraction-coverage/v1",
        "ok": not errors,
        "expected_counts": {key: len(value) for key, value in expected.items()},
        "missing": missing,
        "dynamic_cli_calls": dynamic_cli,
        "parse_failures": parse_failures,
        "documented_external_envs": sorted(documented_envs),
        "errors": errors,
    }
