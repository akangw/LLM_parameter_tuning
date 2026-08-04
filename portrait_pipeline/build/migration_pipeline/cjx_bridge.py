"""Bridge the audited cjx_space extractor into the isolated migration run.

The rich document is retained for Stage-1 and migration auditing.  A narrow
projection is produced only at the boundary to the unchanged ParameterYAML
analyzer, whose input contract predates structured source locations.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import sys
from pathlib import Path

from .coverage import audit_extraction_coverage, augment_indirect_surfaces


def _fallback_default(parameter: dict) -> tuple[object, str]:
    """Convert a deprecated env fallback to the effective config-key type."""
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


def _augment_config_value_fallbacks(extraction: dict, ascend_root: Path) -> dict:
    """Promote `_get_config_value` replacement keys into real candidates.

    vllm-ascend is migrating several deprecated environment variables to
    `additional_config`.  The reviewed cjx extractor handles direct
    `additional_config.get(...)` calls but not this helper form, which caused
    the supported replacement surface to disappear from the portrait queue.
    """
    parameters = list(extraction.get("parameters", []))
    existing = {str(item.get("name")) for item in parameters}
    by_name = {str(item.get("name")): item for item in parameters}
    path = ascend_root / "vllm_ascend" / "ascend_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    additions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 3:
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "_get_config_value":
            continue
        key_node, env_node = node.args[1], node.args[2]
        if not (
            isinstance(key_node, ast.Constant)
            and isinstance(key_node.value, str)
            and isinstance(env_node, ast.Constant)
            and isinstance(env_node.value, str)
        ):
            continue
        name = f"additional_config.{key_node.value}"
        if name in existing:
            continue
        fallback = by_name.get(env_node.value)
        if fallback is None:
            continue
        default, value_type = _fallback_default(fallback)
        start, end = max(0, node.lineno - 5), min(len(lines), node.lineno + 4)
        identity = f"nested:{name}"
        additions.append({
            "id": "param." + hashlib.sha256(identity.encode()).hexdigest()[:16],
            "name": name,
            "type": "nested",
            "scope": "vllm-ascend",
            "category": fallback.get("category", "other"),
            "default": default,
            "value_type": value_type,
            "description": (
                f"Preferred additional_config replacement for deprecated "
                f"{env_node.value}."
            ),
            "source_locations": [{
                "path": path.resolve().relative_to(ascend_root.resolve()).as_posix(),
                "line": node.lineno,
                "symbol": f"additional_config.{key_node.value}",
                "sha256": source_hash,
                "excerpt": "\n".join(
                    f"{index + 1}: {lines[index]}" for index in range(start, end)
                ),
                "repository": "vllm-ascend",
            }],
            "declared_path": ["additional_config", key_node.value],
            "write_path": ["additional_config", key_node.value],
            "source_variants": [{
                "scope": "vllm-ascend",
                "default": default,
                "value_type": value_type,
                "write_path": ["additional_config", key_node.value],
                "declared_path": ["additional_config", key_node.value],
            }],
            "replaces_deprecated": env_node.value,
        })
        existing.add(name)
    document = {
        key: value for key, value in extraction.items() if key != "extraction_hash"
    }
    document["parameters"] = sorted(
        parameters + additions,
        key=lambda item: (str(item.get("type")), str(item.get("name"))),
    )
    payload = __import__("json").dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    document["extraction_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    return document


def extract_and_filter(cjx_root: Path, vllm_root: Path, ascend_root: Path) -> tuple[dict, dict]:
    root = str(cjx_root.resolve())
    if not (cjx_root / "cjx_pipeline" / "extract.py").is_file():
        raise SystemExit(f"cjx_space extractor not found: {cjx_root / 'cjx_pipeline'}")
    if root not in sys.path:
        sys.path.insert(0, root)
    # Import the reviewed implementation itself: no reduced reimplementation
    # and no silent policy drift between the 415-candidate baseline and this run.
    extractor = importlib.import_module("cjx_pipeline.extract")
    analyzer = importlib.import_module("cjx_pipeline.analyze")
    extraction = extractor.extract_parameters(vllm_root, ascend_root)
    # Retain the original focused augmentation for backward-compatible tests,
    # then apply the generic independent pass for subscript and environment
    # spellings.  The second pass is idempotent by (type, name).
    extraction = _augment_config_value_fallbacks(extraction, ascend_root)
    extraction = augment_indirect_surfaces(extraction, vllm_root, ascend_root)
    # The generic environment classifier treats many diagnostic-looking caps
    # as observability.  This one is consumed as an actual float32 logits
    # workspace budget by sparse-indexer prefill, so preserve the correct
    # performance category in every downstream artifact.
    for parameter in extraction["parameters"]:
        if parameter.get("name") == "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB":
            parameter["category"] = "memory"
    coverage = audit_extraction_coverage(extraction, vllm_root, ascend_root)
    extraction["coverage_audit"] = coverage
    payload = json.dumps(
        {key: value for key, value in extraction.items() if key != "extraction_hash"},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    extraction["extraction_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    if not coverage["ok"]:
        raise SystemExit(
            "Extraction coverage audit failed:\n"
            + "\n".join(f"- {item}" for item in coverage["errors"])
            + "\n"
            + json.dumps(coverage["missing"], ensure_ascii=False, indent=2)
        )
    # Paths are normally operational controls, but these two are exceptions:
    # they select/record the actual MoE expert placement and therefore change
    # EPLB activation and inference performance.  The 0706 portraits and the
    # pinned Ascend implementation both confirm that they require Stage-2
    # review even though a free-form path will not become a numeric search axis.
    hard_keep = list(analyzer.DEFAULT_FILTER["hard_keep_name_patterns"]) + [
        # This is a real high-impact memory lifecycle switch (and was a high
        # impact portrait in 0706), not an incidental operational "mode".
        r"^--enable-sleep-mode$",
        # The generic name-only filter cannot infer these consumers.  The
        # Three scale constants feed sparse-attention dynamic quantization,
        # the logits cap controls sparse-indexer prefill chunking/memory, and
        # the media loader count sizes a real request preprocessing thread
        # pool. Stage 2 should decide their impact.
        r"^(?:Q_SCALE_CONSTANT|K_SCALE_CONSTANT|V_SCALE_CONSTANT)$",
        r"^VLLM_SPARSE_INDEXER_MAX_LOGITS_MB$",
        r"^VLLM_MEDIA_LOADING_THREAD_COUNT$",
        r"^additional_config\.eplb_config\.expert_map_(?:path|record_path)$",
        # Debug-oriented, but enabling it disables ACL graph execution.  That
        # makes it a high-impact incompatibility which must be understood by
        # the portrait/tag layer even though it should never be searched on.
        r"^ASCEND_LAUNCH_BLOCKING$",
        # Documented torch_npu/CANN controls are consumed outside the two
        # Python repositories, but directly change dispatch queueing and host
        # scheduling.  They must reach Stage 2 rather than disappear as
        # generic environment variables.
        r"^(?:TASK_QUEUE_ENABLE|CPU_AFFINITY_CONF|OMP_NUM_THREADS|OMP_PROC_BIND)$",
        r"^ASCEND_(?:BUFFER_POOL|CONNECT_TIMEOUT|TRANSFER_TIMEOUT)$",
    ]
    return extraction, analyzer.stage1_filter(
        extraction, policy={"hard_keep_name_patterns": hard_keep}
    )


def legacy_projection(parameters: list[dict]) -> list[dict]:
    """Convert selected structured entries to the original Stage-2 input shape."""
    projected = []
    for item in parameters:
        locations = item.get("source_locations") or []
        source_file = str(locations[0].get("path", "")) if locations else ""
        projected.append({
            "name": item["name"], "type": item["type"],
            "category": item.get("category", "other"),
            "default": item.get("default"), "description": item.get("description"),
            "scope": item.get("scope", "vllm"), "source_file": source_file,
            # Extra evidence is harmless to the old analyzer and is retained in
            # reports/provenance even though its YAML schema does not expose it.
            "value_type": item.get("value_type", "unknown"),
            "source_locations": locations, "structured_id": item.get("id"),
            "replaces_deprecated": item.get("replaces_deprecated"),
        })
    return projected
