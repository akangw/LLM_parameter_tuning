"""Manifest generation: build manifest.yaml summarizing the parse run."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml


def generate_manifest(
    total_params: int,
    stage1_passed: int,
    stage1_skipped: int,
    stage2_results: list[dict],
    output_dir: Path,
    logger: logging.Logger,
    source_commit_vllm: str = "",
    source_commit_vllm_ascend: str = "",
    builder_version: str = "1.0.0",
) -> Path:
    """Generate manifest.yaml summarizing the parse run.

    Args:
        total_params: Total number of input parameters.
        stage1_passed: Number passed to Stage 2.
        stage1_skipped: Number skipped in Stage 1.
        stage2_results: List of dicts with keys: name, performance_impact, status.
        output_dir: Output directory for manifest.yaml.
        logger: Logger instance.
        source_commit_vllm: vllm commit hash.
        source_commit_vllm_ascend: vllm-ascend commit hash.
        builder_version: Version string for the builder.

    Returns:
        Path to the generated manifest.yaml.
    """
    # Count by impact level and status
    impact_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
    status_counts = {"ok": 0, "error": 0}
    for r in stage2_results:
        impact = r.get("performance_impact", "none")
        if impact in impact_counts:
            impact_counts[impact] += 1
        status = r.get("status", "error")
        if status in status_counts:
            status_counts[status] += 1

    perf_related = impact_counts["high"] + impact_counts["medium"] + impact_counts["low"]
    not_perf = impact_counts["none"]

    manifest = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "builder_version": builder_version,
        "input_parameters_file": "parameters.json",
        "input_total_params": total_params,
        "stage1_coarse_filter": {
            "passed": stage1_passed,
            "skipped": stage1_skipped,
        },
        "stage2_deep_analysis": {
            "analyzed": len(stage2_results),
            "performance_high": impact_counts["high"],
            "performance_medium": impact_counts["medium"],
            "performance_low": impact_counts["low"],
            "confirmed_perf_related": perf_related,
            "confirmed_not_perf": not_perf,
            "errors": status_counts["error"],
        },
        "output_yaml_count": status_counts["ok"],
        "output_skipped_count": stage1_skipped + not_perf + status_counts["error"],
        "source_commits": {
            "vllm": source_commit_vllm,
            "vllm-ascend": source_commit_vllm_ascend,
        },
    }

    manifest_path = output_dir / "manifest.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info("Manifest written to %s", manifest_path)
    return manifest_path
