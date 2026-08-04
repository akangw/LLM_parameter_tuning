"""Run the isolated current-source parameter portrait migration pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .cjx_bridge import extract_and_filter, legacy_projection
from .migration import add_migration_context, classify


BUILD_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BUILD_ROOT.parent
if str(BUILD_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILD_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isolated vLLM ParameterYAML migration using current-source evidence.")
    parser.add_argument("--vllm-src", type=Path,
                        default=PROJECT_ROOT / "sources" / "vllm",
                        help="Pinned vLLM source checkout for the target version.")
    parser.add_argument("--vllm-ascend-src", type=Path,
                        default=PROJECT_ROOT / "sources" / "vllm-ascend",
                        help="Pinned vLLM-Ascend source checkout for the target version.")
    parser.add_argument("--legacy-dir", type=Path,
                        default=BUILD_ROOT / "parse_params" / "output" / "params",
                        help="Existing ParameterYAML directory used as migration input.")
    parser.add_argument("--extract-output", type=Path,
                        default=BUILD_ROOT / "extracted_parameters",
                        help="Isolated current-version parameters.json and provenance directory.")
    parser.add_argument("--output", type=Path,
                        default=BUILD_ROOT / "migration_candidates",
                        help="Isolated ParameterYAML directory; never use parse_params/output here.")
    parser.add_argument("--target-context", type=Path,
                        default=BUILD_ROOT / "target-context.snapshot.yaml")
    parser.add_argument("--with-claude", action="store_true",
                        help="Also use the original optional document extractor during extraction.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract, filter, and audit only; do not make LLM calls.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-params", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _load_context(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Target context not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "target-context/v1":
        raise SystemExit(f"Unsupported target context: {path}")
    return data


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                text=True, capture_output=True, timeout=10)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_sources(args: argparse.Namespace, context: dict) -> None:
    for label, path, package in (
        ("vLLM", args.vllm_src, "vllm"),
        ("vLLM-Ascend", args.vllm_ascend_src, "vllm_ascend"),
    ):
        if not (path / package).is_dir():
            raise SystemExit(f"{label} source is invalid: expected {path / package}")
    expected = context.get("release", {})
    for label, path, key in (
        ("vLLM", args.vllm_src, "vllm_commit"),
        ("vLLM-Ascend", args.vllm_ascend_src, "vllm_ascend_commit"),
    ):
        actual, wanted = _git_commit(path), expected.get(key)
        if actual and wanted and actual != wanted:
            raise SystemExit(
                f"{label} commit mismatch: context requires {wanted}, source has {actual}")


def _configure_legacy_modules(args: argparse.Namespace, context: dict) -> None:
    """Point unchanged legacy parser modules at the pinned source checkouts."""
    os.environ["VLLM_SRC"] = str(args.vllm_src.resolve())
    os.environ["VLLM_ASCEND_SRC"] = str(args.vllm_ascend_src.resolve())

    from parse_params import config
    release = context.get("release", {})
    config.VLLM_ROOT = args.vllm_src.resolve()
    config.VLLM_ASCEND_ROOT = args.vllm_ascend_src.resolve()
    config.DOC_SEARCH_DIRS = [(config.VLLM_ASCEND_ROOT / "docs" / "source", "vllm-ascend")]
    config.SOURCE_COMMIT_VLLM = str(release.get("vllm_commit") or "")
    config.SOURCE_COMMIT_VLLM_ASCEND = str(release.get("vllm_ascend_commit") or "")
    config.LLM_CONCURRENCY = args.concurrency


class MigrationPromptBuilder:
    """Original prompt plus auditable, non-authoritative old-portrait hints."""

    def __init__(self, logger: logging.Logger):
        from parse_params.stage2_analyzer import PromptBuilder
        self._base = PromptBuilder(logger)
        self.system_prompt = self._base.system_prompt

    def build_user_prompt(self, param: dict, definition_context: str,
                          usage_contexts: str, doc_contexts: str = "") -> str:
        prompt = self._base.build_user_prompt(
            param, definition_context, usage_contexts, doc_contexts)
        migration = param.get("_migration", {})
        classification = migration.get("migration_class", "CURRENT_ONLY")
        if classification == "CURRENT_ONLY":
            return prompt + (
                "\n\n## Migration status\nThis is a current-version-only parameter. "
                "Do not infer values, constraints, or performance claims from similarly named old parameters.\n")
        profiles = migration.get("legacy_profiles", [])
        safe_fields = ("name", "type", "scope", "default", "valid_choices",
                       "performance_impact", "performance_scope", "impact_detail",
                       "related_parameters", "constraints", "tuning_advice")
        seeds = [{key: profile[key] for key in safe_fields if key in profile}
                 for profile in profiles]
        role = ("high-confidence hypothesis" if classification == "A"
                else "navigation hint only")
        return prompt + (
            "\n\n## Migration status\n"
            f"Classification: {classification}. Legacy material is a {role}. "
            "It is not evidence and must not be copied verbatim. Verify every claim "
            "against the current definition and usage contexts above. Output only the "
            "existing ParameterYAML schema, with current-source conclusions.\n"
            "Legacy portraits (untrusted hints):\n```json\n"
            + json.dumps(seeds, ensure_ascii=False, indent=2, default=str) + "\n```\n")


async def _run(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    context = _load_context(args.target_context)
    _validate_sources(args, context)
    _configure_legacy_modules(args, context)
    if not args.legacy_dir.is_dir():
        raise SystemExit(f"Legacy ParameterYAML directory not found: {args.legacy_dir}")

    output = args.output.resolve()
    extract_output = args.extract_output.resolve()
    if output == (BUILD_ROOT / "parse_params" / "output").resolve():
        raise SystemExit("Refusing to write into parse_params/output; use an isolated --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    extract_output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("migration_pipeline")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)

    extraction, worklist = extract_and_filter(
        args.vllm_src.resolve(), args.vllm_ascend_src.resolve())
    rich_params = list(extraction["parameters"])
    selected_rich = list(worklist["parameters"])
    params = legacy_projection(rich_params)
    selected = legacy_projection(selected_rich)
    (extract_output / "parameters.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (extract_output / "parameters.structured.json").write_text(
        json.dumps(extraction, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "portrait-migration-provenance/v1",
        "target_context": context,
        "sources": {
            "vllm": {"path": str(args.vllm_src.resolve()), "commit": _git_commit(args.vllm_src)},
            "vllm_ascend": {"path": str(args.vllm_ascend_src.resolve()), "commit": _git_commit(args.vllm_ascend_src)},
        },
        "parameter_count": len(params),
        "structured_parameter_count": len(rich_params),
        "stage1_policy": worklist.get("filter_policy_version"),
        "stage1_worklist_hash": worklist.get("worklist_hash"),
    }
    (extract_output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    skipped = list(worklist.get("skipped", []))
    reasons = dict(worklist.get("filter_summary", {}).get("decision_reasons", {}))
    manifest = classify(params, selected, args.legacy_dir.resolve())
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "stage1-skipped.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports / "stage1-summary.json").write_text(json.dumps({
        "input": len(params), "passed": len(selected), "skipped": len(skipped),
        "decision_reasons": reasons, "filter_policy_version": worklist.get("filter_policy_version"),
        "worklist_hash": worklist.get("worklist_hash"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports / "migration-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Stage 1 retained %d/%d parameters; migration plan: %s",
                len(selected), len(params), manifest["summary"]["candidate_plan_counts"])
    if args.dry_run:
        return

    to_analyze = add_migration_context(selected, manifest)
    if args.max_params > 0:
        to_analyze = to_analyze[:args.max_params]
    from parse_params.progress import ProgressManager
    from parse_params.stage2_analyzer import Stage2Analyzer
    progress = ProgressManager(output / "progress.json", logger)
    # New output is isolated; do not accidentally reuse a Stage 1 result from a
    # differently extracted source set.
    if not args.resume:
        progress.set_stage1_results(to_analyze, skipped)
    analyzer = Stage2Analyzer(output, reports, logger)
    analyzer.prompt_builder = MigrationPromptBuilder(logger)
    results = await analyzer.run(to_analyze, progress)
    from parse_params.manifest import generate_manifest
    generate_manifest(
        total_params=len(params), stage1_passed=len(selected), stage1_skipped=len(skipped),
        stage2_results=results, output_dir=output, logger=logger,
        source_commit_vllm=provenance["sources"]["vllm"]["commit"] or "",
        source_commit_vllm_ascend=provenance["sources"]["vllm_ascend"]["commit"] or "",
    )


def main() -> None:
    args = _parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
