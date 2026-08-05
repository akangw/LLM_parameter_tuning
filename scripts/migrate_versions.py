"""Self-contained version-to-portrait migration entry point.

The command never replaces accepted production portraits. Each version pair is
built under an isolated, commit-qualified run directory and can be audited
before activation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
PORTRAIT = ROOT / "portrait_pipeline"
BASE_CONTEXT = PORTRAIT / "build" / "target-context.snapshot.yaml"
DEFAULT_SCENARIO = (
    ROOT
    / "tuning_pipeline"
    / "workflow"
    / "search_space_compiler"
    / "scenario.glm52-a3-aligned-l1.yaml"
)
DEFAULT_LEGACY = PORTRAIT / "outputs" / "ParameterYAML"
SEARCH_SPACE_PROFILES = (
    ROOT / "tuning_pipeline" / "workflow" / "search_space_profiles.yaml"
)


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise SystemExit(
            f"Command failed ({completed.returncode}): {subprocess.list2cmdline(command)}"
        )


def capture(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(completed.stderr.strip() or "Git command failed")
    return completed.stdout.strip()


def checkout(target: Path, url: str, ref: str) -> tuple[Path, str]:
    """Resolve a ref in a migration-only checkout, never the active source tree."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_dir():
        run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(target)])
    run(["git", "-C", str(target), "fetch", "--filter=blob:none", "origin", ref])
    commit = capture(["git", "-C", str(target), "rev-parse", "FETCH_HEAD"])
    run(["git", "-C", str(target), "checkout", "--detach", commit])
    return target, commit


def immutable_checkout(source: Path, target: Path, commit: str) -> Path:
    """Materialize a commit-qualified source tree owned by exactly one run."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_dir():
        run(["git", "clone", "--no-checkout", str(source), str(target)])
        run(["git", "-C", str(target), "checkout", "--detach", commit])
    actual = capture(["git", "-C", str(target), "rev-parse", "HEAD"])
    if actual != commit:
        raise SystemExit(
            f"Immutable source checkout mismatch: {target} has {actual}, expected {commit}"
        )
    return target


def portrait_set_hash(directory: Path) -> str:
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise SystemExit(
            f"Migration mode requires existing ParameterYAML files: {directory}"
        )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"Required input file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Fetch two source versions and build an isolated parameter-portrait migration"
    )
    result.add_argument("--vllm", required=True, help="vLLM tag, branch, or commit")
    result.add_argument(
        "--vllm-ascend", required=True, help="vLLM-Ascend tag, branch, or commit"
    )
    result.add_argument("--provider", choices=["codex", "anthropic"], default="codex")
    result.add_argument(
        "--portrait-mode", choices=["migrate", "rebuild"], default="migrate"
    )
    result.add_argument(
        "--legacy-dir",
        type=Path,
        default=DEFAULT_LEGACY,
        help="existing ParameterYAML input required by portrait-mode=migrate",
    )
    result.add_argument(
        "--tag-provider",
        choices=["codex", "anthropic", "openai_compatible", "command"],
        help="tagging provider; defaults to the portrait provider",
    )
    result.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO,
        help="scenario template controlling Tags recall and Search Limits",
    )
    result.add_argument(
        "--search-space-profile",
        default=None,
        help=(
            "profile from workflow/search_space_profiles.yaml; defaults to "
            "that file's default_profile"
        ),
    )
    result.add_argument(
        "--prepare-only",
        action="store_true",
        help="prepare/audit the queue without LLM execution",
    )
    result.add_argument("--resume", action="store_true")
    result.add_argument("--concurrency", type=int, default=8)
    result.add_argument(
        "--run-root", type=Path, default=PORTRAIT / "build" / "version_migrations"
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if not 1 <= args.concurrency <= 8:
        raise SystemExit("--concurrency must be between 1 and 8")
    legacy_hash = (
        portrait_set_hash(args.legacy_dir.resolve())
        if args.portrait_mode == "migrate"
        else None
    )
    profiles_document = yaml.safe_load(
        SEARCH_SPACE_PROFILES.read_text(encoding="utf-8")
    )
    if not isinstance(profiles_document, dict):
        raise SystemExit("Search-Space profiles document must be an object")
    profiles = profiles_document.get("profiles", {})
    args.search_space_profile = str(
        args.search_space_profile or profiles_document.get("default_profile", "")
    )
    if not args.search_space_profile:
        raise SystemExit("Search-Space profiles document has no default_profile")
    if args.search_space_profile not in profiles:
        raise SystemExit(
            f"Unknown Search-Space profile {args.search_space_profile!r}; "
            f"available={sorted(profiles)}"
        )
    search_space_profile = copy.deepcopy(profiles[args.search_space_profile])
    if search_space_profile.get("status") != "integrated":
        raise SystemExit(
            f"Search-Space profile is not integrated: {args.search_space_profile}"
        )
    profile_mode = str(search_space_profile.get("mode"))
    if profile_mode not in {"automatic_registry", "curated_registry"}:
        raise SystemExit(f"Unsupported Search-Space profile mode: {profile_mode}")
    workflow_root = ROOT / "tuning_pipeline"
    profile_input_keys = ["policy"]
    profile_input_keys.append(
        "compatibility_policy" if profile_mode == "automatic_registry" else "registry"
    )
    profile_inputs = {
        key: {
            "path": str((workflow_root / str(search_space_profile[key])).resolve()),
            "sha256": file_hash(
                (workflow_root / str(search_space_profile[key])).resolve()
            ),
        }
        for key in profile_input_keys
    }
    profile_identity = hashlib.sha256(
        json.dumps(
            {
                "name": args.search_space_profile,
                "definition": search_space_profile,
                "inputs": profile_inputs,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    scenario_template = (
        args.scenario.resolve()
        if args.scenario != DEFAULT_SCENARIO
        else (
            ROOT / "tuning_pipeline" / str(search_space_profile["scenario"])
        ).resolve()
    )
    scenario_hash = file_hash(scenario_template)
    tag_provider = args.tag_provider or args.provider
    source_root = args.run_root.resolve() / "_source_checkouts"
    vllm_resolver, vllm_commit = checkout(
        source_root / "vllm", "https://github.com/vllm-project/vllm.git", args.vllm
    )
    ascend_resolver, ascend_commit = checkout(
        source_root / "vllm-ascend",
        "https://github.com/vllm-project/vllm-ascend.git",
        args.vllm_ascend,
    )
    provider_code = {
        "codex": "cd",
        "anthropic": "an",
        "openai_compatible": "oa",
        "command": "cmd",
    }
    run_id = (
        f"{'mig' if args.portrait_mode == 'migrate' else 'reb'}-"
        f"p{provider_code[args.provider]}-t{provider_code[tag_provider]}_"
        f"v{vllm_commit[:10]}_a{ascend_commit[:10]}_s{scenario_hash[:8]}"
        f"_x{profile_identity[:8]}" + (f"_l{legacy_hash[:8]}" if legacy_hash else "")
    )
    run_root = args.run_root.resolve() / run_id
    vllm_path = immutable_checkout(
        vllm_resolver, run_root / "00_sources" / "vllm", vllm_commit
    )
    ascend_path = immutable_checkout(
        ascend_resolver,
        run_root / "00_sources" / "vllm-ascend",
        ascend_commit,
    )
    extract_dir = run_root / "01_extract"
    candidates_dir = run_root / "02_portrait_plan"
    queue_dir = run_root / "03_portrait_queue"
    tags_dir = run_root / "04_tags"
    search_limits_dir = run_root / "05_search_limits"
    run_root.mkdir(parents=True, exist_ok=True)

    context = copy.deepcopy(yaml.safe_load(BASE_CONTEXT.read_text(encoding="utf-8")))
    context["context_id"] = run_id
    context["release"]["vllm_commit"] = vllm_commit
    context["release"]["vllm_ascend_commit"] = ascend_commit
    context_path = run_root / "target-context.yaml"
    context_path.write_text(
        yaml.safe_dump(context, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    scenario = copy.deepcopy(
        yaml.safe_load(scenario_template.read_text(encoding="utf-8"))
    )
    scenario["scenario_id"] = (
        str(scenario.get("scenario_id", "scenario")) + "-" + run_id
    )
    image = scenario.setdefault("image", {})
    template_matches = (
        image.get("vllm_commit") == vllm_commit
        and image.get("vllm_ascend_commit") == ascend_commit
    )
    image["vllm_commit"] = vllm_commit
    image["vllm_ascend_commit"] = ascend_commit
    # Portraits and tags produced by this run are generated from these exact,
    # immutable source checkouts. Runtime capability verification remains a
    # separate gate when the source pair differs from the scenario template.
    image["parameter_portrait_exact_match"] = True
    if not template_matches:
        image["runtime_verification"] = "pending_for_new_source_pair"
        scenario.setdefault("capabilities", {})[
            "source"
        ] = "scenario_template_unverified_for_new_source_pair"
        scenario["capabilities"]["verified_canonical_parameters"] = []
    scenario_path = run_root / "scenario.snapshot.yaml"
    scenario_path.write_text(
        yaml.safe_dump(scenario, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "version-migration-run/v1",
        "requested_refs": {"vllm": args.vllm, "vllm_ascend": args.vllm_ascend},
        "resolved_commits": {"vllm": vllm_commit, "vllm_ascend": ascend_commit},
        "provider": args.provider,
        "portrait_mode": args.portrait_mode,
        "legacy_portraits": (
            {"path": str(args.legacy_dir.resolve()), "sha256": legacy_hash}
            if legacy_hash
            else None
        ),
        "tag_provider": tag_provider,
        "search_space_profile": {
            "name": args.search_space_profile,
            "definition": search_space_profile,
            "identity_sha256": profile_identity,
            "inputs": profile_inputs,
        },
        "scenario_template": {
            "path": str(scenario_template),
            "sha256": scenario_hash,
        },
        "scenario_snapshot": str(scenario_path),
        "activation_status": "isolated_not_activated",
        "pipeline_status": "preparing_portraits",
    }
    (run_root / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    migration = [
        sys.executable,
        "-m",
        "build.migration_pipeline",
        "--vllm-src",
        str(vllm_path),
        "--vllm-ascend-src",
        str(ascend_path),
        "--legacy-dir",
        str(args.legacy_dir.resolve()),
        "--portrait-mode",
        args.portrait_mode,
        "--extract-output",
        str(extract_dir),
        "--output",
        str(candidates_dir),
        "--target-context",
        str(context_path),
        "--concurrency",
        str(args.concurrency),
    ]
    if args.resume:
        migration.append("--resume")
    if args.provider == "codex" or args.prepare_only:
        migration.append("--dry-run")
    run(migration, cwd=PORTRAIT)

    if args.provider == "codex":
        if (queue_dir / "index.json").is_file():
            if not args.resume:
                raise SystemExit(
                    f"Portrait queue already exists: {queue_dir}; use --resume"
                )
        else:
            run(
                [
                    sys.executable,
                    "-m",
                    "build.codex_portrait_pipeline",
                    "--run-dir",
                    str(queue_dir),
                    "prepare",
                    "--extraction",
                    str(extract_dir / "parameters.structured.json"),
                    "--manifest",
                    str(candidates_dir / "reports" / "migration-manifest.json"),
                ],
                cwd=PORTRAIT,
            )
        if not args.prepare_only:
            run(
                [
                    sys.executable,
                    "-m",
                    "build.codex_portrait_pipeline.supervisor",
                    "--run-dir",
                    str(queue_dir),
                ],
                cwd=PORTRAIT,
            )
            run(
                [
                    sys.executable,
                    "-m",
                    "build.codex_portrait_pipeline",
                    "--run-dir",
                    str(queue_dir),
                    "audit",
                ],
                cwd=PORTRAIT,
            )
    if not args.prepare_only:
        portrait_dir = (
            queue_dir / "params"
            if args.provider == "codex"
            else candidates_dir / "params"
        )
        if not portrait_dir.is_dir():
            raise SystemExit(f"Completed portrait directory is missing: {portrait_dir}")
        run(
            [
                sys.executable,
                "-m",
                "tag_params",
                "--input",
                str(portrait_dir),
                "--output",
                str(tags_dir),
                "--target-context",
                str(context_path),
                "--provider",
                tag_provider,
                "--retry-errors",
                "--max-attempts",
                "3",
                "--workers",
                str(args.concurrency),
            ],
            cwd=ROOT / "tuning_pipeline",
        )
        run(
            [
                sys.executable,
                "-m",
                "tag_params.audit",
                "--input",
                str(portrait_dir),
                "--output",
                str(tags_dir),
            ],
            cwd=ROOT / "tuning_pipeline",
        )
        completion_marker = (
            search_limits_dir / "pipeline_manifest.json"
            if profile_mode == "automatic_registry"
            else search_limits_dir / "manifest.json"
        )
        if not completion_marker.is_file():
            if search_limits_dir.exists():
                raise SystemExit(
                    f"Incomplete Search Limits output exists: {search_limits_dir}; "
                    "use a new version/scenario run directory"
                )
            policy_path = (
                workflow_root / str(search_space_profile["policy"])
            ).resolve()
            if profile_mode == "automatic_registry":
                compatibility_path = (
                    workflow_root / str(search_space_profile["compatibility_policy"])
                ).resolve()
                run(
                    [
                        sys.executable,
                        "-m",
                        "workflow.registry_builder.full_pipeline",
                        "--knowledge-dir",
                        str(tags_dir / "params"),
                        "--scenario",
                        str(scenario_path),
                        "--policy",
                        str(policy_path),
                        "--compatibility-policy",
                        str(compatibility_path),
                        "--source-root",
                        str(run_root / "00_sources"),
                        "--output",
                        str(search_limits_dir),
                    ],
                    cwd=workflow_root,
                )
            elif profile_mode == "curated_registry":
                registry_path = (
                    workflow_root / str(search_space_profile["registry"])
                ).resolve()
                run(
                    [
                        sys.executable,
                        "-m",
                        "workflow.search_space_compiler",
                        "--knowledge-dir",
                        str(tags_dir / "params"),
                        "--scenario",
                        str(scenario_path),
                        "--registry",
                        str(registry_path),
                        "--policy",
                        str(policy_path),
                        "--output",
                        str(search_limits_dir),
                    ],
                    cwd=workflow_root,
                )
        manifest["pipeline_status"] = "search_limits_complete"
        manifest["artifacts"] = {
            "portraits": str(portrait_dir),
            "tags": str(tags_dir / "params"),
            "tag_audit": str(tags_dir / "audit.json"),
            "search_limits": str(
                search_limits_dir / "search_limits"
                if profile_mode == "automatic_registry"
                else search_limits_dir
            ),
            "registry": (
                str(search_limits_dir / "registry.generated.yaml")
                if profile_mode == "automatic_registry"
                else str(
                    ROOT / "tuning_pipeline" / str(search_space_profile["registry"])
                )
            ),
        }
        (run_root / "run-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
