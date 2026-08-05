from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .builder import MODULE_DIR, PROJECT_ROOT
from .compatibility import DEFAULT_POLICY_PATH
from .pipeline import AutomaticRegistryPipeline, DEFAULT_SOURCE_ROOT, write_full_outputs


def parse_args() -> argparse.Namespace:
    compiler_dir = MODULE_DIR.parent / "search_space_compiler"
    parser = argparse.ArgumentParser(
        description=(
            "Run the automatic path from tagged recall through a generated registry "
            "into the existing Search-Space Compiler. This standalone command writes "
            "only to its explicit output directory."
        )
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=PROJECT_ROOT / "tag_params" / "output" / "params",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=compiler_dir / "scenario.glm52-a3-aligned-l1.yaml",
    )
    parser.add_argument("--policy", type=Path, default=compiler_dir / "policy.yaml")
    parser.add_argument(
        "--compatibility-policy", type=Path, default=DEFAULT_POLICY_PATH
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Pinned directory containing vllm/ and vllm-ascend/ checkouts",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = AutomaticRegistryPipeline(
        knowledge_dir=args.knowledge_dir,
        scenario_path=args.scenario,
        policy_path=args.policy,
        compatibility_policy_path=args.compatibility_policy,
        source_root=args.source_root,
    )
    registry, search_result = pipeline.compile()
    summary = {
        **{
            key: value
            for key, value in registry["audit"].items()
            if not isinstance(value, (list, dict))
        },
        "compatibility_rejected_groups": len(
            registry["audit"].get("rejected_groups", [])
        ),
        "eligible_tunable_parameters": search_result["summary"][
            "eligible_tunable_parameters"
        ],
        "active_parameters": search_result["summary"]["active_parameters"],
        "reserve_parameters": search_result["summary"]["reserve_parameters"],
        "fixed_parameters": search_result["summary"]["fixed_parameters"],
        "rejected_parameters": search_result["summary"]["rejected_parameters"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("active:", ", ".join(search_result["active_search_limits"]))
    if args.dry_run:
        print("dry-run: no files written; Controller state unchanged")
        return 0
    output = args.output or (
        MODULE_DIR
        / "runs"
        / ("full_pipeline_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    for path in write_full_outputs(registry, search_result, output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
