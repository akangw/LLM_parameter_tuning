from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .builder import MODULE_DIR, PROJECT_ROOT, RegistryBuilder, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an offline, non-executable registry proposal directly from "
            "tagged parameter portraits. This command never edits the existing "
            "registry or continuous controller."
        )
    )
    compiler_dir = MODULE_DIR.parent / "search_space_compiler"
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
    parser.add_argument(
        "--policy",
        type=Path,
        default=compiler_dir / "policy.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the summary without writing any files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = RegistryBuilder(
        knowledge_dir=args.knowledge_dir,
        scenario_path=args.scenario,
        policy_path=args.policy,
    ).build()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if args.dry_run:
        print("dry-run: no files written; existing registry and controller unchanged")
        return 0
    output = args.output or (
        MODULE_DIR
        / "runs"
        / ("registry_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    for path in write_outputs(result, output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
