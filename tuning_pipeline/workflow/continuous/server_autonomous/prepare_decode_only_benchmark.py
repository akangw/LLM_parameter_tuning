from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def contained(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def prepare(
    *,
    allowed_root: Path,
    source_spec_root: Path,
    target_spec_root: Path,
    suite_overlay: Path,
) -> Path:
    allowed_root = allowed_root.resolve()
    source_spec_root = source_spec_root.resolve()
    target_spec_root = target_spec_root.resolve()
    suite_overlay = suite_overlay.resolve()
    for name, path in {
        "source_spec_root": source_spec_root,
        "target_spec_root": target_spec_root,
        "suite_overlay": suite_overlay,
    }.items():
        if not contained(path, allowed_root):
            raise ValueError(f"{name} is outside allowed_root: {path}")
    if source_spec_root == target_spec_root:
        raise ValueError("source and target benchmark spec roots must differ")
    if not source_spec_root.is_dir():
        raise FileNotFoundError(source_spec_root)
    if not suite_overlay.is_file():
        raise FileNotFoundError(suite_overlay)

    # Deliberately additive: never remove or replace the validated Fast-C32 V2
    # source tree. dirs_exist_ok also makes an interrupted preparation resumable.
    shutil.copytree(source_spec_root, target_spec_root, dirs_exist_ok=True)
    target_suite = target_spec_root / "suites" / suite_overlay.name
    target_suite.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(suite_overlay, target_suite)
    return target_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--source-spec-root", type=Path, required=True)
    parser.add_argument("--target-spec-root", type=Path, required=True)
    parser.add_argument("--suite-overlay", type=Path, required=True)
    args = parser.parse_args()
    target = prepare(
        allowed_root=args.allowed_root,
        source_spec_root=args.source_spec_root,
        target_spec_root=args.target_spec_root,
        suite_overlay=args.suite_overlay,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
