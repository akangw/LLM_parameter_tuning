#!/usr/bin/env python3
"""Warm checkpoint files into the page cache of one physical node."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
from pathlib import Path


def checkpoint_files(roots: list[Path]) -> list[Path]:
    """Return each Safetensors checkpoint once, in deterministic order."""
    files: set[Path] = set()
    for root in roots:
        if not root:
            continue
        if not root.is_dir():
            raise FileNotFoundError(f"checkpoint root is not a directory: {root}")
        files.update(path.resolve() for path in root.rglob("*.safetensors"))
    return sorted(files)


def prefetch_file(path: Path, block_size: int) -> int:
    """Sequentially read one file so its pages enter this node's page cache."""
    total = 0
    buffer = bytearray(block_size)
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as stream:
        while (read_bytes := stream.readinto(buffer)):
            # Touch the returned view before reusing the buffer. readinto() has
            # already copied every byte from the kernel, but this keeps the
            # intended full-block read explicit for alternate Python runtimes.
            _ = view[read_bytes - 1]
            total += read_bytes
    return total


def prefetch(
    roots: list[Path], threads: int, block_size: int
) -> tuple[int, int, float]:
    if threads < 1:
        raise ValueError("threads must be at least 1")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    files = checkpoint_files(roots)
    if not files:
        raise FileNotFoundError(
            "no .safetensors checkpoint files found under: "
            + ", ".join(str(root) for root in roots)
        )

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        total_bytes = sum(
            executor.map(lambda path: prefetch_file(path, block_size), files)
        )
    return len(files), total_bytes, time.perf_counter() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16 * 1024 * 1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files, total_bytes, elapsed = prefetch(args.roots, args.threads, args.block_size)
    safe_elapsed = max(elapsed, 1e-9)
    print(
        "NODE_CHECKPOINT_PREFETCH_COMPLETED "
        f"pid={os.getpid()} files={files} bytes={total_bytes} "
        f"seconds={elapsed:.2f} shards_per_second={files / safe_elapsed:.3f} "
        f"gib_per_second={total_bytes / 1024**3 / safe_elapsed:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
