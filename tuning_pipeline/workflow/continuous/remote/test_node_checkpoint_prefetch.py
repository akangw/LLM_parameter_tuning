from __future__ import annotations

from pathlib import Path

import pytest

from .node_checkpoint_prefetch import checkpoint_files, prefetch


def test_prefetch_reads_all_roots_once(tmp_path: Path) -> None:
    main = tmp_path / "main"
    draft = tmp_path / "draft"
    main.mkdir()
    draft.mkdir()
    (main / "model-00001-of-00002.safetensors").write_bytes(b"a" * 31)
    (main / "model-00002-of-00002.safetensors").write_bytes(b"b" * 17)
    (draft / "draft.safetensors").write_bytes(b"c" * 11)
    (draft / "ignored.json").write_text("{}", encoding="utf-8")

    files = checkpoint_files([main, draft, main])
    assert len(files) == 3

    file_count, total_bytes, elapsed = prefetch(
        [main, draft, main], threads=2, block_size=7
    )
    assert file_count == 3
    assert total_bytes == 59
    assert elapsed >= 0


def test_prefetch_fails_closed_without_checkpoints(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no .safetensors"):
        prefetch([tmp_path], threads=1, block_size=16)
