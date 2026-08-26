from __future__ import annotations

import pytest
import torch

from zero_ttt.training.checkpoint import CheckpointManager


def test_legacy_checkpoint_schema_is_rejected(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"checkpoint_schema_version": 3}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        CheckpointManager.load(path)


def test_publications_are_immutable_run_scoped_and_retained(tmp_path) -> None:
    manager = CheckpointManager(tmp_path, keep=1, publication_keep=2)
    metadata = {
        "checkpoint_schema_version": 4,
        "config_json": "{}",
        "config_sha256": "0" * 64,
    }
    state = {"weight": torch.ones(1)}
    first = manager.save_publication("run-a", 1, 4, state, metadata)
    manager.save_publication("run-a", 2, 8, state, metadata)
    latest = manager.save_publication("run-a", 3, 12, state, metadata)
    assert not first.exists()
    assert latest.exists()
    assert manager.current_publication() == latest
    assert manager.save_publication("run-a", 3, 12, state, metadata) == latest
    with pytest.raises(FileExistsError, match="conflicting"):
        manager.save_publication("run-a", 3, 16, state, metadata)
