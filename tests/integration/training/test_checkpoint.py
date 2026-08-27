from __future__ import annotations

import hashlib

import pytest
import torch

from zero_ttt.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManager,
    checkpoint_metadata,
)


def test_v4_checkpoint_and_publication_schemas_are_rejected(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"checkpoint_schema_version": 4}, path)
    for loader in (CheckpointManager.load, CheckpointManager.load_publication):
        with pytest.raises(
            ValueError,
            match=r"unsupported checkpoint schema v4; expected v5; migration is not supported",
        ):
            loader(path)


def test_publications_are_immutable_run_scoped_and_retained(tmp_path) -> None:
    manager = CheckpointManager(tmp_path, keep=1, publication_keep=2)
    config_json = "{}"
    metadata = checkpoint_metadata(
        config_json,
        hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
    )
    state = {"weight": torch.ones(1)}
    first = manager.save_publication("run-a", 1, 4, state, metadata)
    manager.save_publication("run-a", 2, 8, state, metadata)
    latest = manager.save_publication("run-a", 3, 12, state, metadata)
    assert not first.exists()
    assert latest.exists()
    assert manager.current_publication() == latest
    loaded = manager.load_publication(latest)
    assert loaded["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION == 5
    assert loaded["model_version"] == 3
    assert manager.save_publication("run-a", 3, 12, state, metadata) == latest
    with pytest.raises(FileExistsError, match="conflicting"):
        manager.save_publication("run-a", 3, 16, state, metadata)
