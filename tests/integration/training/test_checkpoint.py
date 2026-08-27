from __future__ import annotations

import hashlib
import json

import pytest
import torch

from zero_ttt.training.checkpoint import CheckpointManager, checkpoint_metadata
from zero_ttt.versioning import MODEL_ARTIFACT_SCHEMA


def test_v5_checkpoint_and_publication_schemas_are_rejected(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"checkpoint_schema_version": 5}, path)
    for loader in (CheckpointManager.load, CheckpointManager.load_publication):
        with pytest.raises(
            ValueError,
            match=r"model artifact.*expected v6.*new run",
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
    pointer = json.loads((manager.publication_dir / "current.json").read_text(encoding="utf-8"))
    publication_metadata = json.loads((latest.parent / "metadata.json").read_text(encoding="utf-8"))
    assert pointer["schema_version"] == MODEL_ARTIFACT_SCHEMA.current
    assert publication_metadata["schema_version"] == MODEL_ARTIFACT_SCHEMA.current
    loaded = manager.load_publication(latest)
    assert loaded["checkpoint_schema_version"] == MODEL_ARTIFACT_SCHEMA.current
    assert loaded["model_version"] == 3
    assert manager.save_publication("run-a", 3, 12, state, metadata) == latest
    with pytest.raises(FileExistsError, match="conflicting"):
        manager.save_publication("run-a", 3, 16, state, metadata)
    with pytest.raises(FileExistsError, match="conflicting"):
        manager.save_publication(
            "run-a",
            3,
            12,
            {"weight": torch.zeros(1)},
            metadata,
        )


def test_legacy_current_pt_is_not_a_publication_pointer(tmp_path) -> None:
    manager = CheckpointManager(tmp_path, keep=1)
    torch.save({"checkpoint_schema_version": 5}, manager.publication_dir / "current.pt")
    assert manager.current_publication() is None


def test_unversioned_publication_pointer_is_rejected(tmp_path) -> None:
    manager = CheckpointManager(tmp_path, keep=1)
    (manager.publication_dir / "current.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"model artifact.*expected v6"):
        manager.current_publication()
