from __future__ import annotations

from pathlib import Path

import pytest

from zero_ttt.control.runs import RunRepository, RuntimeLayout
from zero_ttt.data.catalog_types import SnapshotSummary


def _repository(tmp_path: Path) -> RunRepository:
    profile_root = tmp_path / "profiles"
    profile_root.mkdir()
    (profile_root / "tiny.toml").write_text(
        Path("configs/test.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return RunRepository(
        RuntimeLayout(
            source_root=tmp_path / "data",
            staging_root=tmp_path / "data" / "staging",
            manifest_root=tmp_path / "data" / "manifests",
            catalog_path=tmp_path / "data" / "catalog" / "catalog.sqlite",
            store_root=tmp_path / "data" / "processed",
            run_root=tmp_path / "runs",
            profile_root=profile_root,
        )
    )


def test_run_creation_freezes_profile_and_ignores_legacy_directories(
    tmp_path: Path, monkeypatch
) -> None:
    repository = _repository(tmp_path)
    monkeypatch.setattr(repository, "_validate_snapshot", lambda _snapshot: None)
    (repository.layout.run_root / "legacy" / "console").mkdir(parents=True)

    created = repository.create(" First run ", "tiny", "a" * 64)
    loaded = repository.load(created.run_id)

    assert loaded == created
    assert created.name == "First run"
    assert repository.list_runs() == (created,)
    assert (repository.layout.run_root / created.run_id / "experiment.toml").is_file()
    assert (repository.layout.run_root / created.run_id / "run.json").is_file()


def test_run_name_is_unique_and_frozen_config_is_verified(tmp_path: Path, monkeypatch) -> None:
    repository = _repository(tmp_path)
    monkeypatch.setattr(repository, "_validate_snapshot", lambda _snapshot: None)
    created = repository.create("Alpha", "tiny", "a" * 64)

    with pytest.raises(ValueError, match="already exists"):
        repository.create("alpha", "tiny", "a" * 64)

    frozen = repository.layout.run_root / created.run_id / "experiment.toml"
    frozen.write_text(frozen.read_text(encoding="utf-8").replace("seed = 7", "seed = 8"))
    with pytest.raises(ValueError, match="does not match"):
        repository.load(created.run_id)


@pytest.mark.parametrize(
    ("split", "source_kind", "games", "positions", "message"),
    [
        ("validation", "external", 1, 10, "train snapshot from external data"),
        ("train", "selfplay", 1, 10, "train snapshot from external data"),
        ("train", "external", 0, 0, "contains no trainable data"),
    ],
)
def test_run_creation_rejects_ineligible_cold_start_snapshots(
    tmp_path: Path,
    monkeypatch,
    split: str,
    source_kind: str,
    games: int,
    positions: int,
    message: str,
) -> None:
    repository = _repository(tmp_path)
    repository.layout.catalog_path.parent.mkdir(parents=True)
    repository.layout.catalog_path.touch()
    snapshot_id = "a" * 64
    summary = SnapshotSummary(
        snapshot_id=snapshot_id,
        seed=7,
        split=split,
        validation_fraction=0.1,
        source_kind=source_kind,
        task_id="task" if source_kind == "selfplay" else None,
        created_ns=1,
        games=games,
        positions=positions,
    )

    class FakeCatalog:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def list_snapshots(self) -> tuple[SnapshotSummary, ...]:
            return (summary,)

    monkeypatch.setattr("zero_ttt.control.runs.Catalog", FakeCatalog)

    with pytest.raises(ValueError, match=message):
        repository.create("invalid snapshot", "tiny", snapshot_id)
