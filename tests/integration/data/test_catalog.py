from __future__ import annotations

import dataclasses
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from zero_ttt.data.catalog import Catalog
from zero_ttt.data.catalog_source import CatalogBatchSource
from zero_ttt.data.manifest import ManifestAsset
from zero_ttt.data.records import AnnotationRecord, TrajectoryRecord
from zero_ttt.data.shards import ShardStore
from zero_ttt.game.rules import BOARD_AREA
from zero_ttt.versioning import CATALOG_SCHEMA, RECORD_SCHEMA, SHARD_SCHEMA


def test_shard_catalog_snapshot_and_annotation_sampling(
    tmp_path: Path,
    trajectory_factory,
    manifest_factory,
) -> None:
    store = ShardStore(tmp_path / "processed")
    catalog_path = tmp_path / "catalog.sqlite"
    record = trajectory_factory()
    asset = ManifestAsset("source.zip", record.asset_sha256, 1)
    annotation = AnnotationRecord(
        schema_version=RECORD_SCHEMA.current,
        game_id=record.game_id,
        ply=0,
        teacher_fingerprint="teacher-v1",
        policy_actions=(2,),
        policy_values=(1.0,),
        value=0.25,
        value_available=True,
        score_margin=0.5,
        score_available=True,
        ownership=(0.0,) * BOARD_AREA,
        ownership_available=True,
    )
    trajectory_info = store.write_trajectories([record])
    assert store.read_trajectories(trajectory_info) == (record,)

    with Catalog(catalog_path, store) as catalog:
        catalog.register_asset(manifest_factory(asset), asset)
        catalog.commit_trajectory_shard(trajectory_info, [record])
        annotation_info = store.write_annotations([annotation])
        catalog.commit_annotation_shard(annotation_info, [annotation])
        snapshot_id = catalog.create_snapshot(seed=9, validation_fraction=0.0)
        listed = catalog.list_snapshots()
        assert listed[0].snapshot_id == snapshot_id
        assert listed[0].seed == 9
        assert listed[0].split == "train"
        assert listed[0].games == 1
        assert listed[0].positions == record.trainable_position_count
        assert catalog.create_snapshot(seed=9, validation_fraction=0.0) == snapshot_id
        late_annotation = dataclasses.replace(annotation, teacher_fingerprint="teacher-v2")
        late_info = store.write_annotations([late_annotation])
        catalog.commit_annotation_shard(late_info, [late_annotation])
        assert catalog.create_snapshot(seed=9, validation_fraction=0.0) != snapshot_id
        assert catalog.snapshot_annotations(snapshot_id, "teacher-v2") == ()
        catalog.mark_annotation_deleted(record.game_id, 0, "teacher-v1")
        assert len(catalog.snapshot_annotations(snapshot_id, "teacher-v1")) == 1
        assert catalog.recover() == ()

    with CatalogBatchSource(
        catalog_path,
        store.root,
        snapshot_id,
        annotation_mode="require_exact",
        teacher_fingerprint="teacher-v1",
    ) as source:
        first = source.next_batch(3, np.random.default_rng(4))
        second = source.next_batch(3, np.random.default_rng(4))
        assert np.array_equal(first.board, second.board)
        assert np.array_equal(first.policy, second.policy)
        assert np.all(first.value_mask)
        assert np.allclose(first.value, 0.25)
        assert np.all(first.ownership_mask)
        assert np.allclose(first.policy.sum(axis=1), 1.0)

    with Catalog(catalog_path, store) as catalog:
        catalog.mark_trajectory_deleted(record.game_id)
        assert catalog.garbage_collect() == ()
        assert catalog.snapshot_trajectories(snapshot_id)[0].game_id == record.game_id


def test_pickle_object_shards_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        schema_version=np.asarray(SHARD_SCHEMA.current, dtype=np.int32),
        record_schema_version=np.asarray(RECORD_SCHEMA.current, dtype=np.int32),
        kind=np.asarray(1, dtype=np.uint8),
        bad=np.asarray([object()], dtype=object),
    )
    with (
        pytest.raises(ValueError, match=r"Object arrays cannot be loaded|forbidden"),
        ShardStore._open_validated(path),
    ):
        pass


@pytest.mark.parametrize("dtype", (np.int32, np.int64))
def test_current_integer_scalar_shard_schemas_are_accepted(
    tmp_path: Path,
    dtype,
) -> None:
    path = tmp_path / f"schema-{dtype.__name__}.npz"
    np.savez(
        path,
        schema_version=np.asarray(SHARD_SCHEMA.current, dtype=dtype),
        record_schema_version=np.asarray(RECORD_SCHEMA.current, dtype=dtype),
        kind=np.asarray(1, dtype=np.uint8),
    )
    with ShardStore._open_validated(path):
        pass


@pytest.mark.parametrize(
    "invalid",
    (
        np.asarray(float(SHARD_SCHEMA.current), dtype=np.float64),
        np.asarray(str(SHARD_SCHEMA.current)),
        np.asarray(True),
        np.asarray([SHARD_SCHEMA.current], dtype=np.int32),
    ),
    ids=("float", "string", "boolean", "non-scalar"),
)
@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("schema_version", "NPZ shard"),
        ("record_schema_version", "trajectory/annotation record"),
    ),
)
def test_shard_schemas_require_exact_integer_scalars(
    tmp_path: Path,
    invalid: np.ndarray,
    field: str,
    message: str,
) -> None:
    arrays = {
        "schema_version": np.asarray(SHARD_SCHEMA.current, dtype=np.int32),
        "record_schema_version": np.asarray(RECORD_SCHEMA.current, dtype=np.int32),
        "kind": np.asarray(1, dtype=np.uint8),
    }
    arrays[field] = invalid
    path = tmp_path / "invalid-schema.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=message), ShardStore._open_validated(path):
        pass


def test_catalog_detects_corrupt_registered_shard(
    tmp_path: Path,
    trajectory_factory,
    manifest_factory,
) -> None:
    store = ShardStore(tmp_path / "processed")
    catalog_path = tmp_path / "catalog.sqlite"
    record = trajectory_factory()
    asset = ManifestAsset("source.zip", record.asset_sha256, 1)
    info = store.write_trajectories([record])
    with Catalog(catalog_path, store) as catalog:
        catalog.register_asset(manifest_factory(asset), asset)
        catalog.commit_trajectory_shard(info, [record])
    path = store.resolve(info.relative_path)
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with Catalog(catalog_path, store) as catalog, pytest.raises(ValueError, match="SHA-256"):
        catalog.verify()


def test_recover_preserves_live_temporary_files_and_removes_stale_ones(
    tmp_path: Path,
) -> None:
    store = ShardStore(tmp_path / "processed")
    live = store.trajectory_dir / ".shard-live.tmp"
    stale = store.trajectory_dir / ".shard-stale.tmp"
    live.write_bytes(b"live")
    stale.write_bytes(b"stale")
    old = time.time() - 25 * 60 * 60
    os.utime(stale, (old, old))

    with Catalog(tmp_path / "catalog.sqlite", store) as catalog:
        assert catalog.recover() == ()
    assert live.is_file()
    assert not stale.exists()


def test_gc_tombstone_is_recovered_after_file_delete_interruption(
    tmp_path: Path,
    trajectory_factory,
    manifest_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ShardStore(tmp_path / "processed")
    catalog_path = tmp_path / "catalog.sqlite"
    record = trajectory_factory()
    asset = ManifestAsset("source.zip", record.asset_sha256, 1)
    info = store.write_trajectories([record])
    shard_path = store.resolve(info.relative_path)
    original_unlink = Path.unlink

    def interrupted_unlink(path: Path, *args, **kwargs):
        if path == shard_path:
            raise OSError("injected unlink interruption")
        return original_unlink(path, *args, **kwargs)

    with Catalog(catalog_path, store) as catalog:
        catalog.register_asset(manifest_factory(asset), asset)
        catalog.commit_trajectory_shard(info, [record])
        catalog.mark_trajectory_deleted(record.game_id)
        monkeypatch.setattr(Path, "unlink", interrupted_unlink)
        with pytest.raises(OSError, match="interruption"):
            catalog.garbage_collect()
        deleted = catalog.connection.execute(
            "SELECT deleted FROM shards WHERE shard_sha256=?", (info.sha256,)
        ).fetchone()["deleted"]
        assert deleted == 1
        assert shard_path.is_file()

    monkeypatch.setattr(Path, "unlink", original_unlink)
    with Catalog(catalog_path, store) as catalog:
        assert catalog.recover() == ()
    assert not shard_path.exists()


def snapshot_for_records(
    root: Path,
    records: list[TrajectoryRecord],
    pack_together: bool,
    manifest_factory,
    annotation: AnnotationRecord | None = None,
) -> str:
    store = ShardStore(root / "processed")
    asset = ManifestAsset("source.zip", records[0].asset_sha256, 1)
    with Catalog(root / "catalog.sqlite", store) as catalog:
        catalog.register_asset(manifest_factory(asset), asset)
        groups = [records] if pack_together else [[record] for record in records]
        for group in groups:
            info = store.write_trajectories(group)
            catalog.commit_trajectory_shard(info, group)
        if annotation is not None:
            info = store.write_annotations([annotation])
            catalog.commit_annotation_shard(info, [annotation])
        return catalog.create_snapshot(seed=3, validation_fraction=0.0)


def test_snapshot_identity_uses_logical_content_not_shard_packing(
    tmp_path: Path,
    trajectory_factory,
    manifest_factory,
) -> None:
    first = trajectory_factory()
    second = dataclasses.replace(
        first,
        game_id="c" * 64,
        content_sha256="",
        ordinal=1,
    )
    packed = snapshot_for_records(tmp_path / "packed", [first, second], True, manifest_factory)
    split = snapshot_for_records(tmp_path / "split", [first, second], False, manifest_factory)
    assert packed == split

    changed = dataclasses.replace(first, content_sha256="", value_black=-1.0)
    changed_snapshot = snapshot_for_records(
        tmp_path / "changed", [changed, second], True, manifest_factory
    )
    assert changed_snapshot != packed

    annotation = AnnotationRecord(
        schema_version=RECORD_SCHEMA.current,
        game_id=first.game_id,
        ply=0,
        teacher_fingerprint="teacher-v1",
        policy_actions=(0,),
        policy_values=(1.0,),
        value=0.25,
        value_available=True,
        score_margin=0.0,
        score_available=False,
        ownership=(0.0,) * BOARD_AREA,
        ownership_available=False,
    )
    annotated = snapshot_for_records(
        tmp_path / "annotated",
        [first, second],
        True,
        manifest_factory,
        annotation,
    )
    changed_annotation = dataclasses.replace(annotation, value=-0.25)
    changed_annotated = snapshot_for_records(
        tmp_path / "changed-annotation",
        [first, second],
        True,
        manifest_factory,
        changed_annotation,
    )
    assert annotated != changed_annotated


def test_snapshot_membership_is_built_inside_an_immediate_transaction(
    tmp_path: Path,
    trajectory_factory,
    manifest_factory,
) -> None:
    record = trajectory_factory()
    store = ShardStore(tmp_path / "processed")
    asset = ManifestAsset("source.zip", record.asset_sha256, 1)
    with Catalog(tmp_path / "catalog.sqlite", store) as catalog:
        catalog.register_asset(manifest_factory(asset), asset)
        info = store.write_trajectories([record])
        catalog.commit_trajectory_shard(info, [record])
        statements = []
        catalog.connection.set_trace_callback(statements.append)
        catalog.create_snapshot(seed=6, validation_fraction=0.0)
    begin = next(index for index, sql in enumerate(statements) if "BEGIN IMMEDIATE" in sql)
    selection = next(
        index for index, sql in enumerate(statements) if "FROM trajectories WHERE deleted=0" in sql
    )
    membership = next(
        index for index, sql in enumerate(statements) if "INSERT INTO snapshot_trajectories" in sql
    )
    commit = next(index for index, sql in enumerate(statements) if sql.strip() == "COMMIT")
    assert begin < selection < membership < commit


def test_previous_catalog_is_rejected_without_mutation(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.sqlite"
    connection = sqlite3.connect(catalog_path)
    connection.execute("CREATE TABLE catalog_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.execute("INSERT INTO catalog_meta VALUES('schema_version','3')")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match=r"SQLite catalog.*expected v4.*rebuild"):
        Catalog(catalog_path, ShardStore(tmp_path / "processed"))
    connection = sqlite3.connect(catalog_path)
    version = connection.execute(
        "SELECT value FROM catalog_meta WHERE key='schema_version'"
    ).fetchone()[0]
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    connection.close()
    assert version == "3"
    assert tables == {"catalog_meta"}


def test_previous_records_and_shards_are_rejected(
    tmp_path: Path,
    trajectory_factory,
) -> None:
    with pytest.raises(ValueError, match=r"record.*expected v4"):
        dataclasses.replace(trajectory_factory(), schema_version=3, content_sha256="")
    path = tmp_path / "v3.npz"
    np.savez(
        path,
        schema_version=np.asarray(3, dtype=np.int32),
        record_schema_version=np.asarray(3, dtype=np.int32),
        kind=np.asarray(1, dtype=np.uint8),
    )
    with (
        pytest.raises(ValueError, match=r"NPZ shard.*expected v4.*rebuild"),
        ShardStore._open_validated(path),
    ):
        pass


def test_new_catalog_records_only_the_current_schema(tmp_path: Path) -> None:
    path = tmp_path / "catalog.sqlite"
    with Catalog(path, ShardStore(tmp_path / "processed")) as catalog:
        version = catalog.connection.execute(
            "SELECT value FROM catalog_meta WHERE key='schema_version'"
        ).fetchone()[0]
        assert version == str(CATALOG_SCHEMA.current)


def test_annotation_content_hash_detects_logical_tampering(
    tmp_path: Path,
    trajectory_factory,
) -> None:
    record = trajectory_factory()
    annotation = AnnotationRecord(
        schema_version=RECORD_SCHEMA.current,
        game_id=record.game_id,
        ply=0,
        teacher_fingerprint="teacher-v1",
        policy_actions=(0,),
        policy_values=(1.0,),
        value=0.25,
        value_available=True,
        score_margin=0.0,
        score_available=False,
        ownership=(0.0,) * BOARD_AREA,
        ownership_available=False,
    )
    store = ShardStore(tmp_path / "processed")
    info = store.write_annotations([annotation])
    path = store.resolve(info.relative_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["value"][0] = np.float32(-0.25)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="content_sha256"):
        store.read_annotations(path)
