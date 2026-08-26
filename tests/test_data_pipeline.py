from __future__ import annotations

import hashlib
import dataclasses
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pytest

from zero_ttt.data.catalog import Catalog, TrajectoryLocator
from zero_ttt.data.catalog_source import CatalogBatchSource, SnapshotPositionIndex
from zero_ttt.data.importers import KataGoSgfImporter
from zero_ttt.data.manifest import ManifestAsset, SourceManifest
from zero_ttt.data.pipeline import import_manifest
from zero_ttt.data.records import AnnotationRecord, RECORD_SCHEMA_VERSION, TrajectoryRecord
from zero_ttt.data.shards import ShardStore
from zero_ttt.game.rules import BOARD_AREA


VALID_SGF = (
    b"(;FF[4]GM[1]SZ[19]HA[0]KM[0]"
    b"RU[koPOSITIONALscoreAREAtaxNONEsui1]RE[0]"
    b"C[startTurnIdx=1,mode=normal];B[aa];W[bb];B[];W[])"
)


def _trajectory(asset_sha256: str = "a" * 64) -> TrajectoryRecord:
    moves = (0, 1, 19, 20)
    return TrajectoryRecord(
        schema_version=RECORD_SCHEMA_VERSION,
        game_id="b" * 64,
        content_sha256="",
        dataset_id="test-data",
        asset_sha256=asset_sha256,
        member_path="games/test.sgfs",
        ordinal=0,
        rules="koPOSITIONALscoreAREAtaxNONEsui1",
        komi_half_points=0,
        max_moves=722,
        moves=moves,
        trainable_start_ply=0,
        policy_row_offsets=(0, 1, 2, 3, 4),
        policy_actions=moves,
        policy_values=(1.0, 1.0, 1.0, 1.0),
        value_black=1.0,
        value_available=True,
        score_margin_black=2.0,
        score_available=True,
        ownership_black=(0.0,) * BOARD_AREA,
        ownership_available=False,
    )


def _manifest(asset: ManifestAsset) -> SourceManifest:
    return SourceManifest(
        schema_version=1,
        dataset_id="test-data",
        source_type="katago-g170-sgfs-zip",
        license_id="CC0-1.0",
        license_url="https://example.invalid/license",
        assets=(asset,),
    )


def test_katago_importer_streams_records_and_structured_rejections(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    valid = VALID_SGF
    unsupported_size = valid.replace(b"SZ[19]", b"SZ[13]")
    cross_rules = valid.replace(
        b"koPOSITIONALscoreAREAtaxNONEsui1",
        b"koSIMPLEscoreAREAtaxNONEsui0",
    )
    cleanup = valid[:-1] + b";B[cc])"
    illegal = valid.replace(b";W[bb]", b";W[aa]")
    setup = valid.replace(b"HA[0]", b"HA[0]AB[cc]")
    fork = valid.replace(b"mode=normal", b"mode=fork")
    variation = valid.replace(b";W[bb]", b"(;W[bb])(;W[cc])")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "net/sgfs/games.sgfs",
            b"\n".join(
                (valid, cross_rules, unsupported_size, cleanup, illegal, setup, fork, variation)
            ),
        )
    payload = archive_path.read_bytes()
    asset = ManifestAsset("source.zip", hashlib.sha256(payload).hexdigest(), len(payload))
    manifest = _manifest(asset)

    events = list(KataGoSgfImporter().import_asset(manifest, asset, tmp_path))
    records = [event.record for event in events if event.kind == "trajectory"]
    reasons = [event.reason_code for event in events if event.kind == "reject"]
    assert len(records) == 2
    assert records[0] is not None and records[0].trainable_start_ply == 1
    assert records[0].trainable_position_count == 3
    assert records[0].value_available
    assert records[1] is not None and not records[1].value_available
    assert set(reasons) == {
        "board_size",
        "cleanup_phase",
        "illegal_move",
        "setup_stones",
        "mode",
        "variation",
    }

    altered = ManifestAsset("source.zip", "0" * 64, len(payload))
    with pytest.raises(ValueError, match="integrity"):
        list(KataGoSgfImporter().import_asset(_manifest(altered), altered, tmp_path))


def test_shard_catalog_snapshot_and_annotation_sampling(tmp_path: Path) -> None:
    store = ShardStore(tmp_path / "processed")
    catalog_path = tmp_path / "catalog.sqlite"
    record = _trajectory()
    asset = ManifestAsset("source.zip", record.asset_sha256, 1)
    annotation = AnnotationRecord(
        schema_version=RECORD_SCHEMA_VERSION,
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
        catalog.register_asset(_manifest(asset), asset)
        catalog.commit_trajectory_shard(trajectory_info, [record])
        annotation_info = store.write_annotations([annotation])
        catalog.commit_annotation_shard(annotation_info, [annotation])
        snapshot_id = catalog.create_snapshot(seed=9, validation_fraction=0.0)
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
        schema_version=np.asarray(2, dtype=np.int32),
        record_schema_version=np.asarray(2, dtype=np.int32),
        kind=np.asarray(1, dtype=np.uint8),
        bad=np.asarray([object()], dtype=object),
    )
    with pytest.raises(ValueError, match="Object arrays cannot be loaded|forbidden"):
        with ShardStore._open_validated(path):
            pass


def test_catalog_detects_corrupt_registered_shard(tmp_path: Path) -> None:
    store = ShardStore(tmp_path / "processed")
    catalog_path = tmp_path / "catalog.sqlite"
    record = _trajectory()
    asset = ManifestAsset("source.zip", record.asset_sha256, 1)
    info = store.write_trajectories([record])
    with Catalog(catalog_path, store) as catalog:
        catalog.register_asset(_manifest(asset), asset)
        catalog.commit_trajectory_shard(info, [record])
    path = store.resolve(info.relative_path)
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with Catalog(catalog_path, store) as catalog:
        with pytest.raises(ValueError, match="SHA-256"):
            catalog.verify()


def test_malformed_typed_sgf_properties_are_rejected_without_stopping(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    malformed = (
        VALID_SGF.replace(b"HA[0]", b"HA[x]"),
        VALID_SGF.replace(b"KM[0]", b"KM[x]"),
        VALID_SGF.replace(b"SZ[19]", b"SZ[x]"),
        VALID_SGF.replace(b";B[aa]", b";B[zz]"),
        VALID_SGF.replace(b"RE[0]", b"RE[B+nonsense]"),
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("games.sgfs", b"\n".join((*malformed, VALID_SGF)))
    payload = archive_path.read_bytes()
    asset = ManifestAsset("source.zip", hashlib.sha256(payload).hexdigest(), len(payload))
    events = list(KataGoSgfImporter().import_asset(_manifest(asset), asset, tmp_path))
    assert [event.kind for event in events] == ["reject"] * len(malformed) + ["trajectory"]
    assert all(
        event.reason_code in {"invalid_sgf", "invalid_move", "unsupported_result"}
        for event in events[:-1]
    )


def test_capped_import_counts_only_new_games_and_finishes_only_at_eof(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("games.sgfs", b"\n".join((VALID_SGF,) * 3))
    payload = archive_path.read_bytes()
    asset = ManifestAsset("source.zip", hashlib.sha256(payload).hexdigest(), len(payload))
    manifest_path = tmp_path / "manifest.json"
    _manifest(asset).save(manifest_path)
    store_root = tmp_path / "processed"
    catalog_path = tmp_path / "catalog.sqlite"

    statuses = []
    for _ in range(3):
        summary = import_manifest(
            manifest_path,
            tmp_path,
            store_root,
            catalog_path,
            max_accepted=1,
            target_shard_bytes=1024,
        )
        assert summary.accepted == 1
        with Catalog(catalog_path, ShardStore(store_root)) as catalog:
            statuses.append(catalog.asset_status(asset.sha256))
    assert statuses == ["partial", "partial", "imported"]
    with Catalog(catalog_path, ShardStore(store_root)) as catalog:
        snapshot_id = catalog.create_snapshot(seed=1, validation_fraction=0.0)
        assert len(catalog.snapshot_trajectories(snapshot_id)) == 3


def _snapshot_for_records(
    root: Path,
    records: list[TrajectoryRecord],
    pack_together: bool,
    annotation: AnnotationRecord | None = None,
) -> str:
    store = ShardStore(root / "processed")
    asset = ManifestAsset("source.zip", records[0].asset_sha256, 1)
    with Catalog(root / "catalog.sqlite", store) as catalog:
        catalog.register_asset(_manifest(asset), asset)
        groups = [records] if pack_together else [[record] for record in records]
        for group in groups:
            info = store.write_trajectories(group)
            catalog.commit_trajectory_shard(info, group)
        if annotation is not None:
            info = store.write_annotations([annotation])
            catalog.commit_annotation_shard(info, [annotation])
        return catalog.create_snapshot(seed=3, validation_fraction=0.0)


def test_snapshot_identity_uses_logical_content_not_shard_packing(tmp_path: Path) -> None:
    first = _trajectory()
    second = dataclasses.replace(
        first,
        game_id="c" * 64,
        content_sha256="",
        ordinal=1,
    )
    packed = _snapshot_for_records(tmp_path / "packed", [first, second], True)
    split = _snapshot_for_records(tmp_path / "split", [first, second], False)
    assert packed == split

    changed = dataclasses.replace(first, content_sha256="", value_black=-1.0)
    changed_snapshot = _snapshot_for_records(tmp_path / "changed", [changed, second], True)
    assert changed_snapshot != packed

    annotation = AnnotationRecord(
        schema_version=RECORD_SCHEMA_VERSION,
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
    annotated = _snapshot_for_records(tmp_path / "annotated", [first, second], True, annotation)
    changed_annotation = dataclasses.replace(annotation, value=-0.25)
    changed_annotated = _snapshot_for_records(
        tmp_path / "changed-annotation",
        [first, second],
        True,
        changed_annotation,
    )
    assert annotated != changed_annotated


def test_snapshot_membership_is_built_inside_an_immediate_transaction(tmp_path: Path) -> None:
    record = _trajectory()
    store = ShardStore(tmp_path / "processed")
    asset = ManifestAsset("source.zip", record.asset_sha256, 1)
    with Catalog(tmp_path / "catalog.sqlite", store) as catalog:
        catalog.register_asset(_manifest(asset), asset)
        info = store.write_trajectories([record])
        catalog.commit_trajectory_shard(info, [record])
        statements = []
        catalog.connection.set_trace_callback(statements.append)
        catalog.create_snapshot(seed=6, validation_fraction=0.0)
    begin = next(index for index, sql in enumerate(statements) if "BEGIN IMMEDIATE" in sql)
    selection = next(
        index
        for index, sql in enumerate(statements)
        if "FROM trajectories WHERE deleted=0" in sql
    )
    membership = next(
        index
        for index, sql in enumerate(statements)
        if "INSERT INTO snapshot_trajectories" in sql
    )
    commit = next(index for index, sql in enumerate(statements) if sql.strip() == "COMMIT")
    assert begin < selection < membership < commit


def test_shard_local_sampling_is_position_weighted_and_deterministic() -> None:
    small = TrajectoryLocator("a" * 64, "1" * 64, "s1", "one.npz", 0, 0, 1)
    large = TrajectoryLocator("b" * 64, "2" * 64, "s2", "two.npz", 0, 0, 3)
    index = SnapshotPositionIndex((small, large), (), "none")
    first_rng = np.random.default_rng(12)
    second_rng = np.random.default_rng(12)
    first_draws = [index.draw_batch(4, first_rng) for _ in range(4000)]
    second_draws = [index.draw_batch(4, second_rng) for _ in range(4000)]
    assert [draw[0].trajectory.game_id for draw in first_draws] == [
        draw[0].trajectory.game_id for draw in second_draws
    ]
    assert all(len({ref.trajectory.shard_sha256 for ref in draw}) == 1 for draw in first_draws)
    small_fraction = sum(draw[0].trajectory.game_id == small.game_id for draw in first_draws) / 4000
    assert small_fraction == pytest.approx(0.25, abs=0.03)


def test_catalog_source_reads_one_trajectory_shard_per_microbatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _trajectory()
    second = dataclasses.replace(
        first,
        game_id="c" * 64,
        content_sha256="",
        ordinal=1,
    )
    store = ShardStore(tmp_path / "processed")
    asset = ManifestAsset("source.zip", first.asset_sha256, 1)
    with Catalog(tmp_path / "catalog.sqlite", store) as catalog:
        catalog.register_asset(_manifest(asset), asset)
        for record in (first, second):
            info = store.write_trajectories([record])
            catalog.commit_trajectory_shard(info, [record])
        snapshot_id = catalog.create_snapshot(seed=4, validation_fraction=0.0)

    calls = []
    original = ShardStore.read_verified_trajectories

    def counting_read(self, relative_path: str, expected_sha256: str):
        calls.append(expected_sha256)
        return original(self, relative_path, expected_sha256)

    monkeypatch.setattr(ShardStore, "read_verified_trajectories", counting_read)
    with CatalogBatchSource(tmp_path / "catalog.sqlite", store.root, snapshot_id) as source:
        assert calls == []
        source.next_batch(16, np.random.default_rng(8))
        assert len(calls) == 1


def test_v1_catalog_is_rejected_with_rebuild_guidance(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.sqlite"
    connection = sqlite3.connect(catalog_path)
    connection.execute("CREATE TABLE catalog_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.execute("INSERT INTO catalog_meta VALUES('schema_version','1')")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="rebuild.*v2"):
        Catalog(catalog_path, ShardStore(tmp_path / "processed"))


def test_v1_records_and_shards_are_rejected_with_rebuild_guidance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="record schema"):
        dataclasses.replace(_trajectory(), schema_version=1, content_sha256="")
    path = tmp_path / "v1.npz"
    np.savez(
        path,
        schema_version=np.asarray(1, dtype=np.int32),
        record_schema_version=np.asarray(1, dtype=np.int32),
        kind=np.asarray(1, dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="rebuild.*v2"):
        with ShardStore._open_validated(path):
            pass


def test_annotation_content_hash_detects_logical_tampering(tmp_path: Path) -> None:
    record = _trajectory()
    annotation = AnnotationRecord(
        schema_version=RECORD_SCHEMA_VERSION,
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


def test_annotation_shards_are_loaded_at_most_once_per_microbatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record = _trajectory()
    store = ShardStore(tmp_path / "processed")
    asset = ManifestAsset("source.zip", record.asset_sha256, 1)
    with Catalog(tmp_path / "catalog.sqlite", store) as catalog:
        catalog.register_asset(_manifest(asset), asset)
        trajectory_info = store.write_trajectories([record])
        catalog.commit_trajectory_shard(trajectory_info, [record])
        for ply in range(record.trainable_position_count):
            annotation = AnnotationRecord(
                schema_version=RECORD_SCHEMA_VERSION,
                game_id=record.game_id,
                ply=ply,
                teacher_fingerprint="teacher-v1",
                policy_actions=(),
                policy_values=(),
                value=0.0,
                value_available=False,
                score_margin=0.0,
                score_available=False,
                ownership=(0.0,) * BOARD_AREA,
                ownership_available=False,
            )
            annotation_info = store.write_annotations([annotation])
            catalog.commit_annotation_shard(annotation_info, [annotation])
        snapshot_id = catalog.create_snapshot(seed=5, validation_fraction=0.0)

    calls = []
    original = ShardStore.read_verified_annotations

    def counting_read(self, relative_path: str, expected_sha256: str):
        calls.append(expected_sha256)
        return original(self, relative_path, expected_sha256)

    monkeypatch.setattr(ShardStore, "read_verified_annotations", counting_read)
    with CatalogBatchSource(
        tmp_path / "catalog.sqlite",
        store.root,
        snapshot_id,
        annotation_mode="require_exact",
        teacher_fingerprint="teacher-v1",
    ) as source:
        source.next_batch(64, np.random.default_rng(2))
    assert calls
    assert len(calls) == len(set(calls))
