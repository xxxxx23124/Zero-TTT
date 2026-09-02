from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from zero_ttt.versioning import SOURCE_MANIFEST_SCHEMA
from zero_ttt_data.importers import KataGoSgfImporter
from zero_ttt_data.importing import import_source
from zero_ttt_data.manifest import ManifestAsset, SourceManifest
from zero_ttt_data.snapshots import SnapshotSpec
from zero_ttt_data.unit_of_work import DataUnitOfWork


def test_source_manifest_round_trip_and_previous_schema_rejection(
    tmp_path: Path,
    manifest_factory,
) -> None:
    asset = ManifestAsset("source.zip", "a" * 64, 1)
    manifest = manifest_factory(asset)
    path = tmp_path / "manifest.json"
    manifest.save(path)
    assert SourceManifest.load(path) == manifest
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = SOURCE_MANIFEST_SCHEMA.current - 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"source manifest.*expected v1"):
        SourceManifest.load(path)


def test_katago_importer_streams_records_and_structured_rejections(
    tmp_path: Path,
    valid_sgf,
    manifest_factory,
) -> None:
    archive_path = tmp_path / "source.zip"
    unsupported_size = valid_sgf.replace(b"SZ[19]", b"SZ[13]")
    cross_rules = valid_sgf.replace(
        b"koPOSITIONALscoreAREAtaxNONEsui1",
        b"koSIMPLEscoreAREAtaxNONEsui0",
    )
    cleanup = valid_sgf[:-1] + b";B[cc])"
    illegal = valid_sgf.replace(b";W[bb]", b";W[aa]")
    setup = valid_sgf.replace(b"HA[0]", b"HA[0]AB[cc]")
    fork = valid_sgf.replace(b"mode=normal", b"mode=fork")
    variation = valid_sgf.replace(b";W[bb]", b"(;W[bb])(;W[cc])")
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "net/sgfs/games.sgfs",
            b"\n".join(
                (
                    valid_sgf,
                    cross_rules,
                    unsupported_size,
                    cleanup,
                    illegal,
                    setup,
                    fork,
                    variation,
                )
            ),
        )
    payload = archive_path.read_bytes()
    asset = ManifestAsset("source.zip", hashlib.sha256(payload).hexdigest(), len(payload))
    manifest = manifest_factory(asset)

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
        list(
            KataGoSgfImporter().import_asset(
                manifest_factory(altered),
                altered,
                tmp_path,
            )
        )


def test_malformed_typed_sgf_properties_are_rejected_without_stopping(
    tmp_path: Path,
    valid_sgf,
    manifest_factory,
) -> None:
    archive_path = tmp_path / "source.zip"
    malformed = (
        valid_sgf.replace(b"HA[0]", b"HA[x]"),
        valid_sgf.replace(b"KM[0]", b"KM[x]"),
        valid_sgf.replace(b"SZ[19]", b"SZ[x]"),
        valid_sgf.replace(b";B[aa]", b";B[zz]"),
        valid_sgf.replace(b"RE[0]", b"RE[B+nonsense]"),
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("games.sgfs", b"\n".join((*malformed, valid_sgf)))
    payload = archive_path.read_bytes()
    asset = ManifestAsset("source.zip", hashlib.sha256(payload).hexdigest(), len(payload))
    events = list(KataGoSgfImporter().import_asset(manifest_factory(asset), asset, tmp_path))
    assert [event.kind for event in events] == ["reject"] * len(malformed) + ["trajectory"]
    assert all(
        event.reason_code in {"invalid_sgf", "invalid_move", "unsupported_result"}
        for event in events[:-1]
    )


def test_capped_import_counts_only_new_games_and_finishes_only_at_eof(
    tmp_path: Path,
    valid_sgf,
    manifest_factory,
) -> None:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("games.sgfs", b"\n".join((valid_sgf,) * 3))
    payload = archive_path.read_bytes()
    asset = ManifestAsset("source.zip", hashlib.sha256(payload).hexdigest(), len(payload))
    manifest_path = tmp_path / "manifest.json"
    manifest_factory(asset).save(manifest_path)
    store_root = tmp_path / "processed"
    catalog_path = tmp_path / "catalog.sqlite"

    statuses = []
    for _ in range(3):
        summary = import_source(
            manifest_path=manifest_path,
            source_root=tmp_path,
            shard_root=store_root,
            database_path=catalog_path,
            max_accepted=1,
            target_shard_bytes=1024,
        )
        assert summary.accepted == 1
        with DataUnitOfWork(catalog_path, store_root) as unit:
            statuses.append(unit.repository.asset_status(asset.sha256))
    assert statuses == ["partial", "partial", "imported"]
    with DataUnitOfWork(catalog_path, store_root) as unit:
        snapshot_id = unit.snapshots.create(
            SnapshotSpec(seed=1, split="train", validation_fraction=0.0)
        )
        assert len(unit.snapshots.trajectories(snapshot_id)) == 3


def test_import_soft_stop_flushes_an_atomic_partial_shard_and_resumes(
    tmp_path: Path,
    valid_sgf,
    manifest_factory,
) -> None:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("games.sgfs", b"\n".join((valid_sgf,) * 3))
    payload = archive_path.read_bytes()
    asset = ManifestAsset("source.zip", hashlib.sha256(payload).hexdigest(), len(payload))
    manifest_path = tmp_path / "manifest.json"
    manifest_factory(asset).save(manifest_path)
    calls = 0

    def stop_requested() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 4

    first = import_source(
        manifest_path=manifest_path,
        source_root=tmp_path,
        shard_root=tmp_path / "processed",
        database_path=tmp_path / "catalog.sqlite",
        max_accepted=None,
        target_shard_bytes=1024 * 1024,
        stop_requested=stop_requested,
    )
    assert 0 < first.accepted < 3
    with DataUnitOfWork(tmp_path / "catalog.sqlite", tmp_path / "processed") as unit:
        assert unit.lifecycle.recover() == ()
        assert unit.repository.asset_status(asset.sha256) == "partial"

    resumed = import_source(
        manifest_path=manifest_path,
        source_root=tmp_path,
        shard_root=tmp_path / "processed",
        database_path=tmp_path / "catalog.sqlite",
        max_accepted=None,
    )
    assert first.accepted + resumed.accepted == 3
