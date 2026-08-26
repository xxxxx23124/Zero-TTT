from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from zero_ttt.data.catalog import Catalog, TrajectoryLocator
from zero_ttt.data.catalog_source import CatalogBatchSource, SnapshotPositionIndex
from zero_ttt.data.manifest import ManifestAsset
from zero_ttt.data.records import AnnotationRecord, RECORD_SCHEMA_VERSION
from zero_ttt.data.shards import ShardStore
from zero_ttt.game.rules import BOARD_AREA


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
    assert all(
        len({ref.trajectory.shard_sha256 for ref in draw}) == 1
        for draw in first_draws
    )
    small_fraction = (
        sum(draw[0].trajectory.game_id == small.game_id for draw in first_draws) / 4000
    )
    assert small_fraction == pytest.approx(0.25, abs=0.03)


def test_catalog_source_reads_one_trajectory_shard_per_microbatch(
    tmp_path: Path,
    monkeypatch,
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
    store = ShardStore(tmp_path / "processed")
    asset = ManifestAsset("source.zip", first.asset_sha256, 1)
    with Catalog(tmp_path / "catalog.sqlite", store) as catalog:
        catalog.register_asset(manifest_factory(asset), asset)
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


def test_annotation_shards_are_loaded_at_most_once_per_microbatch(
    tmp_path: Path,
    monkeypatch,
    trajectory_factory,
    manifest_factory,
) -> None:
    record = trajectory_factory()
    store = ShardStore(tmp_path / "processed")
    asset = ManifestAsset("source.zip", record.asset_sha256, 1)
    with Catalog(tmp_path / "catalog.sqlite", store) as catalog:
        catalog.register_asset(manifest_factory(asset), asset)
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
