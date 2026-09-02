from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from zero_ttt_dataset import (
    PortableMixtureBatchSource,
    PortableSnapshotBatchSource,
    ShardStore,
    SnapshotManifest,
)
from zero_ttt_dataset.locators import TrajectoryLocator


def _source(store: ShardStore, snapshot_id: str, record) -> PortableSnapshotBatchSource:
    info = store.write_trajectories([record])
    locator = TrajectoryLocator(
        game_id=record.game_id,
        content_sha256=record.content_sha256,
        shard_sha256=info.sha256,
        relative_path=info.relative_path,
        game_index=0,
        trainable_start_ply=record.trainable_start_ply,
        trainable_positions=record.trainable_position_count,
    )
    manifest = SnapshotManifest.from_locators(
        snapshot_id=snapshot_id,
        seed=1,
        split="train",
        validation_fraction=0.0,
        source_kind="external",
        task_id="",
        trajectories=(locator,),
    )
    return PortableSnapshotBatchSource(manifest, store.root)


def test_portable_mixture_has_deterministic_weighted_component_draws(
    tmp_path: Path, trajectory_factory
) -> None:
    first_record = trajectory_factory()
    second_record = dataclasses.replace(
        first_record,
        game_id="c" * 64,
        asset_sha256="d" * 64,
        content_sha256="",
        value_black=-1.0,
    )
    store = ShardStore(tmp_path / "shards")
    first = _source(store, "first", first_record)
    second = _source(store, "second", second_record)

    class CountingSource:
        def __init__(self, source):
            self.source = source
            self.manifest = source.manifest
            self.calls = 0

        def next_batch(self, batch_size, rng):
            self.calls += 1
            return self.source.next_batch(batch_size, rng)

        def close(self):
            self.source.close()

    counted_first = CountingSource(first)
    counted_second = CountingSource(second)
    mixture = PortableMixtureBatchSource(((counted_first, 1.0), (counted_second, 3.0)))
    rng = np.random.default_rng(21)
    for _ in range(2000):
        mixture.next_batch(1, rng)
    second_fraction = counted_second.calls / 2000
    assert second_fraction == pytest.approx(0.75, abs=0.04)
    assert mixture.component_snapshot_ids == ("first", "second")
    assert len(mixture.sampling_config_sha256) == 64
