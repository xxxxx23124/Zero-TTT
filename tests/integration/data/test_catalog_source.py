from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from zero_ttt.versioning import RECORD_SCHEMA
from zero_ttt_dataset import PortableSnapshotBatchSource, ShardStore, SnapshotManifest
from zero_ttt_dataset.locators import AnnotationLocator, TrajectoryLocator
from zero_ttt_dataset.materialization import TrajectoryBatchMaterializer
from zero_ttt_dataset.records import AnnotationRecord, TrajectoryRecord
from zero_ttt_dataset.sampling import SnapshotPositionIndex


def _locator(record: TrajectoryRecord, info, index: int = 0) -> TrajectoryLocator:
    return TrajectoryLocator(
        game_id=record.game_id,
        content_sha256=record.content_sha256,
        shard_sha256=info.sha256,
        relative_path=info.relative_path,
        game_index=index,
        trainable_start_ply=record.trainable_start_ply,
        trainable_positions=record.trainable_position_count,
    )


def _manifest(
    trajectories: tuple[TrajectoryLocator, ...],
    annotations: tuple[AnnotationLocator, ...] = (),
) -> SnapshotManifest:
    return SnapshotManifest.from_locators(
        snapshot_id="portable-test",
        seed=7,
        split="train",
        validation_fraction=0.0,
        source_kind="external",
        task_id="",
        trajectories=trajectories,
        annotations=annotations,
    )


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
    fraction = sum(draw[0].trajectory.game_id == small.game_id for draw in first_draws) / 4000
    assert fraction == pytest.approx(0.25, abs=0.03)


def test_portable_source_reads_one_trajectory_shard_per_microbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trajectory_factory
) -> None:
    first = trajectory_factory()
    second = dataclasses.replace(first, game_id="c" * 64, content_sha256="", ordinal=1)
    store = ShardStore(tmp_path / "shards")
    first_info = store.write_trajectories([first])
    second_info = store.write_trajectories([second])
    manifest = _manifest((_locator(first, first_info), _locator(second, second_info)))
    calls: list[str] = []
    original = ShardStore.read_verified_trajectories

    def counting_read(self, relative_path: str, expected_sha256: str):
        calls.append(expected_sha256)
        return original(self, relative_path, expected_sha256)

    monkeypatch.setattr(ShardStore, "read_verified_trajectories", counting_read)
    with PortableSnapshotBatchSource(manifest, store.root) as source:
        source.next_batch(16, np.random.default_rng(8))
    assert len(calls) == 1


def test_annotation_shard_is_loaded_once_per_microbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trajectory_factory
) -> None:
    record = trajectory_factory()
    store = ShardStore(tmp_path / "shards")
    trajectory_info = store.write_trajectories([record])
    annotations = [
        AnnotationRecord(
            schema_version=RECORD_SCHEMA.current,
            game_id=record.game_id,
            ply=ply,
            teacher_fingerprint="teacher-v1",
            policy_actions=(),
            policy_values=(),
            value=0.0,
            value_available=False,
            score_margin=0.0,
            score_available=False,
            ownership=(0.0,) * 361,
            ownership_available=False,
        )
        for ply in range(record.trainable_position_count)
    ]
    annotation_info = store.write_annotations(annotations)
    locators = tuple(
        AnnotationLocator(
            game_id=item.game_id,
            content_sha256=item.content_sha256,
            ply=item.ply,
            teacher_fingerprint=item.teacher_fingerprint,
            shard_sha256=annotation_info.sha256,
            relative_path=annotation_info.relative_path,
            record_index=index,
        )
        for index, item in enumerate(annotations)
    )
    manifest = _manifest((_locator(record, trajectory_info),), locators)
    calls: list[str] = []
    original = ShardStore.read_verified_annotations

    def counting_read(self, relative_path: str, expected_sha256: str):
        calls.append(expected_sha256)
        return original(self, relative_path, expected_sha256)

    monkeypatch.setattr(ShardStore, "read_verified_annotations", counting_read)
    with PortableSnapshotBatchSource(
        manifest,
        store.root,
        annotation_mode="require_exact",
        teacher_fingerprint="teacher-v1",
    ) as source:
        source.next_batch(64, np.random.default_rng(2))
    assert calls == [annotation_info.sha256]


def test_multiple_positions_from_one_game_are_replayed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trajectory_factory
) -> None:
    record = trajectory_factory()
    store = ShardStore(tmp_path / "shards")
    info = store.write_trajectories([record])
    manifest = _manifest((_locator(record, info),))
    calls = 0
    original = TrajectoryBatchMaterializer._initial_state

    def counting_initial_state(sample):
        nonlocal calls
        calls += 1
        return original(sample)

    monkeypatch.setattr(
        TrajectoryBatchMaterializer, "_initial_state", staticmethod(counting_initial_state)
    )
    with PortableSnapshotBatchSource(manifest, store.root) as source:
        source.next_batch(32, np.random.default_rng(9))
    assert calls == 1
