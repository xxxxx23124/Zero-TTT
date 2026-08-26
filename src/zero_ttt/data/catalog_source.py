"""Snapshot-bound, shard-local training batches from cataloged records."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from zero_ttt.config import GameConfig
from zero_ttt.data.catalog import AnnotationLocator, Catalog, TrajectoryLocator
from zero_ttt.data.contracts import TrainBatch
from zero_ttt.data.records import AnnotationRecord, TrajectoryRecord
from zero_ttt.data.shards import ShardStore
from zero_ttt.game.features import GLOBAL_FEATURES, POINT_FEATURES, encode_position
from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA, BOARD_SIZE, Color
from zero_ttt.game.state import GameState
from zero_ttt.game.symmetry import augment_sample


AnnotationMode = Literal["none", "prefer_exact", "require_exact"]


@dataclass(frozen=True, slots=True)
class _SampleReference:
    trajectory: TrajectoryLocator
    local_position: int
    annotation: AnnotationLocator | None


@dataclass(frozen=True, slots=True)
class _ShardPositionBucket:
    shard_sha256: str
    trajectories: tuple[TrajectoryLocator, ...]
    cumulative_positions: np.ndarray
    exact_references: tuple[_SampleReference, ...]
    position_count: int


class SnapshotPositionIndex:
    """Pure, immutable position index with weighted shard-local sampling."""

    def __init__(
        self,
        trajectories: tuple[TrajectoryLocator, ...],
        annotations: tuple[AnnotationLocator, ...],
        annotation_mode: AnnotationMode,
    ) -> None:
        if not trajectories:
            raise ValueError("snapshot has no trajectories")
        trajectory_by_game = {locator.game_id: locator for locator in trajectories}
        annotation_by_position = {
            (locator.game_id, locator.ply): locator for locator in annotations
        }
        grouped: OrderedDict[str, list[TrajectoryLocator] | list[_SampleReference]] = (
            OrderedDict()
        )
        if annotation_mode == "require_exact":
            for annotation in annotations:
                trajectory = trajectory_by_game.get(annotation.game_id)
                if trajectory is None:
                    continue
                end_ply = trajectory.trainable_start_ply + trajectory.trainable_positions
                if not trajectory.trainable_start_ply <= annotation.ply < end_ply:
                    continue
                grouped.setdefault(trajectory.shard_sha256, []).append(
                    _SampleReference(
                        trajectory=trajectory,
                        local_position=annotation.ply - trajectory.trainable_start_ply,
                        annotation=annotation,
                    )
                )
        else:
            for trajectory in trajectories:
                if trajectory.trainable_positions:
                    grouped.setdefault(trajectory.shard_sha256, []).append(trajectory)

        buckets = []
        for shard_sha256, entries in grouped.items():
            if annotation_mode == "require_exact":
                exact = tuple(entry for entry in entries if isinstance(entry, _SampleReference))
                if not exact:
                    continue
                buckets.append(
                    _ShardPositionBucket(
                        shard_sha256=shard_sha256,
                        trajectories=(),
                        cumulative_positions=np.empty(0, dtype=np.int64),
                        exact_references=exact,
                        position_count=len(exact),
                    )
                )
                continue
            shard_trajectories = tuple(
                entry for entry in entries if isinstance(entry, TrajectoryLocator)
            )
            cumulative = np.cumsum(
                [locator.trainable_positions for locator in shard_trajectories],
                dtype=np.int64,
            )
            buckets.append(
                _ShardPositionBucket(
                    shard_sha256=shard_sha256,
                    trajectories=shard_trajectories,
                    cumulative_positions=cumulative,
                    exact_references=(),
                    position_count=int(cumulative[-1]),
                )
            )
        if not buckets:
            if annotation_mode == "require_exact":
                raise ValueError("snapshot has no positions for the required teacher")
            raise ValueError("snapshot has no trainable positions")
        self._buckets = tuple(buckets)
        self._bucket_cumulative = np.cumsum(
            [bucket.position_count for bucket in buckets],
            dtype=np.int64,
        )
        self.position_count = int(self._bucket_cumulative[-1])
        self.annotation_mode = annotation_mode
        self._annotation_by_position = annotation_by_position

    def draw_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[_SampleReference, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shard_draw = int(rng.integers(0, self.position_count))
        bucket_index = int(np.searchsorted(self._bucket_cumulative, shard_draw, side="right"))
        bucket = self._buckets[bucket_index]
        draws = rng.integers(0, bucket.position_count, size=batch_size)
        if self.annotation_mode == "require_exact":
            return tuple(bucket.exact_references[int(draw)] for draw in draws)

        references = []
        for draw_value in draws:
            draw = int(draw_value)
            trajectory_index = int(
                np.searchsorted(bucket.cumulative_positions, draw, side="right")
            )
            previous = (
                0
                if trajectory_index == 0
                else int(bucket.cumulative_positions[trajectory_index - 1])
            )
            trajectory = bucket.trajectories[trajectory_index]
            local_position = draw - previous
            ply = trajectory.trainable_start_ply + local_position
            annotation = self._annotation_by_position.get((trajectory.game_id, ply))
            references.append(_SampleReference(trajectory, local_position, annotation))
        return tuple(references)


class TrajectoryBatchMaterializer:
    """Load sampled shards once and turn logical records into a TrainBatch."""

    def __init__(self, store: ShardStore, shard_cache_size: int) -> None:
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive")
        self.store = store
        self.shard_cache_size = shard_cache_size
        self._trajectory_cache: OrderedDict[str, tuple[TrajectoryRecord, ...]] = OrderedDict()
        self._annotation_cache: OrderedDict[str, tuple[AnnotationRecord, ...]] = OrderedDict()

    def _cache_put(self, cache: OrderedDict, key: str, value: tuple) -> tuple:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.shard_cache_size:
            cache.popitem(last=False)
        return value

    def _trajectory_shard(
        self,
        locator: TrajectoryLocator,
    ) -> tuple[TrajectoryRecord, ...]:
        cached = self._trajectory_cache.get(locator.shard_sha256)
        if cached is not None:
            self._trajectory_cache.move_to_end(locator.shard_sha256)
            return cached
        records = self.store.read_verified_trajectories(
            locator.relative_path,
            locator.shard_sha256,
        )
        return self._cache_put(self._trajectory_cache, locator.shard_sha256, records)

    def _annotation_shard(
        self,
        locator: AnnotationLocator,
    ) -> tuple[AnnotationRecord, ...]:
        cached = self._annotation_cache.get(locator.shard_sha256)
        if cached is not None:
            self._annotation_cache.move_to_end(locator.shard_sha256)
            return cached
        records = self.store.read_verified_annotations(
            locator.relative_path,
            locator.shard_sha256,
        )
        return self._cache_put(self._annotation_cache, locator.shard_sha256, records)

    @staticmethod
    def _trajectory_record(
        records: tuple[TrajectoryRecord, ...],
        locator: TrajectoryLocator,
    ) -> TrajectoryRecord:
        try:
            record = records[locator.game_index]
        except IndexError as error:
            raise ValueError("catalog game_index is outside its shard") from error
        if (
            record.game_id != locator.game_id
            or record.content_sha256 != locator.content_sha256
        ):
            raise ValueError("catalog and trajectory shard identities disagree")
        return record

    @staticmethod
    def _annotation_record(
        records: tuple[AnnotationRecord, ...],
        locator: AnnotationLocator,
    ) -> AnnotationRecord:
        try:
            record = records[locator.record_index]
        except IndexError as error:
            raise ValueError("catalog annotation index is outside its shard") from error
        if (
            record.game_id != locator.game_id
            or record.content_sha256 != locator.content_sha256
            or record.ply != locator.ply
            or record.teacher_fingerprint != locator.teacher_fingerprint
        ):
            raise ValueError("catalog and annotation shard identities disagree")
        return record

    def materialize(
        self,
        references: tuple[_SampleReference, ...],
        rng: np.random.Generator,
    ) -> TrainBatch:
        if not references:
            raise ValueError("cannot materialize an empty batch")
        shard_ids = {reference.trajectory.shard_sha256 for reference in references}
        if len(shard_ids) != 1:
            raise ValueError("a shard-local microbatch must reference exactly one trajectory shard")
        trajectory_records = self._trajectory_shard(references[0].trajectory)
        annotation_shards: dict[str, tuple[AnnotationRecord, ...]] = {}
        for reference in references:
            locator = reference.annotation
            if locator is not None and locator.shard_sha256 not in annotation_shards:
                annotation_shards[locator.shard_sha256] = self._annotation_shard(locator)

        batch_size = len(references)
        boards = np.empty(
            (batch_size, POINT_FEATURES, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )
        globals_ = np.empty((batch_size, GLOBAL_FEATURES), dtype=np.float32)
        legal_masks = np.empty((batch_size, ACTION_SIZE), dtype=np.bool_)
        policies = np.zeros((batch_size, ACTION_SIZE), dtype=np.float32)
        values = np.empty(batch_size, dtype=np.float32)
        ownerships = np.empty((batch_size, BOARD_AREA), dtype=np.float32)
        score_margins = np.empty(batch_size, dtype=np.float32)
        value_masks = np.empty(batch_size, dtype=np.bool_)
        ownership_masks = np.empty(batch_size, dtype=np.bool_)
        score_masks = np.empty(batch_size, dtype=np.bool_)

        for index, reference in enumerate(references):
            record = self._trajectory_record(trajectory_records, reference.trajectory)
            ply = record.trainable_start_ply + reference.local_position
            state = GameState.new(
                GameConfig(
                    board_size=BOARD_SIZE,
                    komi_half_points=record.komi_half_points,
                    max_moves=record.max_moves,
                    history_length=8,
                )
            ).play_many(record.moves[:ply])
            features = encode_position(state)
            actions, masses = record.policy_at(reference.local_position)
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[np.asarray(actions, dtype=np.int64)] = np.asarray(masses, dtype=np.float32)
            perspective = 1.0 if state.to_play is Color.BLACK else -1.0
            value = np.float32(record.value_black * perspective)
            score = np.float32(record.score_margin_black * perspective)
            ownership = np.asarray(record.ownership_black, dtype=np.float32) * perspective
            value_mask = record.value_available
            score_mask = record.score_available
            ownership_mask = record.ownership_available

            if reference.annotation is not None:
                locator = reference.annotation
                annotation = self._annotation_record(
                    annotation_shards[locator.shard_sha256],
                    locator,
                )
                if annotation.policy_values:
                    policy.fill(0.0)
                    policy[np.asarray(annotation.policy_actions, dtype=np.int64)] = np.asarray(
                        annotation.policy_values,
                        dtype=np.float32,
                    )
                if annotation.value_available:
                    value = np.float32(annotation.value)
                    value_mask = True
                if annotation.score_available:
                    score = np.float32(annotation.score_margin)
                    score_mask = True
                if annotation.ownership_available:
                    ownership = np.asarray(annotation.ownership, dtype=np.float32)
                    ownership_mask = True

            augmented = augment_sample(
                features,
                policy,
                ownership,
                int(rng.integers(0, 8)),
            )
            boards[index] = augmented.features.board
            globals_[index] = augmented.features.global_features
            legal_masks[index] = augmented.features.legal
            policies[index] = augmented.policy
            values[index] = value
            ownerships[index] = augmented.ownership
            score_margins[index] = score
            value_masks[index] = value_mask
            ownership_masks[index] = ownership_mask
            score_masks[index] = score_mask
        return TrainBatch(
            board=boards,
            global_features=globals_,
            legal=legal_masks,
            policy=policies,
            value=values,
            ownership=ownerships,
            score_margin=score_margins,
            value_mask=value_masks,
            ownership_mask=ownership_masks,
            score_mask=score_masks,
        )


class CatalogBatchSource:
    """Uniform position marginals with one trajectory shard per microbatch."""

    def __init__(
        self,
        catalog_path: str | Path,
        store_root: str | Path,
        snapshot_id: str,
        annotation_mode: AnnotationMode = "none",
        teacher_fingerprint: str | None = None,
        shard_cache_size: int = 4,
    ) -> None:
        if annotation_mode not in {"none", "prefer_exact", "require_exact"}:
            raise ValueError("invalid annotation mode")
        if annotation_mode != "none" and not teacher_fingerprint:
            raise ValueError("annotation modes require an explicit teacher_fingerprint")
        self.store = ShardStore(store_root)
        self.catalog = Catalog(catalog_path, self.store)
        self.snapshot_id = snapshot_id
        self.annotation_mode = annotation_mode
        self.teacher_fingerprint = teacher_fingerprint
        try:
            trajectories = self.catalog.snapshot_trajectories(snapshot_id)
            annotations = (
                self.catalog.snapshot_annotations(snapshot_id, teacher_fingerprint)
                if teacher_fingerprint is not None
                else ()
            )
            self.position_index = SnapshotPositionIndex(
                trajectories,
                annotations,
                annotation_mode,
            )
        except BaseException:
            self.catalog.close()
            raise
        self.position_count = self.position_index.position_count
        self.materializer = TrajectoryBatchMaterializer(self.store, shard_cache_size)
        identity = json.dumps(
            {
                "snapshot_id": snapshot_id,
                "annotation_mode": annotation_mode,
                "teacher_fingerprint": teacher_fingerprint,
                "d4": True,
                "microbatch_shards": 1,
                "sampling": "weighted-shard-local-position-with-replacement-v2",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.sampling_config_sha256 = hashlib.sha256(identity).hexdigest()

    def close(self) -> None:
        self.catalog.close()

    def __enter__(self) -> "CatalogBatchSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def next_batch(self, batch_size: int, rng: np.random.Generator) -> TrainBatch:
        references = self.position_index.draw_batch(batch_size, rng)
        return self.materializer.materialize(references, rng)
