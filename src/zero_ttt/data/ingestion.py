"""Shared trajectory buffering and durable shard registration."""

from __future__ import annotations

from zero_ttt.data.catalog import Catalog
from zero_ttt.data.records import ImportEvent, TrajectoryRecord
from zero_ttt.data.shards import ShardInfo, ShardStore

DEFAULT_TARGET_SHARD_BYTES = 128 * 1024 * 1024


def estimate_trajectory_bytes(record: TrajectoryRecord) -> int:
    """Conservative shared estimate used only to choose shard boundaries."""

    return (
        4096
        + len(record.moves) * 2
        + len(record.policy_actions) * 6
        + record.trainable_position_count * 48
    )


class TrajectoryShardSink:
    def __init__(
        self,
        store: ShardStore,
        catalog: Catalog,
        target_shard_bytes: int,
    ) -> None:
        if target_shard_bytes <= 0:
            raise ValueError("target_shard_bytes must be positive")
        self.store = store
        self.catalog = catalog
        self.target_shard_bytes = target_shard_bytes
        self._pending: list[TrajectoryRecord] = []
        self._pending_rejections: list[ImportEvent] = []
        self._pending_bytes = 0
        self.shard_count = 0
        self.position_count = 0

    def add_rejection(self, event: ImportEvent) -> None:
        if event.kind != "reject":
            raise ValueError("trajectory sink only accepts rejection events here")
        self._pending_rejections.append(event)

    def append(self, record: TrajectoryRecord) -> ShardInfo | None:
        estimate = estimate_trajectory_bytes(record)
        flushed = None
        if self._pending and self._pending_bytes + estimate > self.target_shard_bytes:
            flushed = self.flush()
        self._pending.append(record)
        self._pending_bytes += estimate
        return flushed

    def flush(self) -> ShardInfo | None:
        if not self._pending:
            if self._pending_rejections:
                self.catalog.record_rejections(self._pending_rejections)
                self._pending_rejections = []
            return None
        info = self.store.write_trajectories(self._pending)
        self.catalog.commit_trajectory_shard(
            info,
            self._pending,
            self._pending_rejections,
        )
        self._pending = []
        self._pending_rejections = []
        self._pending_bytes = 0
        self.shard_count += 1
        self.position_count += info.position_count
        return info
