"""Private transaction composition; never exported as a cross-service facade."""

from __future__ import annotations

from pathlib import Path

from zero_ttt_dataset.shards import ShardStore

from zero_ttt_data.maintenance import ShardLifecycle
from zero_ttt_data.repository import CatalogRepository
from zero_ttt_data.session import CatalogSession
from zero_ttt_data.snapshots import SnapshotService


class DataUnitOfWork:
    def __init__(self, database_path: str | Path, shard_root: str | Path) -> None:
        self.store = ShardStore(shard_root)
        self.session = CatalogSession(database_path)
        self.repository = CatalogRepository(self.session.connection, self.store)
        self.snapshots = SnapshotService(self.session.connection)
        self.lifecycle = ShardLifecycle(self.session.connection, self.store)

    def __enter__(self) -> DataUnitOfWork:
        return self

    def __exit__(self, *_: object) -> None:
        self.session.close()
