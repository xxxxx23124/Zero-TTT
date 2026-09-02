"""Crash-safe shard verification, recovery, and garbage collection."""

from __future__ import annotations

import sqlite3
import time

from zero_ttt._io import fsync_directory
from zero_ttt_dataset.shards import ShardStore

STALE_TEMPORARY_SECONDS = 24 * 60 * 60


class ShardLifecycle:
    def __init__(self, connection: sqlite3.Connection, store: ShardStore) -> None:
        self.connection = connection
        self.store = store

    def verify(self) -> None:
        rows = self.connection.execute(
            "SELECT shard_sha256,relative_path FROM shards WHERE deleted=0 ORDER BY shard_sha256"
        ).fetchall()
        for row in rows:
            self.store.verify(row["relative_path"], row["shard_sha256"])

    def recover(self) -> tuple[str, ...]:
        self._remove_stale_temporary_files()
        self._finish_tombstoned_deletions()
        registered = {
            row["relative_path"]
            for row in self.connection.execute("SELECT relative_path FROM shards")
        }
        orphans = tuple(
            sorted(
                path.relative_to(self.store.root).as_posix()
                for path in self.store.root.glob("*/*.npz")
                if path.relative_to(self.store.root).as_posix() not in registered
            )
        )
        touched_directories = set()
        for relative_path in orphans:
            path = self.store.resolve(relative_path)
            try:
                path.unlink()
                touched_directories.add(path.parent)
            except FileNotFoundError:
                continue
        for directory in touched_directories:
            fsync_directory(directory)
        self.verify()
        return orphans

    def _remove_stale_temporary_files(self) -> None:
        cutoff = time.time() - STALE_TEMPORARY_SECONDS
        touched_directories = set()
        for temporary in self.store.root.rglob(".shard-*.tmp"):
            try:
                if temporary.stat().st_mtime > cutoff:
                    continue
                temporary.unlink()
                touched_directories.add(temporary.parent)
            except FileNotFoundError:
                continue
        for directory in touched_directories:
            fsync_directory(directory)

    def _finish_tombstoned_deletions(self) -> None:
        rows = self.connection.execute(
            "SELECT relative_path FROM shards WHERE deleted=1 ORDER BY relative_path"
        ).fetchall()
        touched_directories = set()
        for row in rows:
            path = self.store.resolve(row["relative_path"])
            try:
                path.unlink()
                touched_directories.add(path.parent)
            except FileNotFoundError:
                continue
        for directory in touched_directories:
            fsync_directory(directory)

    def mark_trajectory_deleted(self, game_id: str) -> None:
        with self.connection:
            changed = self.connection.execute(
                "UPDATE trajectories SET deleted=1 WHERE game_id=?", (game_id,)
            ).rowcount
            if changed:
                self.connection.execute(
                    "UPDATE annotations SET deleted=1 WHERE game_id=?", (game_id,)
                )
        if not changed:
            raise KeyError(game_id)

    def mark_annotation_deleted(self, game_id: str, ply: int, teacher_fingerprint: str) -> None:
        with self.connection:
            changed = self.connection.execute(
                """
                UPDATE annotations SET deleted=1
                WHERE game_id=? AND ply=? AND teacher_fingerprint=?
                """,
                (game_id, ply, teacher_fingerprint),
            ).rowcount
        if not changed:
            raise KeyError((game_id, ply, teacher_fingerprint))

    def pin(self, owner: str, shard_sha256: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO shard_pins(owner,shard_sha256) VALUES(?,?)",
                (owner, shard_sha256),
            )

    def unpin(self, owner: str, shard_sha256: str) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM shard_pins WHERE owner=? AND shard_sha256=?",
                (owner, shard_sha256),
            )

    def garbage_collect(self) -> tuple[str, ...]:
        rows = self.connection.execute(self._garbage_query()).fetchall()
        removed = []
        for row in rows:
            with self.connection:
                changed = self.connection.execute(
                    "UPDATE shards SET deleted=1 WHERE shard_sha256=? AND deleted=0",
                    (row["shard_sha256"],),
                ).rowcount
            if not changed:
                continue
            path = self.store.resolve(row["relative_path"])
            try:
                path.unlink()
                fsync_directory(path.parent)
            except FileNotFoundError:
                pass
            removed.append(row["relative_path"])
        return tuple(removed)

    @staticmethod
    def _garbage_query() -> str:
        return """
            SELECT s.shard_sha256,s.relative_path,s.kind
            FROM shards s
            WHERE s.deleted=0
              AND NOT EXISTS(SELECT 1 FROM shard_pins p WHERE p.shard_sha256=s.shard_sha256)
              AND (
                (s.kind='trajectory'
                 AND NOT EXISTS(
                    SELECT 1 FROM trajectories t
                    WHERE t.shard_sha256=s.shard_sha256 AND t.deleted=0
                 )
                 AND NOT EXISTS(
                    SELECT 1 FROM snapshot_trajectories st
                    JOIN trajectories t ON t.game_id=st.game_id
                    WHERE t.shard_sha256=s.shard_sha256
                 ))
                OR
                (s.kind='annotation'
                 AND NOT EXISTS(
                    SELECT 1 FROM annotations a
                    WHERE a.shard_sha256=s.shard_sha256 AND a.deleted=0
                 )
                 AND NOT EXISTS(
                    SELECT 1 FROM snapshot_annotations sa
                    JOIN annotations a
                      ON a.game_id=sa.game_id AND a.ply=sa.ply
                     AND a.teacher_fingerprint=sa.teacher_fingerprint
                    WHERE a.shard_sha256=s.shard_sha256
                 ))
              )
        """
