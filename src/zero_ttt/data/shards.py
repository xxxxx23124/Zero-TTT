"""Content-addressed, pickle-free NPZ trajectory and annotation shards."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from zero_ttt.data.records import AnnotationRecord, RECORD_SCHEMA_VERSION, TrajectoryRecord
SHARD_SCHEMA_VERSION = 2
ShardKind = Literal["trajectory", "annotation"]


@dataclass(frozen=True, slots=True)
class ShardInfo:
    kind: ShardKind
    sha256: str
    relative_path: str
    size_bytes: int
    record_count: int
    position_count: int


def _hex_matrix(values: list[str]) -> np.ndarray:
    if not values:
        return np.empty((0, 32), dtype=np.uint8)
    return np.asarray([list(bytes.fromhex(value)) for value in values], dtype=np.uint8)


def _pack_text(prefix: str, values: list[str]) -> dict[str, np.ndarray]:
    encoded = [value.encode("utf-8") for value in values]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    if encoded:
        offsets[1:] = np.cumsum([len(value) for value in encoded], dtype=np.int64)
    data = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
    return {f"{prefix}_data": data, f"{prefix}_offsets": offsets}


def _unpack_text(archive: np.lib.npyio.NpzFile, prefix: str) -> list[str]:
    data = archive[f"{prefix}_data"]
    offsets = archive[f"{prefix}_offsets"]
    return [
        bytes(data[offsets[index] : offsets[index + 1]]).decode("utf-8")
        for index in range(len(offsets) - 1)
    ]


def _concatenate_int(rows: list[tuple[int, ...]], dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    if rows:
        offsets[1:] = np.cumsum([len(row) for row in rows], dtype=np.int64)
    if offsets[-1]:
        values = np.asarray([value for row in rows for value in row], dtype=dtype)
    else:
        values = np.empty(0, dtype=dtype)
    return offsets, values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ShardStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.trajectory_dir = self.root / "trajectories"
        self.annotation_dir = self.root / "annotations"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_dir.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as error:
            raise ValueError("shard path escapes the store root") from error
        return path

    def write_trajectories(self, records: list[TrajectoryRecord]) -> ShardInfo:
        if not records:
            raise ValueError("cannot write an empty trajectory shard")
        move_offsets, moves = _concatenate_int([record.moves for record in records], np.int16)
        policy_game_offsets = np.zeros(len(records) + 1, dtype=np.int64)
        policy_game_offsets[1:] = np.cumsum(
            [record.trainable_position_count for record in records], dtype=np.int64
        )
        policy_row_offsets = [0]
        policy_actions: list[int] = []
        policy_values: list[float] = []
        for record in records:
            base = policy_row_offsets[-1]
            policy_row_offsets.extend(base + value for value in record.policy_row_offsets[1:])
            policy_actions.extend(record.policy_actions)
            policy_values.extend(record.policy_values)
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(SHARD_SCHEMA_VERSION, dtype=np.int32),
            "record_schema_version": np.asarray(RECORD_SCHEMA_VERSION, dtype=np.int32),
            "kind": np.asarray(1, dtype=np.uint8),
            "game_ids": _hex_matrix([record.game_id for record in records]),
            "content_hashes": _hex_matrix([record.content_sha256 for record in records]),
            "asset_hashes": _hex_matrix([record.asset_sha256 for record in records]),
            "ordinals": np.asarray([record.ordinal for record in records], dtype=np.int64),
            "komi_half_points": np.asarray(
                [record.komi_half_points for record in records], dtype=np.int16
            ),
            "max_moves": np.asarray([record.max_moves for record in records], dtype=np.int32),
            "move_offsets": move_offsets,
            "moves": moves,
            "trainable_start_ply": np.asarray(
                [record.trainable_start_ply for record in records], dtype=np.int32
            ),
            "policy_game_offsets": policy_game_offsets,
            "policy_row_offsets": np.asarray(policy_row_offsets, dtype=np.int64),
            "policy_actions": np.asarray(policy_actions, dtype=np.int16),
            "policy_values": np.asarray(policy_values, dtype=np.float32),
            "value_black": np.asarray(
                [record.value_black for record in records], dtype=np.float32
            ),
            "value_mask": np.asarray(
                [record.value_available for record in records], dtype=np.bool_
            ),
            "score_margin_black": np.asarray(
                [record.score_margin_black for record in records], dtype=np.float32
            ),
            "score_mask": np.asarray(
                [record.score_available for record in records], dtype=np.bool_
            ),
            "ownership_black": np.asarray(
                [record.ownership_black for record in records], dtype=np.float32
            ),
            "ownership_mask": np.asarray(
                [record.ownership_available for record in records], dtype=np.bool_
            ),
        }
        arrays.update(_pack_text("dataset_ids", [record.dataset_id for record in records]))
        arrays.update(_pack_text("member_paths", [record.member_path for record in records]))
        arrays.update(_pack_text("rules", [record.rules for record in records]))
        return self._write("trajectory", arrays, len(records), int(policy_game_offsets[-1]))

    def read_trajectories(self, info_or_path: ShardInfo | str | Path) -> tuple[TrajectoryRecord, ...]:
        path = self._coerce_path(info_or_path)
        with self._open_validated(path, expected_kind=1) as archive:
            dataset_ids = _unpack_text(archive, "dataset_ids")
            member_paths = _unpack_text(archive, "member_paths")
            rules = _unpack_text(archive, "rules")
            records = []
            count = len(archive["game_ids"])
            for index in range(count):
                move_start, move_end = archive["move_offsets"][index : index + 2]
                position_start, position_end = archive["policy_game_offsets"][index : index + 2]
                action_start = archive["policy_row_offsets"][position_start]
                action_end = archive["policy_row_offsets"][position_end]
                row_offsets = archive["policy_row_offsets"][position_start : position_end + 1]
                row_offsets = row_offsets - row_offsets[0]
                records.append(
                    TrajectoryRecord(
                        schema_version=int(archive["record_schema_version"]),
                        game_id=bytes(archive["game_ids"][index]).hex(),
                        content_sha256=bytes(archive["content_hashes"][index]).hex(),
                        dataset_id=dataset_ids[index],
                        asset_sha256=bytes(archive["asset_hashes"][index]).hex(),
                        member_path=member_paths[index],
                        ordinal=int(archive["ordinals"][index]),
                        rules=rules[index],
                        komi_half_points=int(archive["komi_half_points"][index]),
                        max_moves=int(archive["max_moves"][index]),
                        moves=tuple(int(value) for value in archive["moves"][move_start:move_end]),
                        trainable_start_ply=int(archive["trainable_start_ply"][index]),
                        policy_row_offsets=tuple(int(value) for value in row_offsets),
                        policy_actions=tuple(
                            int(value) for value in archive["policy_actions"][action_start:action_end]
                        ),
                        policy_values=tuple(
                            float(value) for value in archive["policy_values"][action_start:action_end]
                        ),
                        value_black=float(archive["value_black"][index]),
                        value_available=bool(archive["value_mask"][index]),
                        score_margin_black=float(archive["score_margin_black"][index]),
                        score_available=bool(archive["score_mask"][index]),
                        ownership_black=tuple(
                            float(value) for value in archive["ownership_black"][index]
                        ),
                        ownership_available=bool(archive["ownership_mask"][index]),
                    )
                )
        return tuple(records)

    def write_annotations(self, records: list[AnnotationRecord]) -> ShardInfo:
        if not records:
            raise ValueError("cannot write an empty annotation shard")
        policy_offsets, policy_actions = _concatenate_int(
            [record.policy_actions for record in records], np.int16
        )
        policy_values = np.asarray(
            [value for record in records for value in record.policy_values], dtype=np.float32
        )
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(SHARD_SCHEMA_VERSION, dtype=np.int32),
            "record_schema_version": np.asarray(RECORD_SCHEMA_VERSION, dtype=np.int32),
            "kind": np.asarray(2, dtype=np.uint8),
            "game_ids": _hex_matrix([record.game_id for record in records]),
            "content_hashes": _hex_matrix([record.content_sha256 for record in records]),
            "plies": np.asarray([record.ply for record in records], dtype=np.int32),
            "policy_row_offsets": policy_offsets,
            "policy_actions": policy_actions,
            "policy_values": policy_values,
            "value": np.asarray([record.value for record in records], dtype=np.float32),
            "value_mask": np.asarray(
                [record.value_available for record in records], dtype=np.bool_
            ),
            "score_margin": np.asarray(
                [record.score_margin for record in records], dtype=np.float32
            ),
            "score_mask": np.asarray(
                [record.score_available for record in records], dtype=np.bool_
            ),
            "ownership": np.asarray(
                [record.ownership for record in records], dtype=np.float32
            ),
            "ownership_mask": np.asarray(
                [record.ownership_available for record in records], dtype=np.bool_
            ),
        }
        arrays.update(
            _pack_text(
                "teacher_fingerprints",
                [record.teacher_fingerprint for record in records],
            )
        )
        return self._write("annotation", arrays, len(records), len(records))

    def read_annotations(self, info_or_path: ShardInfo | str | Path) -> tuple[AnnotationRecord, ...]:
        path = self._coerce_path(info_or_path)
        with self._open_validated(path, expected_kind=2) as archive:
            teachers = _unpack_text(archive, "teacher_fingerprints")
            records = []
            for index in range(len(archive["game_ids"])):
                start, end = archive["policy_row_offsets"][index : index + 2]
                record = AnnotationRecord(
                    schema_version=int(archive["record_schema_version"]),
                    game_id=bytes(archive["game_ids"][index]).hex(),
                    ply=int(archive["plies"][index]),
                    teacher_fingerprint=teachers[index],
                    policy_actions=tuple(
                        int(value) for value in archive["policy_actions"][start:end]
                    ),
                    policy_values=tuple(
                        float(value) for value in archive["policy_values"][start:end]
                    ),
                    value=float(archive["value"][index]),
                    value_available=bool(archive["value_mask"][index]),
                    score_margin=float(archive["score_margin"][index]),
                    score_available=bool(archive["score_mask"][index]),
                    ownership=tuple(float(value) for value in archive["ownership"][index]),
                    ownership_available=bool(archive["ownership_mask"][index]),
                )
                stored_hash = bytes(archive["content_hashes"][index]).hex()
                if record.content_sha256 != stored_hash:
                    raise ValueError("annotation content_sha256 does not match its contents")
                records.append(record)
        return tuple(records)

    def verify(self, relative_path: str, expected_sha256: str) -> None:
        path = self.resolve(relative_path)
        if _sha256_file(path) != expected_sha256:
            raise ValueError(f"shard SHA-256 mismatch: {relative_path}")
        with self._open_validated(path) as archive:
            kind = int(archive["kind"])
        if kind == 1:
            self.read_trajectories(path)
        elif kind == 2:
            self.read_annotations(path)
        else:
            raise ValueError("unexpected shard kind")

    def read_verified_trajectories(
        self,
        relative_path: str,
        expected_sha256: str,
    ) -> tuple[TrajectoryRecord, ...]:
        path = self.resolve(relative_path)
        if _sha256_file(path) != expected_sha256:
            raise ValueError(f"shard SHA-256 mismatch: {relative_path}")
        return self.read_trajectories(path)

    def read_verified_annotations(
        self,
        relative_path: str,
        expected_sha256: str,
    ) -> tuple[AnnotationRecord, ...]:
        path = self.resolve(relative_path)
        if _sha256_file(path) != expected_sha256:
            raise ValueError(f"shard SHA-256 mismatch: {relative_path}")
        return self.read_annotations(path)

    def _coerce_path(self, value: ShardInfo | str | Path) -> Path:
        if isinstance(value, ShardInfo):
            return self.resolve(value.relative_path)
        path = Path(value)
        return path if path.is_absolute() else self.resolve(path.as_posix())

    def _write(
        self,
        kind: ShardKind,
        arrays: dict[str, np.ndarray],
        record_count: int,
        position_count: int,
    ) -> ShardInfo:
        directory = self.trajectory_dir if kind == "trajectory" else self.annotation_dir
        descriptor, temporary_name = tempfile.mkstemp(prefix=".shard-", suffix=".tmp", dir=directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            with self._open_validated(temporary, expected_kind=1 if kind == "trajectory" else 2):
                pass
            digest = _sha256_file(temporary)
            destination = directory / f"{digest}.npz"
            if destination.exists():
                if _sha256_file(destination) != digest:
                    raise ValueError("existing content-addressed shard is corrupt")
                temporary.unlink()
            else:
                os.replace(temporary, destination)
                _fsync_directory(directory)
            relative = destination.relative_to(self.root).as_posix()
            return ShardInfo(
                kind=kind,
                sha256=digest,
                relative_path=relative,
                size_bytes=destination.stat().st_size,
                record_count=record_count,
                position_count=position_count,
            )
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _open_validated(path: Path, expected_kind: int | None = None):
        archive = np.load(path, allow_pickle=False)
        try:
            actual_schema = int(archive["schema_version"])
            if actual_schema != SHARD_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported shard schema v{actual_schema}; rebuild data shards for v2"
                )
            record_schema = int(archive["record_schema_version"])
            if record_schema != RECORD_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported record schema v{record_schema}; rebuild data shards for v2"
                )
            if expected_kind is not None and int(archive["kind"]) != expected_kind:
                raise ValueError("unexpected shard kind")
            for name in archive.files:
                if archive[name].dtype.hasobject:
                    raise ValueError(f"object arrays are forbidden in shards: {name}")
        except BaseException:
            archive.close()
            raise
        return archive
