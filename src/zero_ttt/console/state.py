"""Durable console state, state-machine validation, and single-process locking."""

from __future__ import annotations

import dataclasses
import json
import os
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from zero_ttt.versioning import CONSOLE_STATE_SCHEMA


class TrainingPhase(StrEnum):
    COLD_START = "COLD_START"
    MIXTURE = "MIXTURE"


class Operation(StrEnum):
    READY = "READY"
    COLLECTING = "COLLECTING"
    TRAINING = "TRAINING"
    WARM_STARTING = "WARM_STARTING"
    SOFT_STOPPING = "SOFT_STOPPING"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class MigrationRecord:
    reason: str
    previous_snapshot_id: str
    new_snapshot_id: str
    mixture_manifest_sha256: str
    completed_ns: int


@dataclass(frozen=True, slots=True)
class ConsoleState:
    schema_version: int = CONSOLE_STATE_SCHEMA.current
    phase: TrainingPhase = TrainingPhase.COLD_START
    operation: Operation = Operation.READY
    next_collection_round: int = 0
    last_operation: str = ""
    last_outcome: str = ""
    migrations: tuple[MigrationRecord, ...] = ()


_TRANSITIONS = {
    Operation.READY: {
        Operation.COLLECTING,
        Operation.TRAINING,
        Operation.WARM_STARTING,
        Operation.FAILED,
    },
    Operation.COLLECTING: {Operation.SOFT_STOPPING, Operation.FAILED},
    Operation.TRAINING: {Operation.SOFT_STOPPING, Operation.FAILED},
    Operation.WARM_STARTING: {Operation.SOFT_STOPPING, Operation.FAILED},
    Operation.SOFT_STOPPING: {Operation.READY, Operation.FAILED},
    Operation.FAILED: {Operation.READY},
}


def transition(state: ConsoleState, operation: Operation) -> ConsoleState:
    if operation not in _TRANSITIONS[state.operation]:
        raise ValueError(
            f"invalid console transition: {state.operation} -> {operation}"
        )
    return dataclasses.replace(state, operation=operation)


def _payload(state: ConsoleState) -> dict[str, object]:
    return {
        "schema_version": state.schema_version,
        "phase": state.phase.value,
        "operation": state.operation.value,
        "next_collection_round": state.next_collection_round,
        "last_operation": state.last_operation,
        "last_outcome": state.last_outcome,
        "migrations": [dataclasses.asdict(record) for record in state.migrations],
    }


class StateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ConsoleState:
        if not self.path.exists():
            return ConsoleState()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "phase",
            "operation",
            "next_collection_round",
            "last_operation",
            "last_outcome",
            "migrations",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("console state fields are incomplete or unknown")
        CONSOLE_STATE_SCHEMA.require(payload["schema_version"])
        round_number = payload["next_collection_round"]
        if (
            isinstance(round_number, bool)
            or not isinstance(round_number, int)
            or round_number < 0
        ):
            raise ValueError(
                "console next_collection_round must be a non-negative integer"
            )
        try:
            migrations = tuple(
                MigrationRecord(**item) for item in payload["migrations"]
            )
            return ConsoleState(
                schema_version=CONSOLE_STATE_SCHEMA.current,
                phase=TrainingPhase(payload["phase"]),
                operation=Operation(payload["operation"]),
                next_collection_round=round_number,
                last_operation=str(payload["last_operation"]),
                last_outcome=str(payload["last_outcome"]),
                migrations=migrations,
            )
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("invalid console state") from error

    def save(self, state: ConsoleState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    _payload(state), handle, sort_keys=True, separators=(",", ":")
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()


class ConsoleLock:
    """Advisory one-byte lock; the supported production environment is Linux Docker."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None

    def __enter__(self) -> "ConsoleLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = None
        try:
            handle = self.path.open("a+b")
            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if handle is not None:
                handle.close()
            raise RuntimeError(
                "another training console already owns this run"
            ) from error
        self._handle = handle
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle is None:
            return
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def migration_record(
    reason: str,
    previous_snapshot_id: str,
    new_snapshot_id: str,
    mixture_manifest_sha256: str,
) -> MigrationRecord:
    return MigrationRecord(
        reason=reason,
        previous_snapshot_id=previous_snapshot_id,
        new_snapshot_id=new_snapshot_id,
        mixture_manifest_sha256=mixture_manifest_sha256,
        completed_ns=time.time_ns(),
    )
