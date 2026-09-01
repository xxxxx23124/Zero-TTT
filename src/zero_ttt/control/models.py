"""Typed in-memory state for one serialized console worker."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum


class JobState(StrEnum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOP_REQUESTED = "STOP_REQUESTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


ACTIVE_JOB_STATES = {JobState.STARTING, JobState.RUNNING, JobState.STOP_REQUESTED}


@dataclass(slots=True)
class JobRecord:
    operation_id: str
    operation: str
    state: JobState
    started_ns: int
    finished_ns: int = 0
    pid: int | None = None
    return_code: int | None = None
    error: str = ""

    @classmethod
    def create(cls, operation: str) -> JobRecord:
        return cls(uuid.uuid4().hex, operation, JobState.STARTING, time.time_ns())

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload
