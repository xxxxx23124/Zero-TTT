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
    kind: str
    operation: str
    run_id: str
    state: JobState
    started_ns: int
    stoppable: bool
    finished_ns: int = 0
    pid: int | None = None
    return_code: int | None = None
    error: str = ""
    progress: dict[str, object] | None = None

    @classmethod
    def create(
        cls, kind: str, operation: str, run_id: str = "", *, stoppable: bool = True
    ) -> JobRecord:
        return cls(
            uuid.uuid4().hex,
            kind,
            operation,
            run_id,
            JobState.STARTING,
            time.time_ns(),
            stoppable,
        )

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload
