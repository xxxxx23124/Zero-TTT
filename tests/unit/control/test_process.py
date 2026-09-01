from __future__ import annotations

import io
import json
import signal
import sys
import threading

import pytest

from zero_ttt.control.models import JobState
from zero_ttt.control.process import OperationConflict, WorkerController


def _event(name: str, payload: dict[str, object]) -> str:
    return json.dumps({"event": name, "timestamp_ns": 1, "payload": payload})


def test_worker_projects_jsonl_status_from_a_subprocess(tmp_path) -> None:
    line = _event("status", {"validated": True, "phase": "COLD_START"})
    controller = WorkerController(
        tmp_path / "console.toml",
        command_factory=lambda _operation, _path: (sys.executable, "-c", f"print({line!r})"),
    )

    controller.start("reconcile")
    snapshot = controller.wait(5.0)

    assert snapshot["job"]["state"] == JobState.SUCCEEDED.value
    assert snapshot["console"]["validated"] is True
    assert snapshot["console"]["phase"] == "COLD_START"


class _SignalProcess:
    def __init__(self) -> None:
        self.pid = 42
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO("")
        self.received: int | None = None
        self.finished = threading.Event()

    def wait(self) -> int:
        self.finished.wait(5.0)
        return 0

    def send_signal(self, kind: int) -> None:
        self.received = kind
        self.finished.set()


def test_worker_rejects_concurrency_and_forwards_soft_stop(tmp_path) -> None:
    process = _SignalProcess()
    controller = WorkerController(
        tmp_path / "console.toml",
        command_factory=lambda _operation, _path: ("worker",),
        popen_factory=lambda *_args, **_kwargs: process,
    )
    controller.start("train")

    with pytest.raises(OperationConflict, match="already running"):
        controller.start("collect")
    stopped = controller.soft_stop()
    snapshot = controller.wait(5.0)

    assert stopped["state"] == JobState.STOP_REQUESTED.value
    assert process.received == signal.SIGTERM
    assert snapshot["job"]["state"] == JobState.SUCCEEDED.value
