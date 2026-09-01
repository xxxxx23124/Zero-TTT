from __future__ import annotations

import io
import json
import signal
import sys
import threading
from pathlib import Path

import pytest

from zero_ttt.console.config import RunContext
from zero_ttt.control.models import JobState
from zero_ttt.control.process import OperationConflict, WorkerController
from zero_ttt.control.runs import RuntimeLayout


def _layout(tmp_path: Path) -> RuntimeLayout:
    return RuntimeLayout(
        source_root=tmp_path / "data",
        staging_root=tmp_path / "data" / "staging",
        manifest_root=tmp_path / "data" / "manifests",
        catalog_path=tmp_path / "data" / "catalog" / "catalog.sqlite",
        store_root=tmp_path / "data" / "processed",
        run_root=tmp_path / "runs",
        profile_root=tmp_path / "profiles",
    )


def _event(name: str, payload: dict[str, object]) -> str:
    return json.dumps({"event": name, "timestamp_ns": 1, "payload": payload})


def test_worker_projects_jsonl_status_for_the_selected_run(tmp_path, monkeypatch) -> None:
    line = _event("status", {"validated": True, "phase": "COLD_START"})
    controller = WorkerController(
        _layout(tmp_path),
        command_factory=lambda _job, _context: (sys.executable, "-c", f"print({line!r})"),
    )
    run_id = "a" * 32
    context = RunContext(
        run_id,
        "run",
        tmp_path / "experiment.toml",
        tmp_path / "runs" / run_id,
        tmp_path / "catalog.sqlite",
        tmp_path / "processed",
        "b" * 64,
        1.0,
    )
    monkeypatch.setattr(controller.runs, "context", lambda _run_id, _hours: context)

    controller.start_run(run_id, "reconcile", 1.0)
    controller.wait(5.0)
    snapshot = controller.snapshot(run_id)

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


def test_worker_rejects_global_concurrency_and_forwards_soft_stop(tmp_path) -> None:
    process = _SignalProcess()
    controller = WorkerController(
        _layout(tmp_path),
        command_factory=lambda _job, _context: ("worker",),
        popen_factory=lambda *_args, **_kwargs: process,
    )
    controller.start_data("full-import")

    with pytest.raises(OperationConflict, match="already running"):
        controller.start_data("trial-import")
    stopped = controller.soft_stop()
    snapshot = controller.wait(5.0)

    assert stopped["state"] == JobState.STOP_REQUESTED.value
    assert process.received == signal.SIGTERM
    assert snapshot["job"]["state"] == JobState.SUCCEEDED.value


def test_short_atomic_data_job_cannot_be_stopped(tmp_path) -> None:
    process = _SignalProcess()
    controller = WorkerController(
        _layout(tmp_path),
        command_factory=lambda _job, _context: ("worker",),
        popen_factory=lambda *_args, **_kwargs: process,
    )
    controller.start_data("snapshot-create")
    with pytest.raises(OperationConflict, match="cannot be safely stopped"):
        controller.soft_stop()
    process.finished.set()
    controller.wait(5.0)
