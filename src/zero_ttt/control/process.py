"""Subprocess ownership and JSONL event projection for the training agent."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from zero_ttt.control.models import ACTIVE_JOB_STATES, JobRecord, JobState

OPERATIONS = frozenset({"reconcile", "train", "collect", "warm-start"})


class OperationConflict(RuntimeError):
    pass


CommandFactory = Callable[[str, Path], Sequence[str]]
PopenFactory = Callable[..., subprocess.Popen[str]]


def default_command(operation: str, config_path: Path) -> Sequence[str]:
    return (
        sys.executable,
        "-m",
        "zero_ttt.cli",
        "console",
        "--config",
        str(config_path),
        operation,
        "--events",
        "jsonl",
    )


class WorkerController:
    def __init__(
        self,
        config_path: str | Path,
        *,
        command_factory: CommandFactory = default_command,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        self.config_path = Path(config_path)
        self.command_factory = command_factory
        self.popen_factory = popen_factory
        self._lock = threading.RLock()
        self._job: JobRecord | None = None
        self._process: subprocess.Popen[str] | None = None
        self._console: dict[str, Any] = {"validated": False}
        self._latest_metrics: dict[str, Any] = {}
        self._latest_collection: dict[str, Any] = {}
        self._logs: deque[str] = deque(maxlen=200)
        self._waiter: threading.Thread | None = None

    def _active(self) -> bool:
        return self._job is not None and self._job.state in ACTIVE_JOB_STATES

    def start(self, operation: str) -> dict[str, object]:
        if operation not in OPERATIONS:
            raise ValueError(f"unsupported operation: {operation}")
        with self._lock:
            if self._active():
                raise OperationConflict("another console operation is already running")
            job = JobRecord.create(operation)
            self._job = job
            command = tuple(self.command_factory(operation, self.config_path))
            try:
                process = self.popen_factory(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except Exception as error:
                job.state = JobState.FAILED
                job.finished_ns = time.time_ns()
                job.error = str(error)
                raise
            self._process = process
            job.pid = process.pid
            job.state = JobState.RUNNING
            self._start_threads(process, job.operation_id)
            return job.payload()

    def _start_threads(self, process: subprocess.Popen[str], operation_id: str) -> None:
        stdout = threading.Thread(
            target=self._read_stdout,
            args=(process.stdout, operation_id),
            daemon=True,
        )
        stderr = threading.Thread(
            target=self._read_stderr,
            args=(process.stderr, operation_id),
            daemon=True,
        )
        waiter = threading.Thread(
            target=self._wait,
            args=(process, operation_id, stdout, stderr),
            daemon=True,
        )
        stdout.start()
        stderr.start()
        waiter.start()
        self._waiter = waiter

    def _read_stdout(self, stream: TextIO | None, operation_id: str) -> None:
        if stream is None:
            return
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
                self._apply_event(operation_id, event)
            except (TypeError, ValueError, json.JSONDecodeError):
                self._append_log(f"stdout: {stripped}")

    def _read_stderr(self, stream: TextIO | None, operation_id: str) -> None:
        if stream is None:
            return
        for line in stream:
            if stripped := line.strip():
                with self._lock:
                    self._logs.append(f"stderr: {stripped}")
                    if self._job is not None and self._job.operation_id == operation_id:
                        self._job.error = stripped

    def _append_log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def _apply_event(self, operation_id: str, event: dict[str, Any]) -> None:
        name = str(event.get("event", ""))
        payload = event.get("payload")
        if not isinstance(payload, dict):
            self._append_log(f"invalid event payload: {event!r}")
            return
        with self._lock:
            if self._job is None or self._job.operation_id != operation_id:
                return
            self._logs.append(f"{name}: {json.dumps(payload, ensure_ascii=False)}")
            self._project_event(name, payload)

    def _project_event(self, name: str, payload: dict[str, Any]) -> None:
        if name == "status":
            self._console = dict(payload)
        elif name == "operation_started":
            self._console.update(operation=payload.get("operation"), phase=payload.get("phase"))
        elif name == "operation_finished":
            self._console.update(
                operation="READY",
                phase=payload.get("phase"),
                last_outcome=payload.get("outcome"),
            )
        elif name == "training_step":
            self._latest_metrics = dict(payload)
            self._console.update(
                run_id=payload.get("run_id"),
                optimizer_step=payload.get("step"),
                samples_seen=payload.get("samples_seen"),
            )
        elif name == "training_finished":
            self._console.update(
                run_id=payload.get("run_id"),
                optimizer_step=payload.get("optimizer_step"),
                samples_seen=payload.get("samples_seen"),
                phase=payload.get("phase"),
                checkpoint_path=payload.get("checkpoint_path"),
                publication_path=payload.get("publication_path"),
                selfplay_snapshot_id=payload.get("selfplay_snapshot_id"),
                mixture_manifest_sha256=payload.get("mixture_manifest_sha256"),
            )
            if payload.get("mixture_manifest_sha256"):
                self._console.update(pending_games=0, pending_positions=0)
        elif name == "collection_round":
            self._latest_collection = dict(payload)
            self._update_selfplay(payload)
        elif name == "operation_failed" and self._job is not None:
            self._job.error = f"{payload.get('error_type')}: {payload.get('error')}"

    def _update_selfplay(self, payload: dict[str, Any]) -> None:
        current = self._console.get("selfplay")
        statistics = dict(current) if isinstance(current, dict) else {}
        statistics["sealed_tasks"] = int(statistics.get("sealed_tasks", 0)) + 1
        statistics["games"] = int(statistics.get("games", 0)) + int(
            payload.get("collected_games", 0)
        )
        statistics["positions"] = int(statistics.get("positions", 0)) + int(
            payload.get("new_positions", 0)
        )
        self._console["selfplay"] = statistics
        self._console["pending_games"] = int(self._console.get("pending_games", 0)) + int(
            payload.get("collected_games", 0)
        )
        self._console["pending_positions"] = int(
            self._console.get("pending_positions", 0)
        ) + int(payload.get("new_positions", 0))

    def _wait(
        self,
        process: subprocess.Popen[str],
        operation_id: str,
        stdout: threading.Thread,
        stderr: threading.Thread,
    ) -> None:
        return_code = process.wait()
        stdout.join()
        stderr.join()
        with self._lock:
            if self._job is None or self._job.operation_id != operation_id:
                return
            self._job.return_code = return_code
            self._job.finished_ns = time.time_ns()
            if return_code == 0:
                self._job.state = JobState.SUCCEEDED
            elif self._job.state is JobState.STOP_REQUESTED:
                self._job.state = JobState.INTERRUPTED
            else:
                self._job.state = JobState.FAILED
            self._process = None

    def soft_stop(self) -> dict[str, object]:
        with self._lock:
            if not self._active() or self._process is None or self._job is None:
                raise OperationConflict("no console operation is running")
            if self._job.operation == "reconcile":
                raise OperationConflict("artifact reconciliation cannot be soft-stopped")
            if self._job.state is not JobState.STOP_REQUESTED:
                self._process.send_signal(signal.SIGTERM)
                self._job.state = JobState.STOP_REQUESTED
            return self._job.payload()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            job = None if self._job is None else self._job.payload()
            return {
                "config_path": str(self.config_path),
                "job": job,
                "console": dict(self._console),
                "latest_metrics": dict(self._latest_metrics),
                "latest_collection": dict(self._latest_collection),
                "logs": tuple(self._logs),
            }

    def wait(self, timeout: float | None = None) -> dict[str, object]:
        with self._lock:
            waiter = self._waiter
        if waiter is not None:
            waiter.join(timeout)
            if waiter.is_alive():
                raise TimeoutError("console operation did not finish before the timeout")
        return self.snapshot()

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            job = self._job
            if process is None or job is None or job.state not in ACTIVE_JOB_STATES:
                return
            process.send_signal(signal.SIGTERM)
            if job.operation != "reconcile":
                job.state = JobState.STOP_REQUESTED
        process.wait()
