"""Single subprocess owner for data preparation and training-run jobs."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any, TextIO

from zero_ttt.console.config import RunContext
from zero_ttt.control.data import DATA_OPERATIONS, DataService
from zero_ttt.control.models import ACTIVE_JOB_STATES, JobRecord, JobState
from zero_ttt.control.runs import RunRepository, RuntimeLayout
from zero_ttt.data.catalog import Catalog
from zero_ttt.data.shards import ShardStore

RUN_OPERATIONS = frozenset({"reconcile", "train", "collect", "warm-start"})


class OperationConflict(RuntimeError):
    pass


CommandFactory = Callable[[JobRecord, RunContext | None], Sequence[str]]
PopenFactory = Callable[..., subprocess.Popen[str]]


def default_command(job: JobRecord, context: RunContext | None) -> Sequence[str]:
    if job.kind == "data":
        return (sys.executable, "-m", "zero_ttt.control.data_worker", job.operation)
    if context is None:
        raise ValueError("run jobs require a validated run context")
    return (
        sys.executable,
        "-m",
        "zero_ttt.cli",
        "console",
        "--run-id",
        context.run_id,
        "--name",
        context.name,
        "--config",
        str(context.experiment_config),
        "--run-dir",
        str(context.run_dir),
        "--catalog",
        str(context.catalog_path),
        "--store-root",
        str(context.store_root),
        "--cold-snapshot",
        context.cold_start_snapshot_id,
        "--max-runtime-hours",
        str(context.max_runtime_hours),
        job.operation,
        "--events",
        "jsonl",
    )


class WorkerController:
    def __init__(
        self,
        layout: RuntimeLayout,
        *,
        command_factory: CommandFactory = default_command,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        self.layout = layout
        self.runs = RunRepository(layout)
        self.data = DataService(layout)
        self.command_factory = command_factory
        self.popen_factory = popen_factory
        self._lock = threading.RLock()
        self._job: JobRecord | None = None
        self._process: subprocess.Popen[str] | None = None
        self._console_by_run: dict[str, dict[str, Any]] = {}
        self._latest_metrics: dict[str, Any] = {}
        self._latest_collection: dict[str, Any] = {}
        self._latest_data: dict[str, Any] = {}
        self._logs: deque[str] = deque(maxlen=300)
        self._waiter: threading.Thread | None = None

    def _active(self) -> bool:
        return self._job is not None and self._job.state in ACTIVE_JOB_STATES

    def list_profiles(self) -> tuple[dict[str, object], ...]:
        return tuple(asdict(item) for item in self.runs.list_profiles())

    def list_runs(self) -> tuple[dict[str, object], ...]:
        return tuple(item.payload() for item in self.runs.list_runs())

    def create_run(self, name: str, profile_id: str, snapshot_id: str) -> dict[str, object]:
        with self._lock:
            if self._active():
                raise OperationConflict("a data or training job is running")
            return self.runs.create(name, profile_id, snapshot_id).payload()

    def data_status(self) -> dict[str, object]:
        return asdict(self.data.status())

    def list_snapshots(self) -> tuple[dict[str, object], ...]:
        if not self.layout.catalog_path.is_file():
            return ()
        with Catalog(self.layout.catalog_path, ShardStore(self.layout.store_root)) as catalog:
            return tuple(asdict(item) for item in catalog.list_snapshots())

    def start_run(
        self, run_id: str, operation: str, max_runtime_hours: float
    ) -> dict[str, object]:
        if operation not in RUN_OPERATIONS:
            raise ValueError(f"unsupported run operation: {operation}")
        context = self.runs.context(run_id, max_runtime_hours)
        return self._start(
            JobRecord.create(
                "run", operation, run_id, stoppable=operation != "reconcile"
            ),
            context,
        )

    def start_data(self, operation: str) -> dict[str, object]:
        if operation not in DATA_OPERATIONS:
            raise ValueError(f"unsupported data operation: {operation}")
        return self._start(
            JobRecord.create(
                "data",
                operation,
                stoppable=operation in {"scan", "trial-import", "full-import"},
            ),
            None,
        )

    def _start(self, job: JobRecord, context: RunContext | None) -> dict[str, object]:
        with self._lock:
            if self._active():
                raise OperationConflict("another data or training job is already running")
            self._job = job
            command = tuple(self.command_factory(job, context))
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
            target=self._read_stdout, args=(process.stdout, operation_id), daemon=True
        )
        stderr = threading.Thread(
            target=self._read_stderr, args=(process.stderr, operation_id), daemon=True
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
            job = self._job
            if job is None or job.operation_id != operation_id:
                return
            self._logs.append(f"{name}: {json.dumps(payload, ensure_ascii=False)}")
            if name == "data_progress":
                job.progress = dict(payload)
            elif name.startswith("data_") or name == "snapshot_created":
                self._latest_data = {"event": name, **payload}
            if job.kind == "run":
                console = self._console_by_run.setdefault(job.run_id, {"validated": False})
                self._project_run_event(console, name, payload)

    def _project_run_event(
        self, console: dict[str, Any], name: str, payload: dict[str, Any]
    ) -> None:
        if name == "status":
            console.clear()
            console.update(payload)
        elif name == "operation_started":
            console.update(operation=payload.get("operation"), phase=payload.get("phase"))
        elif name == "operation_finished":
            console.update(operation="READY", phase=payload.get("phase"), last_outcome=payload.get("outcome"))
        elif name == "training_step":
            self._latest_metrics = dict(payload)
            console.update(
                run_id=payload.get("run_id"),
                optimizer_step=payload.get("step"),
                samples_seen=payload.get("samples_seen"),
            )
        elif name == "training_finished":
            console.update(
                run_id=payload.get("run_id"),
                optimizer_step=payload.get("optimizer_step"),
                samples_seen=payload.get("samples_seen"),
                checkpoint_path=payload.get("checkpoint_path"),
                publication_path=payload.get("publication_path"),
                selfplay_snapshot_id=payload.get("selfplay_snapshot_id"),
            )
        elif name == "collection_round":
            self._latest_collection = dict(payload)

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
                raise OperationConflict("no data or training job is running")
            if not self._job.stoppable:
                raise OperationConflict("the current job cannot be safely stopped")
            if self._job.state is not JobState.STOP_REQUESTED:
                self._process.send_signal(signal.SIGTERM)
                self._job.state = JobState.STOP_REQUESTED
            return self._job.payload()

    def snapshot(self, run_id: str | None = None) -> dict[str, object]:
        with self._lock:
            job = None if self._job is None else self._job.payload()
            selected = run_id or ("" if self._job is None else self._job.run_id)
            return {
                "job": job,
                "selected_run_id": selected,
                "console": dict(self._console_by_run.get(selected, {"validated": False})),
                "latest_metrics": dict(self._latest_metrics),
                "latest_collection": dict(self._latest_collection),
                "latest_data": dict(self._latest_data),
                "logs": tuple(self._logs),
            }

    def wait(self, timeout: float | None = None) -> dict[str, object]:
        with self._lock:
            waiter = self._waiter
        if waiter is not None:
            waiter.join(timeout)
            if waiter.is_alive():
                raise TimeoutError("job did not finish before the timeout")
        return self.snapshot()

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            job = self._job
            if process is None or job is None or job.state not in ACTIVE_JOB_STATES:
                return
            process.send_signal(signal.SIGTERM)
            if job.stoppable:
                job.state = JobState.STOP_REQUESTED
        process.wait()
