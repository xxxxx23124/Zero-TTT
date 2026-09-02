"""SQLite-backed jobs, leases, workflows, events, and read models."""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zero_ttt_contracts import (
    ArtifactRef,
    CompleteJobRequest,
    DomainEvent,
    FailJobRequest,
    HeartbeatRequest,
    HeartbeatResponse,
    JobEnvelope,
    JobState,
    LeaseJobRequest,
    ResourceClass,
    RunSpec,
    WorkerCapability,
    WorkerRegistration,
    WorkflowState,
    WorkflowTemplate,
)
from zero_ttt_contracts.hashing import canonical_json_bytes, payload_sha256

_SCHEMA_VERSION = 1


class LeaseConflict(RuntimeError):
    pass


class ControlStore:
    def __init__(
        self,
        path: str | Path,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock_ns = clock_ns
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in {0, _SCHEMA_VERSION}:
            raise RuntimeError(f"unsupported control database schema v{version}")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                capability TEXT NOT NULL,
                version TEXT NOT NULL,
                registered_ns INTEGER NOT NULL,
                last_seen_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                spec_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL DEFAULT '',
                template TEXT NOT NULL,
                state TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
                run_id TEXT NOT NULL DEFAULT '',
                ordinal INTEGER NOT NULL,
                kind TEXT NOT NULL,
                capability TEXT NOT NULL,
                resource_class TEXT NOT NULL,
                state TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                attempt INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                lease_owner TEXT NOT NULL DEFAULT '',
                lease_token TEXT NOT NULL DEFAULT '',
                lease_expires_ns INTEGER NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                result_json TEXT NOT NULL DEFAULT '{}',
                error TEXT NOT NULL DEFAULT '',
                created_ns INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL,
                UNIQUE(workflow_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS job_dependencies (
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                depends_on_job_id TEXT NOT NULL REFERENCES jobs(job_id),
                PRIMARY KEY(job_id, depends_on_job_id)
            );
            CREATE TABLE IF NOT EXISTS resource_leases (
                resource_class TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                lease_token TEXT NOT NULL,
                expires_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                kind TEXT NOT NULL,
                ref_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                kind TEXT NOT NULL,
                level TEXT NOT NULL,
                occurred_ns INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS jobs_claim_idx
              ON jobs(capability,state,ordinal,created_ns);
            CREATE INDEX IF NOT EXISTS events_job_idx ON events(job_id,sequence);
            """
        )
        self.connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")

    def _transaction(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _json(value: Any) -> str:
        return canonical_json_bytes(value).decode("utf-8")

    def register_worker(self, registration: WorkerRegistration) -> dict[str, Any]:
        now = self.clock_ns()
        with self._lock:
            self.connection.execute(
                """
                INSERT INTO workers(worker_id,capability,version,registered_ns,last_seen_ns)
                VALUES(?,?,?,?,?)
                ON CONFLICT(worker_id) DO UPDATE SET
                  capability=excluded.capability,
                  version=excluded.version,
                  last_seen_ns=excluded.last_seen_ns
                """,
                (
                    registration.worker_id,
                    registration.capability.value,
                    registration.version,
                    now,
                    now,
                ),
            )
        return {"worker_id": registration.worker_id, "registered_ns": now}

    def create_run(self, spec: RunSpec) -> RunSpec:
        with self._lock:
            self.connection.execute(
                "INSERT INTO runs(run_id,name,spec_json,created_ns) VALUES(?,?,?,?)",
                (spec.run_id, spec.name, spec.model_dump_json(), spec.created_ns),
            )
        return spec

    def get_run(self, run_id: str) -> RunSpec:
        row = self.connection.execute(
            "SELECT spec_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id}")
        return RunSpec.model_validate_json(row["spec_json"])

    def list_runs(self) -> tuple[RunSpec, ...]:
        rows = self.connection.execute(
            "SELECT spec_json FROM runs ORDER BY created_ns DESC"
        ).fetchall()
        return tuple(RunSpec.model_validate_json(row["spec_json"]) for row in rows)

    def submit_workflow(
        self,
        template: WorkflowTemplate,
        payload: dict[str, Any],
        *,
        run_id: str = "",
        idempotency_key: str | None = None,
    ) -> str:
        if template is not WorkflowTemplate.DATA_BOOTSTRAP:
            self.get_run(run_id)
        identity = idempotency_key or payload_sha256(
            {"template": template.value, "run_id": run_id, "payload": payload}
        )
        existing = self.connection.execute(
            "SELECT workflow_id FROM workflows WHERE payload_json=? AND template=? AND run_id=?",
            (self._json({"identity": identity, "payload": payload}), template.value, run_id),
        ).fetchone()
        if existing is not None:
            return str(existing["workflow_id"])
        workflow_id = uuid.uuid4().hex
        jobs = self._workflow_jobs(template, payload, run_id, identity)
        now = self.clock_ns()
        with self._lock:
            self._transaction()
            try:
                self.connection.execute(
                    "INSERT INTO workflows VALUES(?,?,?,?,?,?,?)",
                    (
                        workflow_id,
                        run_id,
                        template.value,
                        WorkflowState.QUEUED.value,
                        self._json({"identity": identity, "payload": payload}),
                        now,
                        now,
                    ),
                )
                previous_job_id = ""
                for ordinal, job in enumerate(jobs):
                    job_id = uuid.uuid4().hex
                    self.connection.execute(
                        """
                        INSERT INTO jobs(
                          job_id,workflow_id,run_id,ordinal,kind,capability,resource_class,
                          state,payload_json,idempotency_key,created_ns,updated_ns
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            job_id,
                            workflow_id,
                            run_id,
                            ordinal,
                            job["kind"],
                            job["capability"].value,
                            job["resource_class"].value,
                            JobState.QUEUED.value,
                            self._json(job["payload"]),
                            f"{identity}:{ordinal}:{job['kind']}",
                            now,
                            now,
                        ),
                    )
                    if previous_job_id:
                        self.connection.execute(
                            "INSERT INTO job_dependencies VALUES(?,?)",
                            (job_id, previous_job_id),
                        )
                    previous_job_id = job_id
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise
        return workflow_id

    @staticmethod
    def _workflow_jobs(
        template: WorkflowTemplate,
        payload: dict[str, Any],
        run_id: str,
        identity: str,
    ) -> tuple[dict[str, Any], ...]:
        common = {"workflow_input": payload, "workflow_identity": identity, "run_id": run_id}
        if template is WorkflowTemplate.DATA_BOOTSTRAP:
            kinds = (
                "data.scan",
                "data.trial-import",
                "data.verify-trial",
                "data.full-import",
                "data.verify",
                "data.snapshot-train",
                "data.snapshot-validation",
            )
            return tuple(
                {
                    "kind": kind,
                    "capability": WorkerCapability.DATA,
                    "resource_class": ResourceClass.DATA_WRITER,
                    "payload": common,
                }
                for kind in kinds
            )
        if template is WorkflowTemplate.COLD_START:
            return (
                {
                    "kind": "trainer.cold-start",
                    "capability": WorkerCapability.TRAINER,
                    "resource_class": ResourceClass.GPU_EXCLUSIVE,
                    "payload": common,
                },
            )
        return (
            {
                "kind": "selfplay.collect",
                "capability": WorkerCapability.SELFPLAY,
                "resource_class": ResourceClass.GPU_EXCLUSIVE,
                "payload": common,
            },
            {
                "kind": "data.admit-selfplay",
                "capability": WorkerCapability.DATA,
                "resource_class": ResourceClass.DATA_WRITER,
                "payload": common,
            },
            {
                "kind": "data.snapshot-selfplay",
                "capability": WorkerCapability.DATA,
                "resource_class": ResourceClass.DATA_WRITER,
                "payload": common,
            },
            {
                "kind": "trainer.mixture",
                "capability": WorkerCapability.TRAINER,
                "resource_class": ResourceClass.GPU_EXCLUSIVE,
                "payload": common,
            },
        )

    def _requeue_expired(self, now: int) -> None:
        expired = self.connection.execute(
            """
            SELECT job_id,workflow_id,attempt,max_attempts,cancel_requested FROM jobs
            WHERE state IN (?,?,?) AND lease_expires_ns>0 AND lease_expires_ns<=?
            """,
            (
                JobState.LEASED.value,
                JobState.RUNNING.value,
                JobState.CANCEL_REQUESTED.value,
                now,
            ),
        ).fetchall()
        for row in expired:
            job_id = str(row["job_id"])
            self.connection.execute("DELETE FROM resource_leases WHERE job_id=?", (job_id,))
            cancelled = bool(row["cancel_requested"])
            exhausted = int(row["attempt"]) >= int(row["max_attempts"])
            state = (
                JobState.CANCELLED
                if cancelled
                else (JobState.FAILED if exhausted else JobState.QUEUED)
            )
            error = "lease expired after final attempt" if exhausted and not cancelled else ""
            self.connection.execute(
                """
                UPDATE jobs SET state=?,lease_owner='',lease_token='',lease_expires_ns=0,
                  error=?,updated_ns=? WHERE job_id=?
                """,
                (state.value, error, now, job_id),
            )
            self._refresh_workflow(str(row["workflow_id"]), now)

    def lease_job(self, request: LeaseJobRequest) -> JobEnvelope | None:
        now = self.clock_ns()
        expires = now + request.lease_seconds * 1_000_000_000
        with self._lock:
            self._transaction()
            try:
                self._requeue_expired(now)
                rows = self.connection.execute(
                    """
                    SELECT j.* FROM jobs j
                    WHERE j.capability=? AND j.state=?
                      AND NOT EXISTS (
                        SELECT 1 FROM job_dependencies d
                        JOIN jobs parent ON parent.job_id=d.depends_on_job_id
                        WHERE d.job_id=j.job_id AND parent.state!=?
                      )
                    ORDER BY j.created_ns,j.ordinal
                    """,
                    (
                        request.capability.value,
                        JobState.QUEUED.value,
                        JobState.SUCCEEDED.value,
                    ),
                ).fetchall()
                selected = None
                for row in rows:
                    resource = ResourceClass(row["resource_class"])
                    if resource is not ResourceClass.NONE:
                        occupied = self.connection.execute(
                            "SELECT 1 FROM resource_leases WHERE resource_class=? AND expires_ns>?",
                            (resource.value, now),
                        ).fetchone()
                        if occupied is not None:
                            continue
                    selected = row
                    break
                if selected is None:
                    self.connection.commit()
                    return None
                token = secrets.token_urlsafe(32)
                attempt = int(selected["attempt"]) + 1
                self.connection.execute(
                    """
                    UPDATE jobs SET state=?,attempt=?,lease_owner=?,lease_token=?,
                      lease_expires_ns=?,updated_ns=? WHERE job_id=?
                    """,
                    (
                        JobState.LEASED.value,
                        attempt,
                        request.worker_id,
                        token,
                        expires,
                        now,
                        selected["job_id"],
                    ),
                )
                resource = ResourceClass(selected["resource_class"])
                if resource is not ResourceClass.NONE:
                    self.connection.execute(
                        "INSERT OR REPLACE INTO resource_leases VALUES(?,?,?,?)",
                        (resource.value, selected["job_id"], token, expires),
                    )
                self.connection.execute(
                    "UPDATE workers SET last_seen_ns=? WHERE worker_id=?",
                    (now, request.worker_id),
                )
                inputs = self._job_inputs(str(selected["job_id"]), str(selected["run_id"]))
                self.connection.execute(
                    "UPDATE workflows SET state=?,updated_ns=? WHERE workflow_id=?",
                    (WorkflowState.RUNNING.value, now, selected["workflow_id"]),
                )
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise
        payload = json.loads(selected["payload_json"])
        if selected["run_id"]:
            payload["run_spec"] = self.get_run(str(selected["run_id"])).model_dump(mode="json")
        return JobEnvelope(
            job_id=str(selected["job_id"]),
            workflow_id=str(selected["workflow_id"]),
            run_id=str(selected["run_id"]),
            kind=str(selected["kind"]),
            capability=WorkerCapability(selected["capability"]),
            resource_class=ResourceClass(selected["resource_class"]),
            attempt=attempt,
            lease_token=token,
            lease_expires_ns=expires,
            idempotency_key=str(selected["idempotency_key"]),
            payload=payload,
            inputs=inputs,
        )

    def _dependency_artifacts(self, job_id: str) -> tuple[ArtifactRef, ...]:
        rows = self.connection.execute(
            """
            SELECT a.ref_json FROM artifacts a
            JOIN job_dependencies d ON d.depends_on_job_id=a.job_id
            WHERE d.job_id=? ORDER BY a.created_ns,a.artifact_id
            """,
            (job_id,),
        ).fetchall()
        return tuple(ArtifactRef.model_validate_json(row["ref_json"]) for row in rows)

    def _job_inputs(self, job_id: str, run_id: str) -> tuple[ArtifactRef, ...]:
        references = list(self._dependency_artifacts(job_id))
        if run_id:
            run = self.get_run(run_id)
            references.append(run.cold_snapshot)
            rows = self.connection.execute(
                """
                SELECT a.ref_json FROM artifacts a
                JOIN jobs j ON j.job_id=a.job_id
                WHERE j.run_id=? AND a.kind IN ('checkpoint','publication')
                ORDER BY a.created_ns DESC
                """,
                (run_id,),
            ).fetchall()
            seen_kinds: set[str] = set()
            for row in rows:
                reference = ArtifactRef.model_validate_json(row["ref_json"])
                if reference.kind.value in seen_kinds:
                    continue
                references.append(reference)
                seen_kinds.add(reference.kind.value)
        unique: dict[str, ArtifactRef] = {}
        for reference in references:
            unique[reference.artifact_id] = reference
        return tuple(unique.values())

    def _require_lease(self, job_id: str, worker_id: str, token: str) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown job {job_id}")
        if row["lease_owner"] != worker_id or not secrets.compare_digest(row["lease_token"], token):
            raise LeaseConflict("job lease owner or token does not match")
        if int(row["lease_expires_ns"]) <= self.clock_ns():
            raise LeaseConflict("job lease has expired")
        return row

    def heartbeat(self, job_id: str, request: HeartbeatRequest) -> HeartbeatResponse:
        now = self.clock_ns()
        expires = now + request.lease_seconds * 1_000_000_000
        with self._lock:
            self._transaction()
            try:
                row = self._require_lease(job_id, request.worker_id, request.lease_token)
                cancelled = bool(row["cancel_requested"])
                state = JobState.CANCEL_REQUESTED if cancelled else JobState.RUNNING
                self.connection.execute(
                    "UPDATE jobs SET state=?,lease_expires_ns=?,updated_ns=? WHERE job_id=?",
                    (state.value, expires, now, job_id),
                )
                self.connection.execute(
                    "UPDATE resource_leases SET expires_ns=? WHERE job_id=? AND lease_token=?",
                    (expires, job_id, request.lease_token),
                )
                self.connection.execute(
                    "UPDATE workers SET last_seen_ns=? WHERE worker_id=?",
                    (now, request.worker_id),
                )
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise
        return HeartbeatResponse(lease_expires_ns=expires, cancel_requested=cancelled)

    def append_event(self, job_id: str, worker_id: str, token: str, event: DomainEvent) -> int:
        if event.job_id != job_id:
            raise ValueError("event job_id does not match route")
        with self._lock:
            self._require_lease(job_id, worker_id, token)
            existing = self.connection.execute(
                "SELECT sequence,job_id,kind,level,occurred_ns,payload_json FROM events "
                "WHERE event_id=?",
                (event.event_id,),
            ).fetchone()
            if existing is not None:
                identity = (
                    existing["job_id"],
                    existing["kind"],
                    existing["level"],
                    int(existing["occurred_ns"]),
                    existing["payload_json"],
                )
                expected = (
                    job_id,
                    event.kind,
                    event.level.value,
                    event.occurred_ns,
                    self._json(event.payload),
                )
                if identity != expected:
                    raise ValueError("event ID was reused for different content")
                return int(existing["sequence"])
            cursor = self.connection.execute(
                """
                INSERT INTO events(event_id,job_id,kind,level,occurred_ns,payload_json)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    event.event_id,
                    job_id,
                    event.kind,
                    event.level.value,
                    event.occurred_ns,
                    self._json(event.payload),
                ),
            )
        sequence = cursor.lastrowid
        if sequence is None:
            raise RuntimeError("event insert did not return a sequence")
        return sequence

    def complete(self, job_id: str, request: CompleteJobRequest) -> None:
        now = self.clock_ns()
        with self._lock:
            self._transaction()
            try:
                existing_job = self.connection.execute(
                    "SELECT state,result_json FROM jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                if existing_job is None:
                    raise KeyError(f"unknown job {job_id}")
                if JobState(existing_job["state"]) is JobState.SUCCEEDED:
                    stored_artifacts = {
                        row["artifact_id"]: ArtifactRef.model_validate_json(row["ref_json"])
                        for row in self.connection.execute(
                            "SELECT artifact_id,ref_json FROM artifacts WHERE job_id=?", (job_id,)
                        )
                    }
                    expected_artifacts = {item.artifact_id: item for item in request.artifacts}
                    if (
                        json.loads(existing_job["result_json"]) != request.result
                        or stored_artifacts != expected_artifacts
                    ):
                        raise ValueError("duplicate completion does not match stored result")
                    self.connection.commit()
                    return
                row = self._require_lease(job_id, request.worker_id, request.lease_token)
                for artifact in request.artifacts:
                    existing = self.connection.execute(
                        "SELECT ref_json FROM artifacts WHERE artifact_id=?",
                        (artifact.artifact_id,),
                    ).fetchone()
                    if existing is not None:
                        if ArtifactRef.model_validate_json(existing["ref_json"]) != artifact:
                            raise ValueError("artifact ID was reused for different content")
                        continue
                    self.connection.execute(
                        "INSERT INTO artifacts VALUES(?,?,?,?,?)",
                        (
                            artifact.artifact_id,
                            job_id,
                            artifact.kind.value,
                            artifact.model_dump_json(),
                            now,
                        ),
                    )
                self.connection.execute(
                    """
                    UPDATE jobs SET state=?,result_json=?,lease_owner='',lease_token='',
                      lease_expires_ns=0,updated_ns=? WHERE job_id=?
                    """,
                    (JobState.SUCCEEDED.value, self._json(request.result), now, job_id),
                )
                self.connection.execute("DELETE FROM resource_leases WHERE job_id=?", (job_id,))
                self._refresh_workflow(str(row["workflow_id"]), now)
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def fail(self, job_id: str, request: FailJobRequest) -> None:
        now = self.clock_ns()
        with self._lock:
            self._transaction()
            try:
                row = self._require_lease(job_id, request.worker_id, request.lease_token)
                cancelled = bool(row["cancel_requested"])
                retry = (
                    not cancelled
                    and request.retryable
                    and int(row["attempt"]) < int(row["max_attempts"])
                )
                state = (
                    JobState.CANCELLED
                    if cancelled
                    else (JobState.QUEUED if retry else JobState.FAILED)
                )
                self.connection.execute(
                    """
                    UPDATE jobs SET state=?,error=?,lease_owner='',lease_token='',
                      lease_expires_ns=0,updated_ns=? WHERE job_id=?
                    """,
                    (state.value, f"{request.error_type}: {request.message}", now, job_id),
                )
                self.connection.execute("DELETE FROM resource_leases WHERE job_id=?", (job_id,))
                self._refresh_workflow(str(row["workflow_id"]), now)
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def _refresh_workflow(self, workflow_id: str, now: int) -> None:
        states = {
            JobState(row["state"])
            for row in self.connection.execute(
                "SELECT state FROM jobs WHERE workflow_id=?", (workflow_id,)
            )
        }
        if states == {JobState.SUCCEEDED}:
            workflow = WorkflowState.SUCCEEDED
        elif JobState.FAILED in states:
            workflow = WorkflowState.FAILED
        elif JobState.CANCELLED in states:
            workflow = WorkflowState.CANCELLED
        elif states == {JobState.QUEUED}:
            workflow = WorkflowState.QUEUED
        else:
            workflow = WorkflowState.RUNNING
        self.connection.execute(
            "UPDATE workflows SET state=?,updated_ns=? WHERE workflow_id=?",
            (workflow.value, now, workflow_id),
        )

    def cancel(self, job_id: str) -> None:
        now = self.clock_ns()
        with self._lock:
            row = self.connection.execute(
                "SELECT state,workflow_id FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown job {job_id}")
            state = JobState(row["state"])
            if state is JobState.QUEUED:
                self.connection.execute(
                    "UPDATE jobs SET state=?,cancel_requested=1,updated_ns=? WHERE job_id=?",
                    (JobState.CANCELLED.value, now, job_id),
                )
            elif state not in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
                self.connection.execute(
                    "UPDATE jobs SET state=?,cancel_requested=1,updated_ns=? WHERE job_id=?",
                    (JobState.CANCEL_REQUESTED.value, now, job_id),
                )
            self._refresh_workflow(str(row["workflow_id"]), now)

    def retry(self, job_id: str) -> None:
        now = self.clock_ns()
        with self._lock:
            row = self.connection.execute(
                "SELECT state,workflow_id FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown job {job_id}")
            if JobState(row["state"]) not in {JobState.FAILED, JobState.CANCELLED}:
                raise ValueError("only failed or cancelled jobs can be retried")
            self.connection.execute(
                """
                UPDATE jobs SET state=?,cancel_requested=0,error='',updated_ns=? WHERE job_id=?
                """,
                (JobState.QUEUED.value, now, job_id),
            )
            self._refresh_workflow(str(row["workflow_id"]), now)

    def list_jobs(self, workflow_id: str | None = None) -> tuple[dict[str, Any], ...]:
        query = "SELECT * FROM jobs"
        parameters: tuple[Any, ...] = ()
        if workflow_id:
            query += " WHERE workflow_id=?"
            parameters = (workflow_id,)
        query += " ORDER BY created_ns,ordinal"
        return tuple(self._job_payload(row) for row in self.connection.execute(query, parameters))

    def get_job(self, job_id: str) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown job {job_id}")
        return self._job_payload(row)

    @staticmethod
    def _job_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "workflow_id": row["workflow_id"],
            "run_id": row["run_id"],
            "kind": row["kind"],
            "capability": row["capability"],
            "resource_class": row["resource_class"],
            "state": row["state"],
            "attempt": row["attempt"],
            "max_attempts": row["max_attempts"],
            "cancel_requested": bool(row["cancel_requested"]),
            "result": json.loads(row["result_json"]),
            "error": row["error"],
            "created_ns": row["created_ns"],
            "updated_ns": row["updated_ns"],
        }

    def list_workflows(self) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            "SELECT * FROM workflows ORDER BY created_ns DESC"
        ).fetchall()
        return tuple(dict(row) | {"payload": json.loads(row["payload_json"])} for row in rows)

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM workflows WHERE workflow_id=?", (workflow_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown workflow {workflow_id}")
        return dict(row) | {"payload": json.loads(row["payload_json"])}

    def list_artifacts(self, kind: str | None = None) -> tuple[ArtifactRef, ...]:
        query = "SELECT ref_json FROM artifacts"
        parameters: tuple[Any, ...] = ()
        if kind:
            query += " WHERE kind=?"
            parameters = (kind,)
        query += " ORDER BY created_ns DESC"
        return tuple(
            ArtifactRef.model_validate_json(row["ref_json"])
            for row in self.connection.execute(query, parameters)
        )

    def get_artifact(self, artifact_id: str, *, kind: str | None = None) -> ArtifactRef:
        query = "SELECT ref_json FROM artifacts WHERE artifact_id=?"
        parameters: tuple[Any, ...] = (artifact_id,)
        if kind is not None:
            query += " AND kind=?"
            parameters = (artifact_id, kind)
        row = self.connection.execute(query, parameters).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact {artifact_id}")
        return ArtifactRef.model_validate_json(row["ref_json"])

    def events(self, after: int = 0, limit: int = 500) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            "SELECT * FROM events WHERE sequence>? ORDER BY sequence LIMIT ?",
            (after, min(max(limit, 1), 2000)),
        ).fetchall()
        return tuple(
            {
                "sequence": row["sequence"],
                "event_id": row["event_id"],
                "job_id": row["job_id"],
                "kind": row["kind"],
                "level": row["level"],
                "occurred_ns": row["occurred_ns"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "workflows": self.list_workflows(),
            "jobs": self.list_jobs(),
            "runs": tuple(spec.model_dump(mode="json") for spec in self.list_runs()),
            "artifacts": tuple(ref.model_dump(mode="json") for ref in self.list_artifacts()),
            "events": self.events(max(self._latest_event_sequence() - 100, 0)),
        }

    def _latest_event_sequence(self) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(sequence),0) AS value FROM events"
        ).fetchone()
        return int(row["value"])
