from __future__ import annotations

from zero_ttt_contracts import (
    ArtifactKind,
    ArtifactRef,
    CompleteJobRequest,
    DomainEvent,
    HeartbeatRequest,
    JobState,
    LeaseJobRequest,
    WorkerCapability,
    WorkerRegistration,
    WorkflowTemplate,
)
from zero_ttt_control.store import ControlStore


def worker(store: ControlStore, worker_id: str, capability: WorkerCapability) -> None:
    store.register_worker(
        WorkerRegistration(
            worker_id=worker_id,
            capability=capability,
            version="test",
        )
    )


def lease(store: ControlStore, worker_id: str, capability: WorkerCapability):
    return store.lease_job(
        LeaseJobRequest(
            worker_id=worker_id,
            capability=capability,
            lease_seconds=60,
        )
    )


def test_bootstrap_dependencies_and_artifact_handoff(tmp_path) -> None:
    store = ControlStore(tmp_path / "control.sqlite")
    worker(store, "data-1", WorkerCapability.DATA)
    workflow = store.submit_workflow(WorkflowTemplate.DATA_BOOTSTRAP, {"trial_games": 4})
    first = lease(store, "data-1", WorkerCapability.DATA)
    assert first is not None
    assert first.kind == "data.scan"
    artifact = ArtifactRef(
        kind=ArtifactKind.DATASET_SNAPSHOT,
        artifact_id="dataset.scan",
        format_version=1,
        sha256="b" * 64,
        uri="artifact://data/scan.json",
        size_bytes=12,
    )
    completion = CompleteJobRequest(
        worker_id="data-1",
        lease_token=first.lease_token,
        artifacts=(artifact,),
    )
    store.complete(first.job_id, completion)
    store.complete(first.job_id, completion)
    second = lease(store, "data-1", WorkerCapability.DATA)
    assert second is not None
    assert second.kind == "data.trial-import"
    assert second.inputs == (artifact,)
    assert len(store.list_jobs(workflow)) == 7


def test_gpu_lease_is_exclusive_and_cancel_is_durable(tmp_path) -> None:
    store = ControlStore(tmp_path / "control.sqlite")
    snapshot = ArtifactRef(
        kind=ArtifactKind.DATASET_SNAPSHOT,
        artifact_id="dataset.cold",
        format_version=1,
        sha256="c" * 64,
        uri="artifact://data/cold.json",
        size_bytes=1,
    )
    store.connection.execute(
        "INSERT INTO workflows VALUES(?,?,?,?,?,?,?)",
        ("seed", "", "seed", "succeeded", "{}", 1, 1),
    )
    store.connection.execute(
        "INSERT INTO jobs(job_id,workflow_id,ordinal,kind,capability,resource_class,state,payload_json,idempotency_key,created_ns,updated_ns) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        ("seed-job", "seed", 0, "seed", "data", "none", "succeeded", "{}", "seed", 1, 1),
    )
    store.connection.execute(
        "INSERT INTO artifacts VALUES(?,?,?,?,?)",
        (snapshot.artifact_id, "seed-job", snapshot.kind.value, snapshot.model_dump_json(), 1),
    )
    from zero_ttt_contracts import RunSpec

    run = RunSpec(
        run_id="r" * 32,
        name="run",
        profile_id="test",
        profile_sha256="d" * 64,
        profile={},
        cold_snapshot=snapshot,
    )
    store.create_run(run)
    first_workflow = store.submit_workflow(
        WorkflowTemplate.COLD_START, {}, run_id=run.run_id, idempotency_key="one"
    )
    store.submit_workflow(
        WorkflowTemplate.ALPHA_ZERO_ROUND,
        {},
        run_id=run.run_id,
        idempotency_key="two",
    )
    worker(store, "trainer-1", WorkerCapability.TRAINER)
    worker(store, "selfplay-1", WorkerCapability.SELFPLAY)
    first = lease(store, "trainer-1", WorkerCapability.TRAINER)
    assert first is not None
    assert lease(store, "selfplay-1", WorkerCapability.SELFPLAY) is None
    store.cancel(first.job_id)
    status = store.heartbeat(
        first.job_id,
        HeartbeatRequest(
            worker_id="trainer-1",
            lease_token=first.lease_token,
            lease_seconds=60,
        ),
    )
    assert status.cancel_requested
    assert store.list_jobs(first_workflow)[0]["state"] == JobState.CANCEL_REQUESTED.value


def test_expired_lease_is_requeued_after_restart(tmp_path) -> None:
    now = 1_000_000_000
    store = ControlStore(tmp_path / "control.sqlite", clock_ns=lambda: now)
    worker(store, "data-1", WorkerCapability.DATA)
    store.submit_workflow(WorkflowTemplate.DATA_BOOTSTRAP, {})
    job = store.lease_job(
        LeaseJobRequest(worker_id="data-1", capability=WorkerCapability.DATA, lease_seconds=10)
    )
    assert job is not None
    store.close()
    later = now + 11_000_000_000
    restored = ControlStore(tmp_path / "control.sqlite", clock_ns=lambda: later)
    worker(restored, "data-2", WorkerCapability.DATA)
    reclaimed = restored.lease_job(
        LeaseJobRequest(worker_id="data-2", capability=WorkerCapability.DATA, lease_seconds=10)
    )
    assert reclaimed is not None
    assert reclaimed.job_id == job.job_id
    assert reclaimed.attempt == 2


def test_event_retry_and_cursor_resume_are_idempotent(tmp_path) -> None:
    store = ControlStore(tmp_path / "control.sqlite")
    worker(store, "data-1", WorkerCapability.DATA)
    store.submit_workflow(WorkflowTemplate.DATA_BOOTSTRAP, {})
    job = lease(store, "data-1", WorkerCapability.DATA)
    assert job is not None
    event = DomainEvent(event_id="e" * 32, job_id=job.job_id, kind="test.progress")
    first = store.append_event(job.job_id, "data-1", job.lease_token, event)
    second = store.append_event(job.job_id, "data-1", job.lease_token, event)
    assert first == second
    assert store.events(first - 1, 1)[0]["event_id"] == event.event_id
    assert store.events(first, 1) == ()


def test_expired_final_attempt_fails_instead_of_looping_forever(tmp_path) -> None:
    clock = [1_000_000_000]
    store = ControlStore(tmp_path / "control.sqlite", clock_ns=lambda: clock[0])
    worker(store, "data-1", WorkerCapability.DATA)
    workflow = store.submit_workflow(WorkflowTemplate.DATA_BOOTSTRAP, {})
    job_id = store.list_jobs(workflow)[0]["job_id"]
    store.connection.execute("UPDATE jobs SET max_attempts=1 WHERE job_id=?", (job_id,))
    job = store.lease_job(
        LeaseJobRequest(worker_id="data-1", capability=WorkerCapability.DATA, lease_seconds=10)
    )
    assert job is not None
    clock[0] += 11_000_000_000
    assert lease(store, "data-1", WorkerCapability.DATA) is None
    failed = store.list_jobs(workflow)[0]
    assert failed["state"] == JobState.FAILED.value
    assert "lease expired" in failed["error"]
