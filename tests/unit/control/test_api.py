from __future__ import annotations

import pytest


def test_agent_api_v2_exposes_resources_jobs_and_conflicts() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from zero_ttt.control.api import create_app
    from zero_ttt.control.process import OperationConflict

    class StubController:
        def snapshot(self, run_id=None):
            return {"job": None, "selected_run_id": run_id or "", "console": {}}

        def list_profiles(self):
            return ({"profile_id": "tiny"},)

        def data_status(self):
            return {"raw_assets": 1}

        def list_snapshots(self):
            return ({"snapshot_id": "a" * 64},)

        def list_runs(self):
            return ()

        def create_run(self, name, profile_id, snapshot_id):
            return {"name": name, "profile_id": profile_id, "snapshot_id": snapshot_id}

        def start_data(self, operation):
            if operation == "full-import":
                raise OperationConflict("busy")
            return {"operation": operation}

        def start_run(self, run_id, operation, max_runtime_hours):
            return {"run_id": run_id, "operation": operation, "hours": max_runtime_hours}

        def soft_stop(self):
            raise OperationConflict("idle")

        def shutdown(self):
            return None

    with TestClient(create_app(StubController())) as client:
        assert client.get("/api/v2/healthz").json() == {"ok": True}
        assert client.get("/api/v2/status?run_id=abc").json()["selected_run_id"] == "abc"
        assert client.get("/api/v2/profiles").status_code == 200
        assert client.post("/api/v2/data/jobs/scan").status_code == 202
        assert client.post("/api/v2/data/jobs/full-import").status_code == 409
        created = client.post(
            "/api/v2/runs",
            json={"name": "first", "profile_id": "tiny", "cold_snapshot_id": "a" * 64},
        )
        assert created.status_code == 201
        started = client.post(
            f"/api/v2/runs/{'b' * 32}/jobs/train", json={"max_runtime_hours": 1.5}
        )
        assert started.status_code == 202
        assert client.post("/api/v2/jobs/current/soft-stop").status_code == 409
