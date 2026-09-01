from __future__ import annotations

import pytest


def test_agent_api_exposes_status_and_conflict() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from zero_ttt.control.api import create_app
    from zero_ttt.control.process import OperationConflict

    class StubController:
        def start(self, operation: str):
            if operation == "collect":
                raise OperationConflict("busy")
            return {"operation": operation}

        def snapshot(self):
            return {"job": None, "console": {"validated": False}}

        def soft_stop(self):
            raise OperationConflict("idle")

        def shutdown(self):
            return None

    with TestClient(create_app(StubController())) as client:
        assert client.get("/api/v1/healthz").json() == {"ok": True}
        assert client.get("/api/v1/status").status_code == 200
        assert client.post("/api/v1/operations/train").status_code == 202
        assert client.post("/api/v1/operations/collect").status_code == 409
        assert client.post("/api/v1/operations/current/soft-stop").status_code == 409
