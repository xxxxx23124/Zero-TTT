from __future__ import annotations

from fastapi.testclient import TestClient
from zero_ttt_control.api import create_app
from zero_ttt_control.store import ControlStore


def test_openapi_and_strict_workflow_request(tmp_path) -> None:
    client = TestClient(create_app(ControlStore(tmp_path / "control.sqlite")))
    assert client.get("/healthz").json() == {"ok": True}
    document = client.get("/openapi.json").json()
    assert "/internal/v1/jobs/lease" in document["paths"]
    response = client.post(
        "/api/v1/workflows/data-bootstrap",
        json={"parameters": {}, "unexpected": True},
    )
    assert response.status_code == 422
