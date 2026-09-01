"""Small blocking API-v2 client used behind NiceGUI async handlers."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class AgentClientError(RuntimeError):
    pass


class AgentClient:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            method=method,
            data=data,
            headers={} if data is None else {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(raw).get("detail", raw)
            except ValueError:
                detail = raw
            raise AgentClientError(str(detail)) from error
        except (OSError, ValueError) as error:
            raise AgentClientError(f"训练代理不可用: {error}") from error

    def status(self, run_id: str = "") -> dict[str, Any]:
        query = "" if not run_id else f"?{urllib.parse.urlencode({'run_id': run_id})}"
        return self._request("GET", f"/api/v2/status{query}")

    def profiles(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/api/v2/profiles").get("profiles", ()))

    def data_status(self) -> dict[str, Any]:
        return self._request("GET", "/api/v2/data")

    def snapshots(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/api/v2/snapshots").get("snapshots", ()))

    def runs(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/api/v2/runs").get("runs", ()))

    def start_data(self, operation: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v2/data/jobs/{operation}")

    def create_run(self, name: str, profile_id: str, snapshot_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v2/runs",
            {"name": name, "profile_id": profile_id, "cold_snapshot_id": snapshot_id},
        )

    def start_run(
        self, run_id: str, operation: str, max_runtime_hours: float
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v2/runs/{run_id}/jobs/{operation}",
            {"max_runtime_hours": max_runtime_hours},
        )

    def soft_stop(self) -> dict[str, Any]:
        return self._request("POST", "/api/v2/jobs/current/soft-stop")
