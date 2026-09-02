"""Public Control API client; UI never imports another service package."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class ApiError(RuntimeError):
    pass


class ControlApiClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ApiError(f"HTTP {error.code}: {detail}") from error
        except (OSError, ValueError) as error:
            raise ApiError(str(error)) from error

    def snapshot(self) -> dict[str, Any]:
        return self.request("GET", "/api/v1/snapshot")

    def profiles(self) -> list[dict[str, Any]]:
        return list(self.request("GET", "/api/v1/profiles")["profiles"])

    def create_run(self, name: str, profile_id: str, snapshot_id: str) -> dict[str, Any]:
        return self.request(
            "POST",
            "/api/v1/runs",
            {"name": name, "profile_id": profile_id, "cold_snapshot_id": snapshot_id},
        )

    def submit_workflow(
        self, template: str, *, run_id: str = "", parameters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.request(
            "POST",
            f"/api/v1/workflows/{template}",
            {"run_id": run_id, "parameters": parameters or {}},
        )

    def cancel(self, job_id: str) -> None:
        self.request("POST", f"/api/v1/jobs/{job_id}/cancel", {})

    def retry(self, job_id: str) -> None:
        self.request("POST", f"/api/v1/jobs/{job_id}/retry", {})

    def events(self, after: int) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode({"after": after, "limit": 500})
        return list(self.request("GET", f"/api/v1/events?{query}")["events"])
