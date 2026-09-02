"""Small standard-library client for the internal control API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel
from zero_ttt_contracts import (
    CompleteJobRequest,
    DomainEvent,
    FailJobRequest,
    HeartbeatRequest,
    HeartbeatResponse,
    JobEnvelope,
    LeaseJobRequest,
    WorkerRegistration,
)


class ControlClientError(RuntimeError):
    pass


class ControlClient:
    def __init__(self, base_url: str, timeout: float = 35.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: BaseModel | dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload = None
        if isinstance(body, BaseModel):
            payload = body.model_dump_json().encode("utf-8")
        elif body is not None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=payload,
            method=method,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise ControlClientError(f"control API returned HTTP {error.code}: {detail}") from error
        except (OSError, ValueError) as error:
            raise ControlClientError(f"control API request failed: {error}") from error

    def register(self, registration: WorkerRegistration) -> None:
        self._request("POST", "/internal/v1/workers/register", registration)

    def lease(self, request: LeaseJobRequest) -> JobEnvelope | None:
        raw = self._request("POST", "/internal/v1/jobs/lease", request)["job"]
        return None if raw is None else JobEnvelope.model_validate(raw)

    def heartbeat(self, job_id: str, request: HeartbeatRequest) -> HeartbeatResponse:
        raw = self._request("POST", f"/internal/v1/jobs/{job_id}/heartbeat", request)
        return HeartbeatResponse.model_validate(raw)

    def event(self, job: JobEnvelope, worker_id: str, event: DomainEvent) -> int:
        raw = self._request(
            "POST",
            f"/internal/v1/jobs/{job.job_id}/events",
            event,
            {"X-Worker-ID": worker_id, "X-Lease-Token": job.lease_token},
        )
        return int(raw["sequence"])

    def complete(self, job_id: str, request: CompleteJobRequest) -> None:
        self._request("POST", f"/internal/v1/jobs/{job_id}/complete", request)

    def fail(self, job_id: str, request: FailJobRequest) -> None:
        self._request("POST", f"/internal/v1/jobs/{job_id}/fail", request)
