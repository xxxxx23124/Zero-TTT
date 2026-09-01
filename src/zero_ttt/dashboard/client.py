"""Small blocking HTTP client used behind NiceGUI async handlers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class AgentClientError(RuntimeError):
    pass


class AgentClient:
    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}{path}", method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise AgentClientError(f"agent returned HTTP {error.code}: {detail}") from error
        except (OSError, ValueError) as error:
            raise AgentClientError(f"training agent is unavailable: {error}") from error

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/status")

    def start(self, operation: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/operations/{operation}")

    def soft_stop(self) -> dict[str, Any]:
        return self._request("POST", "/api/v1/operations/current/soft-stop")
