"""HTTP-only administration CLI for the public Control API."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Any


def _request(base_url: str, method: str, path: str, payload: object | None = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Control API returned HTTP {error.code}: {detail}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zero-ttt")
    parser.add_argument(
        "--api",
        default=os.environ.get("ZERO_TTT_CONTROL_URL", "http://127.0.0.1:8090"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    run = commands.add_parser("create-run")
    run.add_argument("name")
    run.add_argument("profile_id")
    run.add_argument("cold_snapshot_id")
    workflow = commands.add_parser("submit")
    workflow.add_argument("template", choices=("data-bootstrap", "cold-start", "alpha-zero-round"))
    workflow.add_argument("--run-id", default="")
    workflow.add_argument("--parameters", default="{}")
    workflow.add_argument("--idempotency-key")
    for name in ("cancel", "retry"):
        operation = commands.add_parser(name)
        operation.add_argument("job_id")
    events = commands.add_parser("events")
    events.add_argument("--after", type=int, default=0)
    events.add_argument("--limit", type=int, default=500)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "status":
        result = _request(arguments.api, "GET", "/api/v1/snapshot")
    elif arguments.command == "create-run":
        result = _request(
            arguments.api,
            "POST",
            "/api/v1/runs",
            {
                "name": arguments.name,
                "profile_id": arguments.profile_id,
                "cold_snapshot_id": arguments.cold_snapshot_id,
            },
        )
    elif arguments.command == "submit":
        try:
            parameters = json.loads(arguments.parameters)
        except json.JSONDecodeError as error:
            raise SystemExit(f"--parameters must be a JSON object: {error}") from error
        if not isinstance(parameters, dict):
            raise SystemExit("--parameters must be a JSON object")
        result = _request(
            arguments.api,
            "POST",
            f"/api/v1/workflows/{arguments.template}",
            {
                "run_id": arguments.run_id,
                "parameters": parameters,
                "idempotency_key": arguments.idempotency_key,
            },
        )
    elif arguments.command in {"cancel", "retry"}:
        result = _request(
            arguments.api,
            "POST",
            f"/api/v1/jobs/{arguments.job_id}/{arguments.command}",
            {},
        )
    else:
        result = _request(
            arguments.api,
            "GET",
            f"/api/v1/events?after={arguments.after}&limit={arguments.limit}",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


__all__ = ["main"]
