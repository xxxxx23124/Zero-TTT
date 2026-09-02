"""Pure read-model formatting for the NiceGUI page."""

from __future__ import annotations

from typing import Any

ACTIVE_STATES = {"leased", "running", "cancel-requested"}


def summary(snapshot: dict[str, Any]) -> dict[str, str]:
    jobs: list[dict[str, Any]] = list(snapshot.get("jobs", ()))
    active = [job for job in jobs if job.get("state") in ACTIVE_STATES]
    failed = [job for job in jobs if job.get("state") == "failed"]
    artifacts: list[dict[str, Any]] = list(snapshot.get("artifacts", ()))
    return {
        "active": str(len(active)),
        "failed": str(len(failed)),
        "datasets": str(sum(item.get("kind") == "dataset-snapshot" for item in artifacts)),
        "publications": str(sum(item.get("kind") == "publication" for item in artifacts)),
    }


def latest_jobs(snapshot: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = list(snapshot.get("jobs", ()))
    return sorted(jobs, key=lambda item: int(item.get("updated_ns", 0)), reverse=True)[:limit]
