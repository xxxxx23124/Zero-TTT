"""Pure dashboard projections kept independent from NiceGUI."""

from __future__ import annotations

from typing import Any

ACTIVE_STATES = {"STARTING", "RUNNING", "STOP_REQUESTED"}


def button_availability(
    snapshot: dict[str, Any], run_selected: bool = True
) -> dict[str, bool]:
    job = snapshot.get("job") or {}
    console = snapshot.get("console") or {}
    job_state = str(job.get("state", "IDLE"))
    idle = job_state not in ACTIVE_STATES
    validated = bool(console.get("validated"))
    ready = console.get("operation") == "READY"
    selfplay = console.get("selfplay") or {}
    cold = console.get("phase") == "COLD_START"
    return {
        "reconcile": idle and run_selected,
        "train": idle and run_selected and validated and ready,
        "collect": idle and run_selected and validated and ready and bool(console.get("publication_path")),
        "warm_start": (
            idle
            and run_selected
            and validated
            and ready
            and cold
            and bool(console.get("checkpoint_path"))
            and int(selfplay.get("games", 0)) > 0
        ),
        "soft_stop": job_state in {"STARTING", "RUNNING", "STOP_REQUESTED"}
        and bool(job.get("stoppable", job.get("operation") != "reconcile")),
    }


def overview(snapshot: dict[str, Any]) -> dict[str, str]:
    console = snapshot.get("console") or {}
    job = snapshot.get("job") or {}
    selfplay = console.get("selfplay") or {}
    return {
        "phase": str(console.get("phase", "-")),
        "operation": str(console.get("operation", "-")),
        "job": f"{job.get('operation', '-')} / {job.get('state', 'IDLE')}",
        "run": str(console.get("run_id", "-")) or "-",
        "step": str(console.get("optimizer_step", 0)),
        "samples": str(console.get("samples_seen", 0)),
        "artifacts": str(console.get("artifact_consistency", "尚未校验")),
        "selfplay": (
            f"sealed={selfplay.get('sealed_tasks', 0)}, games={selfplay.get('games', 0)}, "
            f"positions={selfplay.get('positions', 0)}"
        ),
        "pending": (
            f"games={console.get('pending_games', 0)}, "
            f"positions={console.get('pending_positions', 0)}"
        ),
        "outcome": str(console.get("last_outcome", "")) or "-",
    }
