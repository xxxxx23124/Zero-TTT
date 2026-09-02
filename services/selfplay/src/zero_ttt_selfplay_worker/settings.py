"""Environment-only Self-play Service settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SelfPlaySettings:
    artifact_root: Path
    work_root: Path
    control_url: str
    worker_id: str

    @classmethod
    def from_environment(cls) -> SelfPlaySettings:
        return cls(
            artifact_root=Path(os.environ.get("ZERO_TTT_ARTIFACT_ROOT", "/artifacts")),
            work_root=Path(os.environ.get("ZERO_TTT_SELFPLAY_WORK_ROOT", "/work/selfplay")),
            control_url=os.environ.get("ZERO_TTT_CONTROL_URL", "http://control:8090"),
            worker_id=os.environ.get("ZERO_TTT_WORKER_ID", "selfplay-1"),
        )
