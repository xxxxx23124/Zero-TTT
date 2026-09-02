"""Environment-only Trainer Service settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainerSettings:
    artifact_root: Path
    work_root: Path
    control_url: str
    worker_id: str

    @classmethod
    def from_environment(cls) -> TrainerSettings:
        return cls(
            artifact_root=Path(os.environ.get("ZERO_TTT_ARTIFACT_ROOT", "/artifacts")),
            work_root=Path(os.environ.get("ZERO_TTT_TRAINER_WORK_ROOT", "/work/trainer")),
            control_url=os.environ.get("ZERO_TTT_CONTROL_URL", "http://control:8090"),
            worker_id=os.environ.get("ZERO_TTT_WORKER_ID", "trainer-1"),
        )
