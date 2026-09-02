"""Environment-only Data Service settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataSettings:
    raw_root: Path
    work_root: Path
    artifact_root: Path
    database_path: Path
    control_url: str
    worker_id: str

    @classmethod
    def from_environment(cls) -> DataSettings:
        return cls(
            raw_root=Path(os.environ.get("ZERO_TTT_RAW_ROOT", "/raw")),
            work_root=Path(os.environ.get("ZERO_TTT_DATA_WORK_ROOT", "/work/data")),
            artifact_root=Path(os.environ.get("ZERO_TTT_ARTIFACT_ROOT", "/artifacts")),
            database_path=Path(os.environ.get("ZERO_TTT_DATA_DB", "/state/data/data.sqlite")),
            control_url=os.environ.get("ZERO_TTT_CONTROL_URL", "http://control:8090"),
            worker_id=os.environ.get("ZERO_TTT_WORKER_ID", "data-1"),
        )
