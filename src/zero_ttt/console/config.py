"""Validated in-memory inputs for one web-managed training run."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    name: str
    experiment_config: Path
    run_dir: Path
    catalog_path: Path
    store_root: Path
    cold_start_snapshot_id: str
    max_runtime_hours: float

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{32}", self.run_id) is None:
            raise ValueError("run_id must be a server-generated lowercase UUID hex value")
        if not self.name.strip():
            raise ValueError("run name cannot be empty")
        if re.fullmatch(r"[0-9a-f]{64}", self.cold_start_snapshot_id) is None:
            raise ValueError("cold-start snapshot must be a lowercase SHA-256 ID")
        if not math.isfinite(self.max_runtime_hours) or self.max_runtime_hours <= 0:
            raise ValueError("max runtime hours must be finite and positive")

    @property
    def max_runtime_seconds(self) -> float:
        return self.max_runtime_hours * 60.0 * 60.0


ConsoleConfig = RunContext
