"""Strict file-backed console configuration."""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from zero_ttt.versioning import CONSOLE_CONFIG_SCHEMA

DEFAULT_CONSOLE_CONFIG = Path("configs/console.toml")


@dataclass(frozen=True, slots=True)
class MixtureWeights:
    selfplay: float
    cold_start: float


@dataclass(frozen=True, slots=True)
class ConsoleConfig:
    schema_version: int
    experiment_config: Path
    catalog_path: Path
    store_root: Path
    cold_start_snapshot_id: str
    max_runtime_hours: float
    mixture: MixtureWeights

    @property
    def max_runtime_seconds(self) -> float:
        return self.max_runtime_hours * 60.0 * 60.0


def _configured_path(value: object, field: str, parent: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"console.{field} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else (parent / path).resolve()


def _positive_weight(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"console.mixture.{field} must be numeric")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"console.mixture.{field} must be finite and positive")
    return weight


def _mixture_weights(value: object) -> MixtureWeights:
    if not isinstance(value, dict):
        raise TypeError("console.mixture must be a table")
    expected = {"selfplay_weight", "cold_start_weight"}
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        detail = unknown or missing
        kind = "unknown" if unknown else "missing"
        raise ValueError(f"console.mixture: {kind} fields: {', '.join(detail)}")
    return MixtureWeights(
        selfplay=_positive_weight(value["selfplay_weight"], "selfplay_weight"),
        cold_start=_positive_weight(value["cold_start_weight"], "cold_start_weight"),
    )


def load_console_config(path: str | Path = DEFAULT_CONSOLE_CONFIG) -> ConsoleConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    expected = {
        "schema_version",
        "experiment_config",
        "catalog_path",
        "store_root",
        "cold_start_snapshot_id",
        "max_runtime_hours",
        "mixture",
    }
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown:
        raise ValueError(f"console: unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"console: missing fields: {', '.join(missing)}")
    CONSOLE_CONFIG_SCHEMA.require(raw["schema_version"])
    snapshot_id = raw["cold_start_snapshot_id"]
    if (
        not isinstance(snapshot_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", snapshot_id) is None
        or snapshot_id == "0" * 64
    ):
        raise ValueError(
            "console.cold_start_snapshot_id must be an existing lowercase SHA-256 snapshot ID"
        )
    hours = raw["max_runtime_hours"]
    if isinstance(hours, bool) or not isinstance(hours, int | float):
        raise TypeError("console.max_runtime_hours must be numeric")
    hours = float(hours)
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("console.max_runtime_hours must be finite and positive")
    parent = config_path.resolve().parent
    return ConsoleConfig(
        schema_version=CONSOLE_CONFIG_SCHEMA.current,
        experiment_config=_configured_path(raw["experiment_config"], "experiment_config", parent),
        catalog_path=_configured_path(raw["catalog_path"], "catalog_path", parent),
        store_root=_configured_path(raw["store_root"], "store_root", parent),
        cold_start_snapshot_id=snapshot_id,
        max_runtime_hours=hours,
        mixture=_mixture_weights(raw["mixture"]),
    )
