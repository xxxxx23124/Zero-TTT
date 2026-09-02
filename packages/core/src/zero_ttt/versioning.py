"""Single source of truth for machine-readable Zero-TTT schema versions."""

from __future__ import annotations

from dataclasses import dataclass


class UnsupportedSchemaError(ValueError):
    """Raised when a persisted artifact is not on the exact current schema."""


@dataclass(frozen=True, slots=True)
class SchemaSpec:
    name: str
    current: int
    rebuild_hint: str

    def require(self, actual: object) -> None:
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != self.current:
            raise UnsupportedSchemaError(
                f"incompatible {self.name} schema {actual!r}; expected v{self.current}; "
                f"{self.rebuild_hint}"
            )


EXPERIMENT_CONFIG_SCHEMA = SchemaSpec(
    "experiment config",
    8,
    "rewrite the config from a current template",
)
MODEL_ARTIFACT_SCHEMA = SchemaSpec(
    "model artifact",
    1,
    "start a new run and publish a current model artifact",
)
RECORD_SCHEMA = SchemaSpec(
    "trajectory/annotation record",
    1,
    "re-import or recollect the source games",
)
SHARD_SCHEMA = SchemaSpec(
    "NPZ shard",
    1,
    "rebuild the processed data shards",
)
CATALOG_SCHEMA = SchemaSpec(
    "Data Service SQLite",
    1,
    "rebuild the Data Service state and snapshots",
)
SOURCE_MANIFEST_SCHEMA = SchemaSpec(
    "source manifest",
    1,
    "recreate the manifest from the source assets",
)
SELFPLAY_TASK_SCHEMA = SchemaSpec(
    "self-play task manifest",
    1,
    "recollect the self-play task",
)

ALL_SCHEMAS = (
    EXPERIMENT_CONFIG_SCHEMA,
    MODEL_ARTIFACT_SCHEMA,
    RECORD_SCHEMA,
    SHARD_SCHEMA,
    CATALOG_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    SELFPLAY_TASK_SCHEMA,
)
