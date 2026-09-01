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
    8,
    "start a new run and publish a current model artifact",
)
RECORD_SCHEMA = SchemaSpec(
    "trajectory/annotation record",
    4,
    "re-import or recollect the source games",
)
SHARD_SCHEMA = SchemaSpec(
    "NPZ shard",
    4,
    "rebuild the processed data shards",
)
CATALOG_SCHEMA = SchemaSpec(
    "SQLite catalog",
    4,
    "rebuild the catalog and snapshots",
)
SOURCE_MANIFEST_SCHEMA = SchemaSpec(
    "source manifest",
    2,
    "recreate the manifest from the source assets",
)
TRAINING_MIXTURE_SCHEMA = SchemaSpec(
    "training mixture manifest",
    2,
    "recreate the mixture from current snapshots",
)
SELFPLAY_TASK_SCHEMA = SchemaSpec(
    "self-play task manifest",
    2,
    "recollect the self-play task",
)
RUN_SPEC_SCHEMA = SchemaSpec(
    "training run specification",
    1,
    "create a new training run from the web interface",
)
CONSOLE_STATE_SCHEMA = SchemaSpec(
    "training console state",
    2,
    "inspect the run artifacts and recreate only the console state file",
)

ALL_SCHEMAS = (
    EXPERIMENT_CONFIG_SCHEMA,
    MODEL_ARTIFACT_SCHEMA,
    RECORD_SCHEMA,
    SHARD_SCHEMA,
    CATALOG_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    TRAINING_MIXTURE_SCHEMA,
    SELFPLAY_TASK_SCHEMA,
    RUN_SPEC_SCHEMA,
    CONSOLE_STATE_SCHEMA,
)
