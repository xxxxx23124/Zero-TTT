from __future__ import annotations

import pytest

from zero_ttt.versioning import (
    ALL_SCHEMAS,
    CATALOG_SCHEMA,
    CONSOLE_CONFIG_SCHEMA,
    CONSOLE_STATE_SCHEMA,
    EXPERIMENT_CONFIG_SCHEMA,
    MODEL_ARTIFACT_SCHEMA,
    RECORD_SCHEMA,
    SELFPLAY_TASK_SCHEMA,
    SHARD_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    TRAINING_MIXTURE_SCHEMA,
    UnsupportedSchemaError,
)


def test_schema_registry_has_the_clean_break_versions() -> None:
    assert ALL_SCHEMAS == (
        EXPERIMENT_CONFIG_SCHEMA,
        MODEL_ARTIFACT_SCHEMA,
        RECORD_SCHEMA,
        SHARD_SCHEMA,
        CATALOG_SCHEMA,
        SOURCE_MANIFEST_SCHEMA,
        TRAINING_MIXTURE_SCHEMA,
        SELFPLAY_TASK_SCHEMA,
        CONSOLE_CONFIG_SCHEMA,
        CONSOLE_STATE_SCHEMA,
    )
    assert [schema.current for schema in ALL_SCHEMAS] == [
        7,
        7,
        4,
        4,
        4,
        2,
        2,
        2,
        2,
        1,
    ]
    assert len({schema.name for schema in ALL_SCHEMAS}) == len(ALL_SCHEMAS)


@pytest.mark.parametrize("actual", (None, 3, 6, "7", True, 8))
def test_schema_requires_the_exact_current_integer(actual: object) -> None:
    with pytest.raises(
        UnsupportedSchemaError,
        match=r"experiment config.*expected v7.*current template",
    ):
        EXPERIMENT_CONFIG_SCHEMA.require(actual)


def test_schema_accepts_the_exact_current_integer() -> None:
    EXPERIMENT_CONFIG_SCHEMA.require(EXPERIMENT_CONFIG_SCHEMA.current)
