from __future__ import annotations

import pytest
from zero_ttt.versioning import (
    ALL_SCHEMAS,
    CATALOG_SCHEMA,
    EXPERIMENT_CONFIG_SCHEMA,
    MODEL_ARTIFACT_SCHEMA,
    RECORD_SCHEMA,
    SELFPLAY_TASK_SCHEMA,
    SHARD_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
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
        SELFPLAY_TASK_SCHEMA,
    )
    assert [schema.current for schema in ALL_SCHEMAS] == [
        8,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert len({schema.name for schema in ALL_SCHEMAS}) == len(ALL_SCHEMAS)


@pytest.mark.parametrize("actual", (None, 3, 7, "8", True, 9))
def test_schema_requires_the_exact_current_integer(actual: object) -> None:
    with pytest.raises(
        UnsupportedSchemaError,
        match=r"experiment config.*expected v8.*current template",
    ):
        EXPERIMENT_CONFIG_SCHEMA.require(actual)


def test_schema_accepts_the_exact_current_integer() -> None:
    EXPERIMENT_CONFIG_SCHEMA.require(EXPERIMENT_CONFIG_SCHEMA.current)
