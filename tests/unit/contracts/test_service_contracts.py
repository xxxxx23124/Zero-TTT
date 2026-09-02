from __future__ import annotations

import pytest
from pydantic import ValidationError
from zero_ttt_contracts import (
    ArtifactKind,
    ArtifactRef,
    JobEnvelope,
    ResourceClass,
    WorkerCapability,
)


def artifact() -> ArtifactRef:
    return ArtifactRef(
        kind=ArtifactKind.DATASET_SNAPSHOT,
        artifact_id="dataset.test",
        format_version=1,
        sha256="a" * 64,
        uri="artifact://data/snapshots/test.json",
        size_bytes=42,
    )


def test_artifact_contract_is_strict_and_versioned() -> None:
    assert artifact().schema_version == 1
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate(artifact().model_dump() | {"unknown": True})
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate(artifact().model_dump() | {"sha256": "ABC"})
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate(artifact().model_dump() | {"uri": "file:///tmp/a"})


def test_job_envelope_round_trip_preserves_artifact_identity() -> None:
    envelope = JobEnvelope(
        job_id="j" * 32,
        workflow_id="w" * 32,
        kind="trainer.cold-start",
        capability=WorkerCapability.TRAINER,
        resource_class=ResourceClass.GPU_EXCLUSIVE,
        attempt=1,
        lease_token="token",
        lease_expires_ns=1,
        idempotency_key="identity",
        payload={"steps": 1},
        inputs=(artifact(),),
    )
    assert JobEnvelope.model_validate_json(envelope.model_dump_json()) == envelope
