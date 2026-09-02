"""Strict HTTP and artifact contracts; no service implementation dependencies."""

from __future__ import annotations

import re
import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONTRACT_SCHEMA_VERSION = 1
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactKind(StrEnum):
    SOURCE_MANIFEST = "source-manifest"
    DATA_VERIFICATION = "data-verification"
    DATASET_SNAPSHOT = "dataset-snapshot"
    SELFPLAY_BUNDLE = "selfplay-bundle"
    CHECKPOINT = "checkpoint"
    PUBLICATION = "publication"
    METRICS = "metrics"


class ResourceClass(StrEnum):
    NONE = "none"
    DATA_WRITER = "data-writer"
    GPU_EXCLUSIVE = "gpu-exclusive"


class WorkerCapability(StrEnum):
    DATA = "data"
    TRAINER = "trainer"
    SELFPLAY = "selfplay"


class JobState(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel-requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATES = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


class WorkflowState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowTemplate(StrEnum):
    DATA_BOOTSTRAP = "data-bootstrap"
    COLD_START = "cold-start"
    ALPHA_ZERO_ROUND = "alpha-zero-round"


class EventLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ArtifactRef(StrictModel):
    schema_version: int = CONTRACT_SCHEMA_VERSION
    kind: ArtifactKind
    artifact_id: str
    format_version: int = Field(ge=1)
    sha256: str
    uri: str
    size_bytes: int = Field(ge=0)
    labels: dict[str, str] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _current_schema(cls, value: int) -> int:
        if value != CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"expected contract schema v{CONTRACT_SCHEMA_VERSION}")
        return value

    @field_validator("artifact_id")
    @classmethod
    def _valid_artifact_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("artifact_id must be a lowercase stable identifier")
        return value

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        if _HEX_64.fullmatch(value) is None:
            raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")
        return value

    @field_validator("uri")
    @classmethod
    def _valid_uri(cls, value: str) -> str:
        if not value.startswith("artifact://"):
            raise ValueError("artifact URI must use artifact://")
        return value


class DomainEvent(StrictModel):
    schema_version: int = CONTRACT_SCHEMA_VERSION
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    job_id: str
    kind: str
    level: EventLevel = EventLevel.INFO
    occurred_ns: int = Field(default_factory=time.time_ns, gt=0)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _valid_kind(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("event kind must be a lowercase stable identifier")
        return value


class JobEnvelope(StrictModel):
    schema_version: int = CONTRACT_SCHEMA_VERSION
    job_id: str
    workflow_id: str
    run_id: str = ""
    kind: str
    capability: WorkerCapability
    resource_class: ResourceClass
    attempt: int = Field(ge=1)
    lease_token: str
    lease_expires_ns: int = Field(gt=0)
    idempotency_key: str
    payload: dict[str, Any]
    inputs: tuple[ArtifactRef, ...] = ()


class LeaseJobRequest(StrictModel):
    worker_id: str
    capability: WorkerCapability
    lease_seconds: int = Field(default=60, ge=10, le=3600)
    wait_seconds: int = Field(default=20, ge=0, le=30)


class HeartbeatRequest(StrictModel):
    worker_id: str
    lease_token: str
    lease_seconds: int = Field(default=60, ge=10, le=3600)


class HeartbeatResponse(StrictModel):
    lease_expires_ns: int
    cancel_requested: bool


class CompleteJobRequest(StrictModel):
    worker_id: str
    lease_token: str
    result: dict[str, Any] = Field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()


class FailJobRequest(StrictModel):
    worker_id: str
    lease_token: str
    error_type: str
    message: str
    retryable: bool = True


class WorkerRegistration(StrictModel):
    worker_id: str
    capability: WorkerCapability
    version: str


class RunSpec(StrictModel):
    schema_version: int = CONTRACT_SCHEMA_VERSION
    run_id: str
    name: str = Field(min_length=1, max_length=80)
    profile_id: str
    profile_sha256: str
    profile: dict[str, Any]
    cold_snapshot: ArtifactRef
    created_ns: int = Field(default_factory=time.time_ns, gt=0)

    @field_validator("profile_sha256")
    @classmethod
    def _valid_profile_sha256(cls, value: str) -> str:
        if _HEX_64.fullmatch(value) is None:
            raise ValueError("profile_sha256 must be a lowercase SHA-256")
        return value
