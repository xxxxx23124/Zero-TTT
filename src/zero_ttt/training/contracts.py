"""Typed identities and summaries for persisted training artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class LearnerDataIdentity:
    snapshot_id: str
    sampling_config_sha256: str
    mixture_manifest_sha256: str = ""
    component_snapshot_ids: tuple[str, ...] = ()

    @classmethod
    def from_raw(cls, raw: object) -> "LearnerDataIdentity | None":
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("checkpoint data identity is invalid")
        expected = {field.name for field in fields(cls)}
        if set(raw) != expected:
            raise ValueError("checkpoint data identity is incomplete")
        components = raw["component_snapshot_ids"]
        if not isinstance(components, (list, tuple)) or not all(
            isinstance(value, str) for value in components
        ):
            raise ValueError("checkpoint data identity is invalid")
        string_fields = (
            raw["snapshot_id"],
            raw["sampling_config_sha256"],
            raw["mixture_manifest_sha256"],
        )
        if not all(isinstance(value, str) for value in string_fields):
            raise ValueError("checkpoint data identity is invalid")
        return cls(
            snapshot_id=raw["snapshot_id"],
            sampling_config_sha256=raw["sampling_config_sha256"],
            mixture_manifest_sha256=raw["mixture_manifest_sha256"],
            component_snapshot_ids=tuple(components),
        )


@dataclass(frozen=True, slots=True)
class ModelArtifactIdentity:
    run_id: str
    optimizer_step: int
    samples_seen: int
    config_json: str
    config_sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    identity: ModelArtifactIdentity
    data_identity: LearnerDataIdentity | None
    learner_state: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: object) -> "CheckpointSummary":
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload is invalid")
        if "data_identity" not in payload:
            raise ValueError("checkpoint data identity is incomplete")
        data_identity = LearnerDataIdentity.from_raw(payload["data_identity"])
        state = payload.get("learner_state")
        expected_state = {
            "optimizer_step",
            "samples_seen",
            "ema_pending_samples",
            "next_ema_sample",
            "next_publish_sample",
            "last_published_step",
            "last_published_samples",
            "run_id",
        }
        if not isinstance(state, dict) or set(state) != expected_state:
            raise ValueError("checkpoint learner state is incomplete")
        identity = _artifact_identity(
            run_id=state["run_id"],
            optimizer_step=state["optimizer_step"],
            samples_seen=state["samples_seen"],
            config_json=payload.get("config_json"),
            config_sha256=payload.get("config_sha256"),
            artifact="checkpoint",
        )
        return cls(
            identity=identity,
            data_identity=data_identity,
            learner_state=dict(state),
        )


@dataclass(frozen=True, slots=True)
class PublicationSummary:
    identity: ModelArtifactIdentity

    @classmethod
    def from_payload(cls, payload: object) -> "PublicationSummary":
        if not isinstance(payload, dict):
            raise ValueError("publication payload is invalid")
        return cls(
            identity=_artifact_identity(
                run_id=payload.get("run_id"),
                optimizer_step=payload.get("model_version"),
                samples_seen=payload.get("samples_seen"),
                config_json=payload.get("config_json"),
                config_sha256=payload.get("config_sha256"),
                artifact="publication",
            )
        )


def _artifact_identity(
    *,
    run_id: object,
    optimizer_step: object,
    samples_seen: object,
    config_json: object,
    config_sha256: object,
    artifact: str,
) -> ModelArtifactIdentity:
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{artifact} run identity is invalid")
    for name, value in (
        ("optimizer step", optimizer_step),
        ("samples seen", samples_seen),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{artifact} {name} is invalid")
    if not isinstance(config_json, str) or not isinstance(config_sha256, str):
        raise ValueError(f"{artifact} configuration metadata is invalid")
    digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    if config_sha256 != digest:
        raise ValueError(f"{artifact} configuration SHA-256 does not match")
    return ModelArtifactIdentity(
        run_id=run_id,
        optimizer_step=optimizer_step,
        samples_seen=samples_seen,
        config_json=config_json,
        config_sha256=config_sha256,
    )
