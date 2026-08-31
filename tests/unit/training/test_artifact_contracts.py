from __future__ import annotations

import hashlib

import pytest

from zero_ttt.training.contracts import (
    CheckpointSummary,
    LearnerDataIdentity,
    PublicationSummary,
)


def _config() -> tuple[str, str]:
    payload = "{}"
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state() -> dict[str, object]:
    return {
        "optimizer_step": 3,
        "samples_seen": 12,
        "ema_pending_samples": 0,
        "next_ema_sample": 16,
        "next_publish_sample": 16,
        "last_published_step": 2,
        "last_published_samples": 8,
        "run_id": "run-a",
    }


def test_checkpoint_and_publication_summaries_share_full_identity() -> None:
    config_json, config_sha256 = _config()
    data_identity = LearnerDataIdentity(
        "mixture:" + "1" * 64,
        "2" * 64,
        "1" * 64,
        ("3" * 64, "4" * 64),
    )
    checkpoint = CheckpointSummary.from_payload(
        {
            "tensor_precision": "float32",
            "config_json": config_json,
            "config_sha256": config_sha256,
            "learner_state": _state(),
            "data_identity": {
                "snapshot_id": data_identity.snapshot_id,
                "sampling_config_sha256": data_identity.sampling_config_sha256,
                "mixture_manifest_sha256": data_identity.mixture_manifest_sha256,
                "component_snapshot_ids": data_identity.component_snapshot_ids,
            },
        }
    )
    publication = PublicationSummary.from_payload(
        {
            "tensor_precision": "float32",
            "config_json": config_json,
            "config_sha256": config_sha256,
            "run_id": "run-a",
            "model_version": 3,
            "samples_seen": 12,
        }
    )
    assert checkpoint.identity == publication.identity
    assert checkpoint.data_identity == data_identity


def test_artifact_summaries_reject_invalid_hashes_and_identity_shapes() -> None:
    config_json, config_sha256 = _config()
    payload = {
        "tensor_precision": "float32",
        "config_json": config_json,
        "config_sha256": config_sha256,
        "learner_state": _state(),
        "data_identity": {
            "snapshot_id": "snapshot",
            "sampling_config_sha256": "sampling",
            "mixture_manifest_sha256": "",
            "component_snapshot_ids": "not-a-sequence-of-identities",
        },
    }
    with pytest.raises(ValueError, match="data identity is invalid"):
        CheckpointSummary.from_payload(payload)

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        PublicationSummary.from_payload(
            {
                "tensor_precision": "float32",
                "config_json": config_json,
                "config_sha256": "0" * 64,
                "run_id": "run-a",
                "model_version": 3,
                "samples_seen": 12,
            }
        )
