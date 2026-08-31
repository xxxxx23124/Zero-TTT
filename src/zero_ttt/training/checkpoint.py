"""Atomic full checkpoints, immutable FP32 publications, and fault snapshots."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from zero_ttt._io import atomic_write_json, fsync_directory, sha256_file
from zero_ttt.precision import TENSOR_PRECISION, require_fp32_state
from zero_ttt.versioning import MODEL_ARTIFACT_SCHEMA


class CheckpointManager:
    def __init__(self, run_dir: str | Path, keep: int, publication_keep: int = 1) -> None:
        if keep <= 0 or publication_keep <= 0:
            raise ValueError("checkpoint retention limits must be positive")
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.publication_dir = self.run_dir / "published"
        self.fault_dir = self.run_dir / "faults"
        self.keep = keep
        self.publication_keep = publication_keep
        for directory in (self.checkpoint_dir, self.publication_dir, self.fault_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_torch_save(payload: Any, destination: Path) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".t", dir=destination.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _validate_fp32_payload(payload: dict[str, Any]) -> None:
        if payload.get("tensor_precision") != TENSOR_PRECISION:
            raise ValueError(f"model artifact tensor precision must be {TENSOR_PRECISION}")
        for state_name in ("fast_state", "slow_state"):
            state = payload.get(state_name)
            if isinstance(state, dict):
                require_fp32_state(state, f"model artifact {state_name}")

    def save_full(self, step: int, payload: dict[str, Any]) -> Path:
        MODEL_ARTIFACT_SCHEMA.require(payload.get("checkpoint_schema_version"))
        self._validate_fp32_payload(payload)
        destination = self.checkpoint_dir / f"step_{step:012d}.pt"
        self._atomic_torch_save(payload, destination)
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        for old in checkpoints[: -self.keep]:
            old.unlink()
        return destination

    @staticmethod
    def _atomic_json_save(payload: dict[str, Any], destination: Path) -> None:
        atomic_write_json(destination, payload)

    @staticmethod
    def _sha256(path: Path) -> str:
        return sha256_file(path)

    @staticmethod
    def _publication_payload_matches(
        existing: dict[str, Any],
        expected: dict[str, Any],
    ) -> bool:
        identity_fields = (
            "checkpoint_schema_version",
            "tensor_precision",
            "config_json",
            "config_sha256",
            "model_version",
            "run_id",
            "samples_seen",
        )
        if any(existing.get(name) != expected.get(name) for name in identity_fields):
            return False
        existing_state = existing.get("slow_state")
        expected_state = expected.get("slow_state")
        if not isinstance(existing_state, dict) or not isinstance(expected_state, dict):
            return False
        if existing_state.keys() != expected_state.keys():
            return False
        return all(
            isinstance(existing_state[name], torch.Tensor)
            and isinstance(expected_state[name], torch.Tensor)
            and torch.equal(existing_state[name].cpu(), expected_state[name].cpu())
            for name in expected_state
        )

    def save_publication(
        self,
        run_id: str,
        step: int,
        samples_seen: int,
        slow_state: dict[str, torch.Tensor],
        metadata: dict[str, Any],
    ) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ValueError("run_id contains unsafe path characters")
        MODEL_ARTIFACT_SCHEMA.require(metadata.get("checkpoint_schema_version"))
        if metadata.get("tensor_precision") != TENSOR_PRECISION:
            raise ValueError(f"publication tensor precision must be {TENSOR_PRECISION}")
        payload = self._build_publication_payload(run_id, step, samples_seen, slow_state, metadata)
        run_directory = self.publication_dir / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        immutable = run_directory / f"step_{step:012d}"
        if immutable.exists():
            model_path, digest = self._validate_existing_publication(
                immutable, run_id, step, samples_seen, payload
            )
            self._write_current_pointer(model_path, run_id, step, samples_seen, digest)
            return model_path
        model_path, digest = self._write_new_publication(
            run_directory, immutable, run_id, step, samples_seen, payload
        )
        self._write_current_pointer(model_path, run_id, step, samples_seen, digest)
        self._prune_publications(run_directory)
        return model_path

    @staticmethod
    def _build_publication_payload(
        run_id: str,
        step: int,
        samples_seen: int,
        slow_state: dict[str, torch.Tensor],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        require_fp32_state(slow_state, "publication slow_state")
        fp32_state = {name: tensor.detach().cpu() for name, tensor in slow_state.items()}
        return {
            **metadata,
            "model_version": step,
            "run_id": run_id,
            "samples_seen": samples_seen,
            "slow_state": fp32_state,
        }

    def _validate_existing_publication(
        self,
        immutable: Path,
        run_id: str,
        step: int,
        samples_seen: int,
        payload: dict[str, Any],
    ) -> tuple[Path, str]:
        metadata_path = immutable / "metadata.json"
        model_path = immutable / "model.pt"
        if not metadata_path.is_file() or not model_path.is_file():
            raise FileExistsError(f"incomplete publication already exists: {run_id}:{step}")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        MODEL_ARTIFACT_SCHEMA.require(existing.get("schema_version"))
        digest = self._sha256(model_path)
        identity = (run_id, step, samples_seen, digest)
        stored = (
            existing.get("run_id"),
            existing.get("optimizer_step"),
            existing.get("samples_seen"),
            existing.get("sha256"),
        )
        existing_payload = self.load_publication(model_path, map_location="cpu")
        if stored != identity or not self._publication_payload_matches(existing_payload, payload):
            raise FileExistsError(f"conflicting publication already exists: {run_id}:{step}")
        return model_path, digest

    def _write_new_publication(
        self,
        run_directory: Path,
        immutable: Path,
        run_id: str,
        step: int,
        samples_seen: int,
        payload: dict[str, Any],
    ) -> tuple[Path, str]:
        temporary = Path(tempfile.mkdtemp(prefix=".p-", dir=run_directory))
        try:
            model_path = temporary / "model.pt"
            self._atomic_torch_save(payload, model_path)
            digest = self._sha256(model_path)
            publication_metadata = {
                "schema_version": MODEL_ARTIFACT_SCHEMA.current,
                "run_id": run_id,
                "optimizer_step": step,
                "samples_seen": samples_seen,
                "sha256": digest,
                "model_file": "model.pt",
            }
            self._atomic_json_save(publication_metadata, temporary / "metadata.json")
            os.replace(temporary, immutable)
            fsync_directory(run_directory)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return immutable / "model.pt", digest

    def _write_current_pointer(
        self,
        model_path: Path,
        run_id: str,
        step: int,
        samples_seen: int,
        digest: str,
    ) -> None:
        self._atomic_json_save(
            {
                "schema_version": MODEL_ARTIFACT_SCHEMA.current,
                "run_id": run_id,
                "optimizer_step": step,
                "samples_seen": samples_seen,
                "sha256": digest,
                "model_path": model_path.relative_to(self.run_dir).as_posix(),
            },
            self.publication_dir / "current.json",
        )

    def _prune_publications(self, run_directory: Path) -> None:
        publications = sorted(path for path in run_directory.glob("step_*") if path.is_dir())
        for old in publications[: -self.publication_keep]:
            shutil.rmtree(old)

    def save_fault(self, step: int, payload: dict[str, Any], reason: str) -> Path:
        MODEL_ARTIFACT_SCHEMA.require(payload.get("checkpoint_schema_version"))
        self._validate_fp32_payload(payload)
        payload = {**payload, "fault_reason": reason, "fault_time_ns": time.time_ns()}
        destination = self.fault_dir / f"fault_{step:012d}_{time.time_ns()}.pt"
        self._atomic_torch_save(payload, destination)
        return destination

    def latest_checkpoint(self) -> Path | None:
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def current_publication(self) -> Path | None:
        current = self.publication_dir / "current.json"
        if current.exists():
            payload = json.loads(current.read_text(encoding="utf-8"))
            MODEL_ARTIFACT_SCHEMA.require(payload.get("schema_version"))
            path = self.run_dir / payload["model_path"]
            if not path.is_file() or self._sha256(path) != payload["sha256"]:
                raise ValueError("current publication pointer is invalid")
            return path
        return None

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        MODEL_ARTIFACT_SCHEMA.require(payload.get("checkpoint_schema_version"))
        CheckpointManager._validate_fp32_payload(payload)
        return payload

    @classmethod
    def load_publication(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        payload = cls.load(path, map_location=map_location)
        if not isinstance(payload.get("model_version"), int):
            raise ValueError("publication has no integer model_version")
        state = payload.get("slow_state")
        if not isinstance(state, dict) or not state:
            raise ValueError("publication has no slow_state")
        return payload


def checkpoint_metadata(config_json: str, config_sha256: str) -> dict[str, Any]:
    # Parsing here also proves that only normalized JSON is embedded.
    json.loads(config_json)
    return {
        "checkpoint_schema_version": MODEL_ARTIFACT_SCHEMA.current,
        "tensor_precision": TENSOR_PRECISION,
        "config_json": config_json,
        "config_sha256": config_sha256,
    }
