"""Atomic full checkpoints, immutable BF16 publications, and fault snapshots."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_SCHEMA_VERSION = 5


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
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(payload, temporary)
            with temporary.open("rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def save_full(self, step: int, payload: dict[str, Any]) -> Path:
        destination = self.checkpoint_dir / f"step_{step:012d}.pt"
        self._atomic_torch_save(payload, destination)
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        for old in checkpoints[:-self.keep]:
            old.unlink()
        return destination

    @staticmethod
    def _atomic_json_save(payload: dict[str, Any], destination: Path) -> None:
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

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
        bf16_state = {
            name: (
                tensor.detach().to(device="cpu", dtype=torch.bfloat16)
                if tensor.is_floating_point()
                else tensor.detach().cpu()
            )
            for name, tensor in slow_state.items()
        }
        payload = {
            **metadata,
            "model_version": step,
            "run_id": run_id,
            "samples_seen": samples_seen,
            "slow_state": bf16_state,
        }
        run_directory = self.publication_dir / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        immutable = run_directory / f"step_{step:012d}"
        if immutable.exists():
            metadata_path = immutable / "metadata.json"
            model_path = immutable / "model.pt"
            if not metadata_path.is_file() or not model_path.is_file():
                raise FileExistsError(f"incomplete publication already exists: {run_id}:{step}")
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                existing.get("run_id") != run_id
                or existing.get("optimizer_step") != step
                or existing.get("samples_seen") != samples_seen
                or existing.get("sha256") != self._sha256(model_path)
            ):
                raise FileExistsError(f"conflicting publication already exists: {run_id}:{step}")
            relative_model = model_path.relative_to(self.run_dir).as_posix()
            self._atomic_json_save(
                {
                    "run_id": run_id,
                    "optimizer_step": step,
                    "samples_seen": samples_seen,
                    "sha256": existing["sha256"],
                    "model_path": relative_model,
                },
                self.publication_dir / "current.json",
            )
            return model_path
        temporary = run_directory / f".step_{step:012d}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            model_path = temporary / "model.pt"
            self._atomic_torch_save(payload, model_path)
            digest = self._sha256(model_path)
            publication_metadata = {
                "run_id": run_id,
                "optimizer_step": step,
                "samples_seen": samples_seen,
                "sha256": digest,
                "model_file": "model.pt",
            }
            self._atomic_json_save(publication_metadata, temporary / "metadata.json")
            os.replace(temporary, immutable)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        relative_model = (immutable / "model.pt").relative_to(self.run_dir).as_posix()
        self._atomic_json_save(
            {
                "run_id": run_id,
                "optimizer_step": step,
                "samples_seen": samples_seen,
                "sha256": digest,
                "model_path": relative_model,
            },
            self.publication_dir / "current.json",
        )
        publications = sorted(
            path for path in run_directory.glob("step_*") if path.is_dir()
        )
        for old in publications[: -self.publication_keep]:
            shutil.rmtree(old)
        return immutable / "model.pt"

    def save_fault(self, step: int, payload: dict[str, Any], reason: str) -> Path:
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
            path = self.run_dir / payload["model_path"]
            if not path.is_file() or self._sha256(path) != payload["sha256"]:
                raise ValueError("current publication pointer is invalid")
            return path
        legacy = self.publication_dir / "current.pt"
        return legacy if legacy.exists() else None

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        actual_schema = payload.get("checkpoint_schema_version")
        if actual_schema != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported checkpoint schema v{actual_schema}; "
                f"expected v{CHECKPOINT_SCHEMA_VERSION}; migration is not supported"
            )
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
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_json": config_json,
        "config_sha256": config_sha256,
    }
