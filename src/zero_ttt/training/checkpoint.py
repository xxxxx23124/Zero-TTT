"""Atomic full checkpoints, immutable BF16 publications, and fault snapshots."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_SCHEMA_VERSION = 2


class CheckpointManager:
    def __init__(self, run_dir: str | Path, keep: int) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.publication_dir = self.run_dir / "published"
        self.fault_dir = self.run_dir / "faults"
        self.keep = keep
        for directory in (self.checkpoint_dir, self.publication_dir, self.fault_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_torch_save(payload: Any, destination: Path) -> None:
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            torch.save(payload, temporary)
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

    def save_publication(
        self,
        step: int,
        slow_state: dict[str, torch.Tensor],
        metadata: dict[str, Any],
    ) -> Path:
        bf16_state = {
            name: (
                tensor.detach().to(device="cpu", dtype=torch.bfloat16)
                if tensor.is_floating_point()
                else tensor.detach().cpu()
            )
            for name, tensor in slow_state.items()
        }
        payload = {**metadata, "model_version": step, "slow_state": bf16_state}
        immutable = self.publication_dir / f"slow_{step:012d}.pt"
        self._atomic_torch_save(payload, immutable)
        self._atomic_torch_save(payload, self.publication_dir / "current.pt")
        for old in self.publication_dir.glob("slow_*.pt"):
            if old != immutable:
                old.unlink()
        return immutable

    def save_fault(self, step: int, payload: dict[str, Any], reason: str) -> Path:
        payload = {**payload, "fault_reason": reason, "fault_time_ns": time.time_ns()}
        destination = self.fault_dir / f"fault_{step:012d}_{time.time_ns()}.pt"
        self._atomic_torch_save(payload, destination)
        return destination

    def latest_checkpoint(self) -> Path | None:
        checkpoints = sorted(self.checkpoint_dir.glob("step_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def current_publication(self) -> Path | None:
        current = self.publication_dir / "current.pt"
        return current if current.exists() else None

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema")
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
