"""TensorBoard adapter for structured console events."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class TensorBoardEventSink:
    """Lazily open one writer per durable learner run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.writer = None
        self.run_id = ""

    def _open(self, run_id: str, purge_step: int | None = None) -> None:
        if self.writer is not None and self.run_id == run_id:
            return
        self.close()
        from torch.utils.tensorboard import SummaryWriter

        log_dir = self.run_dir / "tensorboard" / run_id
        self.writer = SummaryWriter(log_dir=str(log_dir), purge_step=purge_step)
        self.run_id = run_id

    def _training_started(self, payload: Mapping[str, Any]) -> None:
        initial_step = int(payload["optimizer_step"])
        self._open(str(payload["run_id"]), purge_step=initial_step + 1)
        assert self.writer is not None
        self.writer.add_text("run/config", str(payload["config_json"]), initial_step)
        self.writer.add_text("run/config_sha256", str(payload["config_sha256"]), initial_step)
        self.writer.add_text("run/data_identity", str(payload["data_identity"]), initial_step)

    def _training_step(self, payload: Mapping[str, Any]) -> None:
        self._open(str(payload["run_id"]))
        assert self.writer is not None
        step = int(payload["step"])
        groups = {
            "loss": ("total_loss", "policy_loss", "value_loss", "ownership_loss", "score_loss"),
            "optimization": ("learning_rate", "base_gradient_norm", "hypernet_gradient_norm"),
            "hypernet": (
                "hyper_a_saturation",
                "hyper_b_saturation",
                "hyper_dynamic_rms",
                "hyper_static_rms",
            ),
            "timing": ("step_seconds", "ema_update_seconds"),
            "throughput": ("positions_per_second",),
            "cuda": ("allocated_gib", "reserved_gib", "max_allocated_gib"),
            "progress": ("samples_seen",),
        }
        for group, names in groups.items():
            for name in names:
                value = payload.get(name)
                if value is not None:
                    self.writer.add_scalar(f"{group}/{name}", float(value), step)
        self.writer.flush()

    def _collection_round(self, payload: Mapping[str, Any]) -> None:
        self._open(str(payload["run_id"]))
        assert self.writer is not None
        round_number = int(payload["round_number"])
        batching = payload["batching"]
        scalar_names = (
            "collected_games",
            "skipped_games",
            "new_positions",
            "simulations",
            "wall_seconds",
            "simulations_per_second",
            "rules_seconds",
        )
        for name in scalar_names:
            self.writer.add_scalar(f"selfplay/{name}", float(payload[name]), round_number)
        for name, value in dict(batching).items():
            self.writer.add_scalar(f"selfplay/batching_{name}", float(value), round_number)
        self.writer.flush()

    def emit(self, name: str, payload: Mapping[str, Any]) -> None:
        if name == "training_started":
            self._training_started(payload)
        elif name == "training_step":
            self._training_step(payload)
        elif name == "collection_round":
            self._collection_round(payload)

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.flush()
        self.writer.close()
        self.writer = None
        self.run_id = ""
