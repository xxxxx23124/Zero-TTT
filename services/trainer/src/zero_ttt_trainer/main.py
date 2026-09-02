"""Trainer worker entrypoint."""

from __future__ import annotations

from zero_ttt_contracts import WorkerCapability
from zero_ttt_worker import ControlClient, WorkerRunner

from zero_ttt_trainer.jobs import TrainingJobHandler
from zero_ttt_trainer.settings import TrainerSettings


def main() -> None:
    settings = TrainerSettings.from_environment()
    runner = WorkerRunner(
        ControlClient(settings.control_url),
        worker_id=settings.worker_id,
        capability=WorkerCapability.TRAINER,
        version="0.1.0",
        handlers=TrainingJobHandler(settings).mapping(),
        lease_seconds=120,
    )
    runner.run_forever()


if __name__ == "__main__":
    main()
