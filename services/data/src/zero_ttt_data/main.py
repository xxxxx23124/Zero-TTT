"""Data worker entrypoint."""

from __future__ import annotations

from zero_ttt_contracts import WorkerCapability
from zero_ttt_worker import ControlClient, WorkerRunner

from zero_ttt_data.handlers import DataJobHandlers
from zero_ttt_data.settings import DataSettings


def main() -> None:
    settings = DataSettings.from_environment()
    runner = WorkerRunner(
        ControlClient(settings.control_url),
        worker_id=settings.worker_id,
        capability=WorkerCapability.DATA,
        version="0.1.0",
        handlers=DataJobHandlers(settings).mapping(),
    )
    runner.run_forever()


if __name__ == "__main__":
    main()
