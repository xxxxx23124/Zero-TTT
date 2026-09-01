"""Subprocess entry point for safely stoppable data preparation jobs."""

from __future__ import annotations

import argparse
import json
import time

from zero_ttt.console.runtime import SoftStopSignals
from zero_ttt.control.data import DATA_OPERATIONS, DataService
from zero_ttt.control.runs import RuntimeLayout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zero-ttt-data-worker")
    parser.add_argument("operation", choices=sorted(DATA_OPERATIONS))
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)

    def emit(event: str, payload: dict[str, object]) -> None:
        print(
            json.dumps(
                {"event": event, "timestamp_ns": time.time_ns(), "payload": payload},
                ensure_ascii=False,
            ),
            flush=True,
        )

    with SoftStopSignals() as signals:
        completed = DataService(RuntimeLayout.from_environment()).run(
            arguments.operation,
            emit,
            lambda: signals.requested,
        )
    return 0 if completed else 130


if __name__ == "__main__":
    raise SystemExit(main())
