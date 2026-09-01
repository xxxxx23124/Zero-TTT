"""Training-agent process entry point."""

from __future__ import annotations

import os

import uvicorn

from zero_ttt.control.api import create_app
from zero_ttt.control.process import WorkerController
from zero_ttt.control.runs import RuntimeLayout


def main() -> None:
    host = os.environ.get("ZERO_TTT_AGENT_HOST", "0.0.0.0")
    port = int(os.environ.get("ZERO_TTT_AGENT_PORT", "8090"))
    controller = WorkerController(RuntimeLayout.from_environment())
    uvicorn.run(create_app(controller), host=host, port=port)


if __name__ == "__main__":
    main()
