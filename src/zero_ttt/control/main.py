"""Training-agent process entry point."""

from __future__ import annotations

import os

import uvicorn

from zero_ttt.control.api import create_app
from zero_ttt.control.process import WorkerController


def main() -> None:
    config = os.environ.get("ZERO_TTT_CONSOLE_CONFIG", "configs/console.toml")
    host = os.environ.get("ZERO_TTT_AGENT_HOST", "0.0.0.0")
    port = int(os.environ.get("ZERO_TTT_AGENT_PORT", "8090"))
    uvicorn.run(create_app(WorkerController(config)), host=host, port=port)


if __name__ == "__main__":
    main()
