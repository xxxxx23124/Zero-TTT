"""UI service entrypoint."""

from __future__ import annotations

import os

from nicegui import ui

from zero_ttt_ui.client import ControlApiClient
from zero_ttt_ui.page import DashboardPage


def main() -> None:
    control_url = os.environ.get("ZERO_TTT_CONTROL_URL", "http://control:8090")
    tensorboard_url = os.environ.get("ZERO_TTT_TENSORBOARD_URL", "http://127.0.0.1:6006")

    @ui.page("/")
    def index() -> None:
        DashboardPage(ControlApiClient(control_url), tensorboard_url).build()

    ui.run(
        host=os.environ.get("ZERO_TTT_UI_HOST", "0.0.0.0"),
        port=int(os.environ.get("ZERO_TTT_UI_PORT", "8080")),
        title="Zero-TTT",
        reload=False,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
