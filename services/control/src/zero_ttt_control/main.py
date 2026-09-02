"""Control service entrypoint."""

from __future__ import annotations

import uvicorn

from zero_ttt_control.api import create_app
from zero_ttt_control.settings import ControlSettings
from zero_ttt_control.store import ControlStore


def main() -> None:
    settings = ControlSettings.from_environment()
    app = create_app(
        ControlStore(settings.database_path),
        profile_root=str(settings.profile_root),
        close_store=True,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
