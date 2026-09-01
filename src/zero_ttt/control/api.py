"""FastAPI surface for the container-internal training agent."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException

from zero_ttt.control.process import OPERATIONS, OperationConflict, WorkerController


def create_app(controller: WorkerController) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        with suppress(OperationConflict):
            controller.start("reconcile")
        yield
        await asyncio.to_thread(controller.shutdown)

    app = FastAPI(title="Zero-TTT Training Agent", version="1", lifespan=lifespan)

    @app.get("/api/v1/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/v1/status")
    def status() -> dict[str, object]:
        return controller.snapshot()

    @app.post("/api/v1/operations/{operation}", status_code=202)
    def start(operation: str) -> dict[str, object]:
        if operation not in OPERATIONS:
            raise HTTPException(status_code=404, detail="unknown console operation")
        try:
            return controller.start(operation)
        except OperationConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/operations/current/soft-stop", status_code=202)
    def soft_stop() -> dict[str, object]:
        try:
            return controller.soft_stop()
        except OperationConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return app
