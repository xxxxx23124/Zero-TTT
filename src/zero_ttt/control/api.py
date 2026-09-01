"""Version-2 FastAPI surface for local data and training control."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from zero_ttt.control.data import DATA_OPERATIONS
from zero_ttt.control.process import RUN_OPERATIONS, OperationConflict, WorkerController


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    profile_id: str
    cold_snapshot_id: str


class RunJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_runtime_hours: float = Field(gt=0)


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, KeyError):
        return HTTPException(status_code=404, detail=str(error).strip("'"))
    if isinstance(error, ValueError):
        return HTTPException(status_code=422, detail=str(error))
    return HTTPException(status_code=409, detail=str(error))


def _register_read_routes(app: FastAPI, controller: WorkerController) -> None:
    @app.get("/api/v2/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/v2/status")
    def status(run_id: str | None = Query(default=None)) -> dict[str, object]:
        return controller.snapshot(run_id)

    @app.get("/api/v2/profiles")
    def profiles() -> dict[str, object]:
        try:
            return {"profiles": controller.list_profiles()}
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/v2/data")
    def data_status() -> dict[str, object]:
        try:
            return controller.data_status()
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/v2/snapshots")
    def snapshots() -> dict[str, object]:
        try:
            return {"snapshots": controller.list_snapshots()}
        except Exception as error:
            raise _http_error(error) from error

    @app.get("/api/v2/runs")
    def runs() -> dict[str, object]:
        try:
            return {"runs": controller.list_runs()}
        except Exception as error:
            raise _http_error(error) from error


def _register_data_routes(app: FastAPI, controller: WorkerController) -> None:
    @app.post("/api/v2/data/jobs/{operation}", status_code=202)
    def start_data_job(operation: str) -> dict[str, object]:
        if operation not in DATA_OPERATIONS:
            raise HTTPException(status_code=404, detail="unknown data operation")
        try:
            return controller.start_data(operation)
        except (OperationConflict, FileNotFoundError, RuntimeError, ValueError) as error:
            raise _http_error(error) from error


def _register_run_routes(app: FastAPI, controller: WorkerController) -> None:
    @app.post("/api/v2/runs", status_code=201)
    def create_run(request: CreateRunRequest) -> dict[str, object]:
        try:
            return controller.create_run(
                request.name, request.profile_id, request.cold_snapshot_id
            )
        except (OperationConflict, FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            raise _http_error(error) from error

    @app.post("/api/v2/runs/{run_id}/jobs/{operation}", status_code=202)
    def start_run_job(
        run_id: str, operation: str, request: RunJobRequest
    ) -> dict[str, object]:
        if operation not in RUN_OPERATIONS:
            raise HTTPException(status_code=404, detail="unknown run operation")
        try:
            return controller.start_run(run_id, operation, request.max_runtime_hours)
        except (OperationConflict, FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            raise _http_error(error) from error

    @app.post("/api/v2/jobs/current/soft-stop", status_code=202)
    def soft_stop() -> dict[str, object]:
        try:
            return controller.soft_stop()
        except OperationConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error


def create_app(controller: WorkerController) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await asyncio.to_thread(controller.shutdown)

    app = FastAPI(title="Zero-TTT Local Training Agent", version="2", lifespan=lifespan)
    _register_read_routes(app, controller)
    _register_data_routes(app, controller)
    _register_run_routes(app, controller)
    return app
