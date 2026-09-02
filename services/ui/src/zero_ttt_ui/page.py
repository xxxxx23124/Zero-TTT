"""NiceGUI components and API actions; no workflow state machine."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from nicegui import ui

from zero_ttt_ui.client import ApiError, ControlApiClient
from zero_ttt_ui.view_model import latest_jobs, summary


class DashboardPage:
    def __init__(self, client: ControlApiClient, tensorboard_url: str) -> None:
        self.client = client
        self.tensorboard_url = tensorboard_url
        self.snapshot: dict[str, Any] = {}
        self.event_cursor = 0
        self.events: list[dict[str, Any]] = []
        self.labels: dict[str, Any] = {}
        # NiceGUI's dynamic element hierarchy does not expose stable public type aliases.
        self.profile: Any = None
        self.dataset: Any = None
        self.run: Any = None
        self.run_name: Any = None
        self.steps: Any = None
        self.games: Any = None
        self.job_id: Any = None
        self.job_table: Any = None
        self.event_log: Any = None

    def build(self) -> None:
        with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-4"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Zero-TTT 微服务训练中心").classes("text-2xl font-bold")
                ui.link("TensorBoard", self.tensorboard_url, new_tab=True)
            with ui.row().classes("w-full gap-3"):
                for key, title in (
                    ("active", "运行中"),
                    ("failed", "失败"),
                    ("datasets", "Dataset"),
                    ("publications", "Publication"),
                ):
                    with ui.card().classes("grow"):
                        ui.label(title).classes("text-sm text-gray-500")
                        self.labels[key] = ui.label("0").classes("text-2xl")
            self._actions()
            self.job_table = ui.table(
                columns=[
                    {"name": "kind", "label": "作业", "field": "kind"},
                    {"name": "state", "label": "状态", "field": "state"},
                    {"name": "attempt", "label": "尝试", "field": "attempt"},
                    {"name": "error", "label": "错误", "field": "error"},
                ],
                rows=[],
                row_key="job_id",
            ).classes("w-full")
            self.event_log = (
                ui.textarea(label="持久事件流").props("readonly autogrow").classes("w-full")
            )
        ui.timer(0.1, self.load_resources, once=True)
        ui.timer(1.0, self.refresh)
        ui.timer(1.0, self.refresh_events)

    def _actions(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("有限工作流").classes("font-semibold")
            with ui.row().classes("items-end gap-3"):
                ui.button("初始化数据", on_click=lambda: self.submit("data-bootstrap"))
                self.run_name = ui.input("新 Run 名称")
                self.profile = ui.select({}, label="Profile")
                self.dataset = ui.select({}, label="Cold snapshot")
                ui.button("创建 Run", on_click=self.create_run)
            with ui.row().classes("items-end gap-3"):
                self.run = ui.select({}, label="Run")
                self.steps = ui.number("训练 steps", value=1, min=1, precision=0)
                self.games = ui.number("自博弈棋局", value=64, min=1, precision=0)
                ui.button("Cold start", on_click=lambda: self.submit("cold-start"))
                ui.button("AlphaZero 单轮", on_click=lambda: self.submit("alpha-zero-round"))
            with ui.row().classes("items-end gap-3"):
                self.job_id = ui.input("Job ID")
                ui.button("取消作业", on_click=lambda: self.job_action("cancel"))
                ui.button("重试作业", on_click=lambda: self.job_action("retry"))

    async def load_resources(self) -> None:
        try:
            profiles, snapshot = await asyncio.gather(
                asyncio.to_thread(self.client.profiles),
                asyncio.to_thread(self.client.snapshot),
            )
        except ApiError as error:
            ui.notify(str(error), type="negative")
            return
        self.snapshot = snapshot
        profile_options = {item["profile_id"]: item["profile_id"] for item in profiles}
        dataset_options = {
            item["artifact_id"]: item["artifact_id"]
            for item in snapshot.get("artifacts", ())
            if item.get("kind") == "dataset-snapshot"
            and item.get("labels", {}).get("split") == "train"
        }
        run_options = {item["run_id"]: item["name"] for item in snapshot.get("runs", ())}
        self.profile.set_options(profile_options)
        self.dataset.set_options(dataset_options)
        self.run.set_options(run_options)

    async def create_run(self) -> None:
        try:
            await asyncio.to_thread(
                self.client.create_run,
                str(self.run_name.value or ""),
                str(self.profile.value or ""),
                str(self.dataset.value or ""),
            )
            ui.notify("Run 已创建并冻结 Profile 与 Dataset。", type="positive")
            await self.load_resources()
        except (ApiError, TypeError, ValueError) as error:
            ui.notify(str(error), type="negative", multi_line=True)

    async def submit(self, template: str) -> None:
        run_id = "" if template == "data-bootstrap" else str(self.run.value or "")
        parameters: dict[str, Any] = {}
        try:
            if template == "cold-start":
                parameters["steps"] = int(self.steps.value)
            elif template == "alpha-zero-round":
                parameters.update(steps=int(self.steps.value), games=int(self.games.value))
            await asyncio.to_thread(
                self.client.submit_workflow,
                template,
                run_id=run_id,
                parameters=parameters,
            )
            ui.notify(f"已提交 {template}", type="positive")
        except (ApiError, TypeError, ValueError) as error:
            ui.notify(str(error), type="negative", multi_line=True)

    async def job_action(self, action: str) -> None:
        job_id = str(self.job_id.value or "").strip()
        if not job_id:
            ui.notify("请输入 Job ID", type="warning")
            return
        operation = self.client.cancel if action == "cancel" else self.client.retry
        try:
            await asyncio.to_thread(operation, job_id)
            ui.notify(f"已提交 {action}", type="positive")
        except ApiError as error:
            ui.notify(str(error), type="negative", multi_line=True)

    async def refresh(self) -> None:
        try:
            self.snapshot = await asyncio.to_thread(self.client.snapshot)
        except ApiError:
            return
        for key, value in summary(self.snapshot).items():
            self.labels[key].set_text(value)
        self.job_table.rows = latest_jobs(self.snapshot)
        self.job_table.update()

    async def refresh_events(self) -> None:
        try:
            events = await asyncio.to_thread(self.client.events, self.event_cursor)
        except ApiError:
            return
        if not events:
            return
        self.event_cursor = int(events[-1]["sequence"])
        self.events.extend(events)
        self.events = self.events[-200:]
        self.event_log.value = "\n".join(
            json.dumps(item, ensure_ascii=False) for item in self.events
        )
        self.event_log.update()
