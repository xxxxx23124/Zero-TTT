"""Local-only NiceGUI training dashboard."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from nicegui import ui

from zero_ttt.dashboard.client import AgentClient, AgentClientError
from zero_ttt.dashboard.view_model import button_availability, overview


class DashboardPage:
    def __init__(self, client: AgentClient, tensorboard_url: str) -> None:
        self.client = client
        self.tensorboard_url = tensorboard_url
        self.labels: dict[str, Any] = {}
        self.buttons: dict[str, Any] = {}
        self.config_view = None
        self.metrics_view = None
        self.logs_view = None

    def build(self) -> None:
        ui.add_head_html(
            "<style>body{background:#f4f6f8}.q-card{border:1px solid #e5e7eb}</style>"
        )
        with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-5"):
            self._header()
            self._actions()
            self._overview_cards()
            with ui.grid(columns=2).classes("w-full gap-4"):
                self._details_card()
                self._metrics_card()
            self._logs_card()
        ui.timer(0.2, self.refresh, once=True)
        ui.timer(2.0, self.refresh)

    def _header(self) -> None:
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("Zero-TTT 训练控制台").classes("text-3xl font-semibold")
                ui.label("冷启动、MCTS 自博弈与 mixture 训练").classes("text-gray-500")
            ui.link("打开 TensorBoard", self.tensorboard_url, new_tab=True).classes(
                "text-blue-700 font-medium"
            )

    def _actions(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("操作").classes("text-lg font-medium")
            with ui.row().classes("items-center gap-3"):
                self.buttons["train"] = ui.button("开始 / 继续训练", on_click=self.start_train)
                self.buttons["collect"] = ui.button("开始 MCTS 收集", on_click=self.start_collect)
                self.buttons["warm_start"] = ui.button(
                    "进入 mixture", on_click=self.start_warm_start
                )
                self.buttons["soft_stop"] = ui.button(
                    "安全暂停", on_click=self.soft_stop, color="negative"
                )
                self.buttons["reconcile"] = ui.button(
                    "重新校验", on_click=self.start_reconcile, color="secondary"
                )

    def _overview_cards(self) -> None:
        fields = (
            ("phase", "训练阶段"),
            ("operation", "控制台状态"),
            ("job", "活动任务"),
            ("step", "Optimizer step"),
            ("samples", "Samples seen"),
            ("artifacts", "产物一致性"),
        )
        with ui.grid(columns=3).classes("w-full gap-4"):
            for key, title in fields:
                with ui.card().classes("w-full"):
                    ui.label(title).classes("text-sm text-gray-500")
                    self.labels[key] = ui.label("-").classes("text-xl font-medium break-all")

    def _details_card(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("运行与数据").classes("text-lg font-medium")
            for key, title in (
                ("run", "Run"),
                ("selfplay", "自博弈"),
                ("pending", "待纳入 mixture"),
                ("outcome", "上一轮结果"),
            ):
                with ui.row().classes("w-full items-start"):
                    ui.label(title).classes("w-32 text-gray-500")
                    self.labels[key] = ui.label("-").classes("flex-1 break-all")
            ui.separator()
            ui.label("只读配置摘要").classes("text-sm text-gray-500")
            self.config_view = ui.textarea(value="尚未校验").props("readonly autogrow").classes(
                "w-full font-mono text-xs"
            )

    def _metrics_card(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("最新指标").classes("text-lg font-medium")
            self.metrics_view = ui.textarea(value="暂无指标").props("readonly autogrow").classes(
                "w-full font-mono text-xs"
            )

    def _logs_card(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("最近事件与错误").classes("text-lg font-medium")
            self.logs_view = ui.textarea(value="").props("readonly autogrow").classes(
                "w-full font-mono text-xs"
            )

    async def _start(self, operation: str) -> None:
        try:
            await asyncio.to_thread(self.client.start, operation)
            ui.notify(f"已提交操作: {operation}", type="positive")
        except AgentClientError as error:
            ui.notify(str(error), type="negative", multi_line=True)
        await self.refresh()

    async def start_train(self) -> None:
        await self._start("train")

    async def start_collect(self) -> None:
        await self._start("collect")

    async def start_warm_start(self) -> None:
        await self._start("warm-start")

    async def start_reconcile(self) -> None:
        await self._start("reconcile")

    async def soft_stop(self) -> None:
        try:
            await asyncio.to_thread(self.client.soft_stop)
            ui.notify("已请求安全暂停; 正在等待当前边界完成。", type="warning")
        except AgentClientError as error:
            ui.notify(str(error), type="negative", multi_line=True)
        await self.refresh()

    async def refresh(self) -> None:
        try:
            snapshot = await asyncio.to_thread(self.client.status)
        except AgentClientError as error:
            if "job" in self.labels:
                self.labels["job"].set_text(f"训练代理不可用: {error}")
            return
        values = overview(snapshot)
        for key, value in values.items():
            if key in self.labels:
                self.labels[key].set_text(value)
        availability = button_availability(snapshot)
        for key, enabled in availability.items():
            if key in self.buttons:
                if enabled:
                    self.buttons[key].enable()
                else:
                    self.buttons[key].disable()
        self._update_text_views(snapshot)

    def _update_text_views(self, snapshot: dict[str, Any]) -> None:
        assert self.config_view is not None
        assert self.metrics_view is not None
        assert self.logs_view is not None
        console = snapshot.get("console") or {}
        configuration = console.get("configuration") or {}
        metrics = snapshot.get("latest_metrics") or snapshot.get("latest_collection") or {}
        logs = snapshot.get("logs") or ()
        self.config_view.value = json.dumps(configuration, ensure_ascii=False, indent=2)
        self.metrics_view.value = json.dumps(metrics, ensure_ascii=False, indent=2)
        self.logs_view.value = "\n".join(str(line) for line in logs[-100:])
        self.config_view.update()
        self.metrics_view.update()
        self.logs_view.update()


def main() -> None:
    agent_url = os.environ.get("ZERO_TTT_AGENT_URL", "http://training-agent:8090")
    tensorboard_url = os.environ.get("ZERO_TTT_TENSORBOARD_URL", "http://127.0.0.1:6006")

    @ui.page("/")
    def index() -> None:
        DashboardPage(AgentClient(agent_url), tensorboard_url).build()

    ui.run(
        host=os.environ.get("ZERO_TTT_UI_HOST", "0.0.0.0"),
        port=int(os.environ.get("ZERO_TTT_UI_PORT", "8080")),
        title="Zero-TTT 训练控制台",
        reload=False,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
