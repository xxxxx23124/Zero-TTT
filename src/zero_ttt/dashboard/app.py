"""Local-only NiceGUI data preparation and training dashboard."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from nicegui import ui

from zero_ttt.dashboard.client import AgentClient, AgentClientError
from zero_ttt.dashboard.view_model import button_availability, overview

TERMINAL_JOB_STATES = {"SUCCEEDED", "FAILED", "INTERRUPTED"}


class DashboardPage:
    def __init__(self, client: AgentClient, tensorboard_url: str) -> None:
        self.client = client
        self.tensorboard_url = tensorboard_url
        self.selected_run_id = ""
        self.last_finished_operation_id = ""
        self.labels: dict[str, Any] = {}
        self.buttons: dict[str, Any] = {}
        self.profile_select = None
        self.snapshot_select = None
        self.run_select = None
        self.run_name = None
        self.runtime_hours = None
        self.data_view = None
        self.config_view = None
        self.metrics_view = None
        self.logs_view = None

    def build(self) -> None:
        ui.add_head_html(
            "<style>body{background:#f4f6f8}.q-card{border:1px solid #e5e7eb}</style>"
        )
        with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-5"):
            self._header()
            self._data_card()
            self._run_card()
            self._actions()
            self._overview_cards()
            with ui.grid(columns=2).classes("w-full gap-4"):
                self._details_card()
                self._metrics_card()
            self._logs_card()
        ui.timer(0.2, self.load_resources, once=True)
        ui.timer(0.4, self.refresh, once=True)
        ui.timer(2.0, self.refresh)

    def _header(self) -> None:
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("Zero-TTT 本地训练中心").classes("text-3xl font-semibold")
                ui.label("准备数据、创建训练任务并安全运行").classes("text-gray-500")
            ui.link("打开 TensorBoard", self.tensorboard_url, new_tab=True).classes(
                "text-blue-700 font-medium"
            )

    def _data_card(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("1. 数据准备").classes("text-xl font-medium")
            ui.label(
                "把 KataGo g170 ZIP 放入 Windows 数据目录后, 按顺序执行以下步骤。"
            ).classes("text-gray-500")
            with ui.row().classes("items-center gap-3"):
                for key, title, operation in (
                    ("scan", "扫描并校验", "scan"),
                    ("trial_import", "试导入 1000 局", "trial-import"),
                    ("full_import", "继续全量导入", "full-import"),
                    ("verify", "校验数据", "verify"),
                    ("snapshot", "创建训练 Snapshot", "snapshot-create"),
                ):
                    self.buttons[key] = ui.button(
                        title, on_click=lambda _event, op=operation: self.start_data(op)
                    )
            self.data_view = ui.textarea(value="正在读取数据目录…").props(
                "readonly autogrow"
            ).classes("w-full font-mono text-xs")

    def _run_card(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("2. 训练任务").classes("text-xl font-medium")
            with ui.grid(columns=4).classes("w-full gap-3"):
                self.run_name = ui.input("任务名称", placeholder="例如: 首次冷启动")
                self.profile_select = ui.select({}, label="训练方案")
                self.snapshot_select = ui.select({}, label="Cold snapshot")
                self.buttons["create_run"] = ui.button(
                    "创建任务", on_click=self.create_run
                ).classes("self-end")
            with ui.row().classes("w-full items-end gap-3"):
                self.run_select = ui.select({}, label="当前任务", on_change=self.change_run).classes(
                    "flex-1"
                )
                self.runtime_hours = ui.number(
                    "本次运行时长 (小时)", value=8.0, min=0.01, step=0.5
                ).classes("w-56")

    def _actions(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("3. 运行监控").classes("text-xl font-medium")
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
            ui.label("冻结配置摘要").classes("text-sm text-gray-500")
            self.config_view = ui.textarea(value="尚未校验").props("readonly autogrow").classes(
                "w-full font-mono text-xs"
            )

    def _metrics_card(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("最新指标 / 数据进度").classes("text-lg font-medium")
            self.metrics_view = ui.textarea(value="暂无指标").props("readonly autogrow").classes(
                "w-full font-mono text-xs"
            )

    def _logs_card(self) -> None:
        with ui.card().classes("w-full"):
            ui.label("最近事件与错误").classes("text-lg font-medium")
            self.logs_view = ui.textarea(value="").props("readonly autogrow").classes(
                "w-full font-mono text-xs"
            )

    async def load_resources(self) -> None:
        try:
            profiles, data, snapshots, runs = await asyncio.gather(
                asyncio.to_thread(self.client.profiles),
                asyncio.to_thread(self.client.data_status),
                asyncio.to_thread(self.client.snapshots),
                asyncio.to_thread(self.client.runs),
            )
        except AgentClientError as error:
            ui.notify(str(error), type="negative", multi_line=True)
            return
        assert self.profile_select is not None
        assert self.snapshot_select is not None
        assert self.run_select is not None
        profile_options = {item["profile_id"]: item["profile_id"] for item in profiles}
        cold = [
            item
            for item in snapshots
            if item.get("source_kind") == "external"
            and item.get("split") == "train"
            and int(item.get("games", 0)) > 0
            and int(item.get("positions", 0)) > 0
        ]
        snapshot_options = {
            item["snapshot_id"]: (
                f"{item['snapshot_id'][:12]}… | {item['games']} 局 | {item['positions']} positions"
            )
            for item in cold
        }
        run_options = {item["run_id"]: item["name"] for item in runs}
        self.profile_select.set_options(profile_options)
        self.snapshot_select.set_options(snapshot_options)
        self.run_select.set_options(run_options)
        if profile_options and self.profile_select.value not in profile_options:
            self.profile_select.value = next(iter(profile_options))
        if snapshot_options and self.snapshot_select.value not in snapshot_options:
            self.snapshot_select.value = next(iter(snapshot_options))
        if run_options and self.selected_run_id not in run_options:
            self.selected_run_id = next(iter(run_options))
            self.run_select.value = self.selected_run_id
        self._update_data_view(data)

    def _update_data_view(self, data: dict[str, Any]) -> None:
        assert self.data_view is not None
        gib = float(data.get("raw_bytes", 0)) / 1024**3
        self.data_view.value = (
            f"原始目录: {data.get('raw_directory', '-')}\n"
            f"原始文件: {data.get('raw_assets', 0)} 个 / {gib:.2f} GiB\n"
            f"Manifest: {data.get('manifest_assets', 0)} 个文件\n"
            f"导入: imported={data.get('imported_assets', 0)}, "
            f"partial={data.get('partial_assets', 0)}, games={data.get('games', 0)}, "
            f"positions={data.get('positions', 0)}\n"
            f"全量完成: {'是' if data.get('full_import_complete') else '否'} | "
            f"校验有效: {'是' if data.get('verification_current') else '否'}"
        )
        self.data_view.update()
        idle = not self._active_job(getattr(self, "latest_snapshot", {}))
        availability = {
            "scan": idle and int(data.get("raw_assets", 0)) > 0,
            "trial_import": idle and bool(data.get("manifest_exists")),
            "full_import": idle and int(data.get("games", 0)) > 0,
            "verify": idle and int(data.get("games", 0)) > 0,
            "snapshot": idle
            and bool(data.get("full_import_complete"))
            and bool(data.get("verification_current")),
        }
        for key, enabled in availability.items():
            self._set_enabled(key, enabled)

    @staticmethod
    def _active_job(snapshot: dict[str, Any]) -> bool:
        return str((snapshot.get("job") or {}).get("state", "IDLE")) in {
            "STARTING",
            "RUNNING",
            "STOP_REQUESTED",
        }

    def _set_enabled(self, key: str, enabled: bool) -> None:
        button = self.buttons.get(key)
        if button is None:
            return
        button.enable() if enabled else button.disable()

    async def start_data(self, operation: str) -> None:
        try:
            await asyncio.to_thread(self.client.start_data, operation)
            ui.notify(f"已提交数据操作: {operation}", type="positive")
        except AgentClientError as error:
            ui.notify(str(error), type="negative", multi_line=True)
        await self.refresh()

    async def create_run(self) -> None:
        assert self.run_name is not None
        assert self.profile_select is not None
        assert self.snapshot_select is not None
        try:
            created = await asyncio.to_thread(
                self.client.create_run,
                str(self.run_name.value or ""),
                str(self.profile_select.value or ""),
                str(self.snapshot_select.value or ""),
            )
            self.selected_run_id = str(created["run_id"])
            ui.notify("训练任务已创建, 配置与 snapshot 已冻结。", type="positive")
            await self.load_resources()
            await self._start_run("reconcile")
        except AgentClientError as error:
            ui.notify(str(error), type="negative", multi_line=True)

    async def change_run(self, event: Any) -> None:
        self.selected_run_id = str(event.value or "")
        await self.refresh()

    async def _start_run(self, operation: str) -> None:
        if not self.selected_run_id:
            ui.notify("请先创建或选择训练任务。", type="warning")
            return
        assert self.runtime_hours is not None
        try:
            await asyncio.to_thread(
                self.client.start_run,
                self.selected_run_id,
                operation,
                float(self.runtime_hours.value),
            )
            ui.notify(f"已提交训练操作: {operation}", type="positive")
        except (AgentClientError, TypeError, ValueError) as error:
            ui.notify(str(error), type="negative", multi_line=True)
        await self.refresh()

    async def start_train(self) -> None:
        await self._start_run("train")

    async def start_collect(self) -> None:
        await self._start_run("collect")

    async def start_warm_start(self) -> None:
        await self._start_run("warm-start")

    async def start_reconcile(self) -> None:
        await self._start_run("reconcile")

    async def soft_stop(self) -> None:
        try:
            await asyncio.to_thread(self.client.soft_stop)
            ui.notify("已请求安全暂停, 正在等待当前原子边界。", type="warning")
        except AgentClientError as error:
            ui.notify(str(error), type="negative", multi_line=True)
        await self.refresh()

    async def refresh(self) -> None:
        try:
            snapshot = await asyncio.to_thread(self.client.status, self.selected_run_id)
        except AgentClientError as error:
            if "job" in self.labels:
                self.labels["job"].set_text(str(error))
            return
        self.latest_snapshot = snapshot
        values = overview(snapshot)
        for key, value in values.items():
            if key in self.labels:
                self.labels[key].set_text(value)
        availability = button_availability(snapshot, bool(self.selected_run_id))
        for key, enabled in availability.items():
            self._set_enabled(key, enabled)
        active = self._active_job(snapshot)
        self._set_enabled("create_run", not active)
        for key in ("scan", "trial_import", "full_import", "verify", "snapshot"):
            if active:
                self._set_enabled(key, False)
        self._update_text_views(snapshot)
        job = snapshot.get("job") or {}
        operation_id = str(job.get("operation_id", ""))
        if (
            operation_id
            and operation_id != self.last_finished_operation_id
            and job.get("state") in TERMINAL_JOB_STATES
        ):
            self.last_finished_operation_id = operation_id
            await self.load_resources()

    def _update_text_views(self, snapshot: dict[str, Any]) -> None:
        assert self.config_view is not None
        assert self.metrics_view is not None
        assert self.logs_view is not None
        console = snapshot.get("console") or {}
        configuration = console.get("configuration") or {}
        job = snapshot.get("job") or {}
        metrics = (
            job.get("progress")
            or snapshot.get("latest_metrics")
            or snapshot.get("latest_collection")
            or snapshot.get("latest_data")
            or {}
        )
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
        title="Zero-TTT 本地训练中心",
        reload=False,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
