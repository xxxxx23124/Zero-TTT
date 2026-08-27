"""Monotonic runtime budgets and cooperative process-signal soft stopping."""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable


class RuntimeBudget:
    def __init__(
        self, seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._clock = clock
        self.deadline = clock() + seconds

    @property
    def expired(self) -> bool:
        return self._clock() >= self.deadline


class SoftStopSignals:
    def __init__(self) -> None:
        self._requested = threading.Event()
        self._previous: dict[signal.Signals, object] = {}

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    def request(self) -> None:
        self._requested.set()

    def __enter__(self) -> "SoftStopSignals":
        if threading.current_thread() is not threading.main_thread():
            return self

        def handler(_signum, _frame) -> None:
            self.request()

        for kind in (signal.SIGINT, signal.SIGTERM):
            self._previous[kind] = signal.getsignal(kind)
            signal.signal(kind, handler)
        return self

    def __exit__(self, *_: object) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for kind, previous in self._previous.items():
            signal.signal(kind, previous)
        self._previous.clear()
