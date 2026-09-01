"""Structured console events consumed by operators and observability adapters."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class ConsoleEventSink(Protocol):
    def emit(self, name: str, payload: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class NullEventSink:
    def emit(self, name: str, payload: Mapping[str, Any]) -> None:
        del name, payload

    def close(self) -> None:
        return


@dataclass(slots=True)
class JsonLineEventSink:
    output: Callable[[str], None] = print

    def emit(self, name: str, payload: Mapping[str, Any]) -> None:
        event = {
            "event": name,
            "timestamp_ns": time.time_ns(),
            "payload": dict(payload),
        }
        self.output(json.dumps(event, ensure_ascii=False, separators=(",", ":")))

    def close(self) -> None:
        return


class CompositeEventSink:
    def __init__(self, sinks: Iterable[ConsoleEventSink]) -> None:
        self.sinks = tuple(sinks)

    def emit(self, name: str, payload: Mapping[str, Any]) -> None:
        for sink in self.sinks:
            sink.emit(name, payload)

    def close(self) -> None:
        errors: list[Exception] = []
        for sink in reversed(self.sinks):
            try:
                sink.close()
            except Exception as error:  # pragma: no cover - defensive shutdown path
                errors.append(error)
        if errors:
            raise errors[0]
