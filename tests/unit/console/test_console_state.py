from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from zero_ttt.console.state import (
    ConsoleLock,
    ConsoleState,
    Operation,
    StateStore,
    TrainingPhase,
    migration_record,
    transition,
)


def test_state_round_trip_and_atomic_migration_history(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "console" / "state.json")
    state = transition(ConsoleState(), Operation.WARM_STARTING)
    state = dataclasses.replace(
        state,
        phase=TrainingPhase.MIXTURE,
        next_collection_round=3,
        migrations=(migration_record("warm_start", "cold", "selfplay", "m" * 64),),
    )
    store.save(state)
    assert store.load() == state
    assert not tuple((tmp_path / "console").glob(".*.tmp"))


def test_state_machine_rejects_unsafe_direct_transitions() -> None:
    with pytest.raises(ValueError, match="invalid console transition"):
        transition(ConsoleState(), Operation.SOFT_STOPPING)
    collecting = transition(ConsoleState(), Operation.COLLECTING)
    stopping = transition(collecting, Operation.SOFT_STOPPING)
    assert transition(stopping, Operation.READY).operation is Operation.READY


def test_console_lock_rejects_a_second_owner(tmp_path: Path) -> None:
    path = tmp_path / "console.lock"
    with ConsoleLock(path):
        with pytest.raises(RuntimeError, match="already owns"):
            with ConsoleLock(path):
                pass
    with ConsoleLock(path):
        pass
