from __future__ import annotations

import json

import pytest

from zero_ttt.console.events import JsonLineEventSink
from zero_ttt.observability import TensorBoardEventSink


def test_json_line_event_sink_emits_one_machine_readable_record() -> None:
    lines: list[str] = []
    sink = JsonLineEventSink(lines.append)

    sink.emit("training_step", {"step": 3, "total_loss": 1.25})

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "training_step"
    assert event["payload"] == {"step": 3, "total_loss": 1.25}
    assert isinstance(event["timestamp_ns"], int)


def test_tensorboard_sink_records_training_and_collection_scalars(tmp_path) -> None:
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    sink = TensorBoardEventSink(tmp_path)
    sink.emit(
        "training_started",
        {
            "run_id": "run-a",
            "optimizer_step": 0,
            "config_json": "{}",
            "config_sha256": "a" * 64,
            "data_identity": {},
        },
    )
    sink.emit(
        "training_step",
        {
            "run_id": "run-a",
            "step": 1,
            "samples_seen": 4,
            "total_loss": 1.0,
            "policy_loss": 0.5,
            "value_loss": 0.25,
            "ownership_loss": 0.1,
            "score_loss": 0.05,
            "learning_rate": 0.001,
            "base_gradient_norm": 1.2,
            "hypernet_gradient_norm": None,
            "hyper_a_saturation": 0.0,
            "hyper_b_saturation": 0.0,
            "hyper_dynamic_rms": 0.1,
            "hyper_static_rms": 0.2,
            "step_seconds": 2.0,
            "ema_update_seconds": None,
            "positions_per_second": 2.0,
            "allocated_gib": None,
            "reserved_gib": None,
            "max_allocated_gib": None,
        },
    )
    sink.emit(
        "collection_round",
        {
            "run_id": "run-a",
            "round_number": 2,
            "collected_games": 4,
            "skipped_games": 0,
            "new_positions": 64,
            "simulations": 512,
            "wall_seconds": 8.0,
            "simulations_per_second": 64.0,
            "rules_seconds": 1.0,
            "batching": {"fill_ratio": 0.75, "cache_hit_ratio": 0.25},
        },
    )
    sink.close()

    accumulator = EventAccumulator(str(tmp_path / "tensorboard" / "run-a"))
    accumulator.Reload()
    assert accumulator.Scalars("loss/total_loss")[0].value == pytest.approx(1.0)
    assert accumulator.Scalars("progress/samples_seen")[0].step == 1
    assert accumulator.Scalars("selfplay/simulations_per_second")[0].value == pytest.approx(
        64.0
    )
    assert accumulator.Scalars("selfplay/batching_fill_ratio")[0].step == 2
