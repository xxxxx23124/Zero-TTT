from zero_ttt.dashboard.view_model import button_availability


def test_buttons_follow_console_readiness_and_phase() -> None:
    snapshot = {
        "job": {"operation": "reconcile", "state": "SUCCEEDED"},
        "console": {
            "validated": True,
            "operation": "READY",
            "phase": "COLD_START",
            "checkpoint_path": "/runs/checkpoint.pt",
            "publication_path": "/runs/model.pt",
            "selfplay": {"games": 16},
        },
    }

    availability = button_availability(snapshot)

    assert availability == {
        "reconcile": True,
        "train": True,
        "collect": True,
        "warm_start": True,
        "soft_stop": False,
    }


def test_only_soft_stop_is_available_while_training() -> None:
    snapshot = {
        "job": {"operation": "train", "state": "RUNNING"},
        "console": {"validated": True, "operation": "TRAINING"},
    }

    availability = button_availability(snapshot)

    assert availability["soft_stop"] is True
    assert not any(availability[name] for name in ("reconcile", "train", "collect", "warm_start"))
