from zero_ttt.console.runtime import RuntimeBudget, SoftStopSignals


def test_runtime_budget_uses_monotonic_deadline() -> None:
    now = [10.0]
    budget = RuntimeBudget(5.0, lambda: now[0])
    assert not budget.expired
    now[0] = 15.0
    assert budget.expired


def test_soft_stop_can_be_requested_cooperatively() -> None:
    stop = SoftStopSignals()
    assert not stop.requested
    stop.request()
    assert stop.requested
