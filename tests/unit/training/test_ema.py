from __future__ import annotations

import torch
from torch import nn

from zero_ttt.training.ema import ema_decay, update_slow_weights


def test_sample_based_ema_uses_equivalent_batched_decay() -> None:
    fast = nn.Linear(2, 2, bias=False)
    slow = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        fast.weight.fill_(1.0)
        slow.weight.zero_()
    decay = update_slow_weights(slow, fast, samples=32, half_life_samples=32)
    assert decay == ema_decay(32, 32) == 0.5
    assert torch.allclose(slow.weight, torch.full_like(slow.weight, 0.5))


def test_ema_synchronizes_named_buffers() -> None:
    fast = nn.BatchNorm1d(2)
    slow = nn.BatchNorm1d(2)
    with torch.no_grad():
        fast.running_mean.copy_(torch.tensor([2.0, 3.0]))
        fast.running_var.copy_(torch.tensor([4.0, 5.0]))
        fast.num_batches_tracked.fill_(7)
        slow.running_mean.zero_()
        slow.running_var.fill_(1.0)
        slow.num_batches_tracked.zero_()
    update_slow_weights(slow, fast, samples=1, half_life_samples=1)
    assert torch.equal(slow.running_mean, fast.running_mean)
    assert torch.equal(slow.running_var, fast.running_var)
    assert torch.equal(slow.num_batches_tracked, fast.num_batches_tracked)


def test_fp32_ema_retains_updates_smaller_than_bfloat16_resolution() -> None:
    fast = nn.Linear(1, 1, bias=False).float()
    slow = nn.Linear(1, 1, bias=False).float()
    with torch.no_grad():
        slow.weight.fill_(1.0)
        fast.weight.fill_(1.001)
    update_slow_weights(slow, fast, samples=1, half_life_samples=1)
    assert slow.weight.dtype == torch.float32
    assert 1.0 < slow.weight.item() < fast.weight.item()
    assert torch.tensor(slow.weight.item(), dtype=torch.bfloat16).item() == 1.0
