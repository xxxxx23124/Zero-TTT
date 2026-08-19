"""Sample-count based exponential moving averages for slow weights."""

from __future__ import annotations

import math

import torch
from torch import nn


def ema_decay(samples: int, half_life_samples: int) -> float:
    if samples < 0 or half_life_samples <= 0:
        raise ValueError("EMA samples must be non-negative and half-life positive")
    return math.exp(math.log(0.5) * samples / half_life_samples)


@torch.no_grad()
def update_slow_weights(
    slow: nn.Module,
    fast: nn.Module,
    samples: int,
    half_life_samples: int,
) -> float:
    decay = ema_decay(samples, half_life_samples)
    slow_parameters = dict(slow.named_parameters())
    fast_parameters = dict(fast.named_parameters())
    slow_buffers = dict(slow.named_buffers())
    fast_buffers = dict(fast.named_buffers())
    if slow_parameters.keys() != fast_parameters.keys() or slow_buffers.keys() != fast_buffers.keys():
        raise ValueError("fast and slow model structures do not match")
    for name, slow_tensor in slow_parameters.items():
        fast_tensor = fast_parameters[name].detach().to(
            device=slow_tensor.device,
            dtype=slow_tensor.dtype,
        )
        slow_tensor.mul_(decay).add_(fast_tensor, alpha=1.0 - decay)
    # RoPE frequencies and other persistent buffers are state, not learned weights.
    for name, slow_tensor in slow_buffers.items():
        fast_tensor = fast_buffers[name].detach().to(
            device=slow_tensor.device,
            dtype=slow_tensor.dtype,
        )
        slow_tensor.copy_(fast_tensor)
    return decay
