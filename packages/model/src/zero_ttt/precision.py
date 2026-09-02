"""Fixed strict-FP32 policy for all neural-network execution and artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

TENSOR_DTYPE = torch.float32
TENSOR_PRECISION = "float32"


def configure_strict_fp32() -> None:
    """Select full FP32 kernels instead of reduced-precision CUDA fast paths."""

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def require_fp32_module(module: nn.Module, name: str) -> None:
    """Reject modules containing floating parameters or buffers outside FP32."""

    tensors = (*module.named_parameters(), *module.named_buffers())
    for tensor_name, tensor in tensors:
        if tensor.is_floating_point() and tensor.dtype != TENSOR_DTYPE:
            raise TypeError(f"{name}.{tensor_name} must use {TENSOR_PRECISION}")


def require_fp32_state(state: Mapping[str, Any], name: str) -> None:
    """Reject persisted model state containing non-FP32 floating tensors."""

    for tensor_name, tensor in state.items():
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.is_floating_point()
            and tensor.dtype != TENSOR_DTYPE
        ):
            raise TypeError(f"{name}.{tensor_name} must use {TENSOR_PRECISION}")
