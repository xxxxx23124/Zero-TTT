"""Verify that the Zero-TTT development container can use the host GPU."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import torch


EXPECTED_TORCH_PREFIX = "2.13.0"
EXPECTED_CUDA = "13.2"
EXPECTED_COMPUTE_CAPABILITY = (8, 9)


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def main() -> None:
    workspace = Path("/workspace")
    assert workspace.is_dir(), "/workspace is missing"
    assert (workspace / "README.md").is_file(), "project bind mount is missing"
    assert shutil.which("nvcc"), "nvcc is missing; use the devel image"

    print(command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]))
    print(command_output(["nvcc", "--version"]).splitlines()[-1])
    print(f"Python:  {shutil.which('python')}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")
    print(f"cuDNN:   {torch.backends.cudnn.version()}")

    assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__
    assert torch.version.cuda == EXPECTED_CUDA, torch.version.cuda
    assert torch.cuda.is_available(), "torch.cuda.is_available() returned False"
    assert torch.backends.cudnn.is_available(), "cuDNN is unavailable"

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    assert capability == EXPECTED_COMPUTE_CAPABILITY, capability

    generator = torch.Generator(device=device).manual_seed(0)
    left = torch.randn((1024, 1024), device=device, dtype=torch.float16, generator=generator)
    right = torch.randn((1024, 1024), device=device, dtype=torch.float16, generator=generator)
    result = left @ right
    torch.cuda.synchronize(device)
    checksum = result.float().mean().item()
    assert math.isfinite(checksum), checksum

    print(f"GPU:     {torch.cuda.get_device_name(device)} (SM {capability[0]}.{capability[1]})")
    print(f"GPU matrix multiplication passed; checksum={checksum:.6f}")
    print("Zero-TTT Docker GPU smoke test passed.")


if __name__ == "__main__":
    main()
