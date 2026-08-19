ARG PYTORCH_IMAGE=pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    UV_CACHE_DIR=/cache/uv \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        htop \
        less \
        nano \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p \
        /cache/huggingface \
        /cache/pip \
        /cache/torch \
        /cache/torch_extensions \
        /cache/uv \
    && chmod -R a+rwx /cache

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --break-system-packages --editable ".[dev]"

CMD ["/bin/bash"]
