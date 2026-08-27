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
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CXX=g++

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

COPY third_party/open_spiel ./third_party/open_spiel
RUN set -eux; \
    fetch_repo() { \
        destination="$1"; url="$2"; revision="$3"; \
        git clone --filter=blob:none --no-checkout "$url" "$destination"; \
        git -C "$destination" checkout --detach "$revision"; \
        rm -rf "$destination/.git"; \
    }; \
    fetch_repo third_party/open_spiel/pybind11 https://github.com/pybind/pybind11.git e86205cb2ba070755d582856430c0f83c0af6694; \
    fetch_repo third_party/open_spiel/open_spiel/games/bridge/double_dummy_solver https://github.com/jblespiau/dds.git 091ea94358a4016d4fb6069dea5c452cdc98d0bd; \
    fetch_repo third_party/open_spiel/open_spiel/abseil-cpp https://github.com/abseil/abseil-cpp.git d38452e1ee03523a208362186fd42248ff2609f6; \
    fetch_repo third_party/open_spiel/open_spiel/json https://github.com/nlohmann/json.git 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03; \
    fetch_repo third_party/open_spiel/open_spiel/pybind11_json https://github.com/pybind/pybind11_json.git d0bf434be9d287d73a963ff28745542daf02c08f; \
    fetch_repo third_party/open_spiel/open_spiel/pybind11_abseil https://github.com/pybind/pybind11_abseil.git 73992b5
RUN python -m pip install --break-system-packages --no-build-isolation ./third_party/open_spiel \
    && python -c "import pyspiel; from open_spiel.python.algorithms import mcts; print(pyspiel.__doc__ is not None, mcts.MCTSBot.__name__)"
ENV ZERO_TTT_OPEN_SPIEL_REVISION=112b77704631fc2ce7ad8e4581f6ca09798ce15a

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --break-system-packages --editable ".[dev]"

CMD ["/bin/bash"]
