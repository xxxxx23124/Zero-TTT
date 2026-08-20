ARG KATAGO_BUILD_IMAGE=pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel
ARG KATAGO_RUNTIME_IMAGE=nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

FROM ${KATAGO_BUILD_IMAGE} AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        g++ \
        ninja-build \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/KataGo/cpp /src/katago/cpp
RUN cmake \
        -S /src/katago/cpp \
        -B /src/katago/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_BACKEND=CUDA \
        -DNO_GIT_REVISION=1 \
        -DCUDNN_INCLUDE_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include \
        -DCUDNN_LIBRARY=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 \
    && cmake --build /src/katago/build --parallel

FROM ${KATAGO_RUNTIME_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libgomp1 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 katago

COPY --from=builder /src/katago/build/katago /usr/local/bin/katago
USER katago
WORKDIR /home/katago
ENTRYPOINT ["katago"]
