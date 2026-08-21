# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ARG CI_CUDA_VERSION=12.8.1
ARG CI_UBUNTU_VERSION=22.04
ARG CI_BASE_IMAGE=nvcr.io/nvidia/cuda:${CI_CUDA_VERSION}-cudnn-devel-ubuntu${CI_UBUNTU_VERSION}

FROM ${CI_BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG NCCL_VERSION=2.26.2-1+cuda12.8
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ARG UV_CONCURRENT_BUILDS=1
ARG MAX_JOBS=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CONCURRENT_BUILDS=${UV_CONCURRENT_BUILDS} \
    MAX_JOBS=${MAX_JOBS} \
    NVTE_BUILD_MAX_JOBS=${MAX_JOBS} \
    NVTE_BUILD_THREADS_PER_JOB=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/usr/bin:$PATH"

# `uv sync` builds alpasim-grpc from the LFS-tracked alpasim repo, here and in the
# runtime container. git-lfs resolves the checkout's LFS filter; --skip-smudge keeps
# blobs as pointers since the grpc protos are plain text and need no LFS content.
RUN apt-get update && apt-get install -y --allow-change-held-packages --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    git-lfs \
    libnccl-dev=${NCCL_VERSION} \
    libnccl2=${NCCL_VERSION} \
    ninja-build \
    redis-server \
    && git lfs install --system --skip-smudge \
    && rm -rf /var/lib/apt/lists/*

RUN test -x "$(git --exec-path)/git-remote-https"

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

WORKDIR /alpagym

COPY . /alpagym

# Install the complete locked workspace into the image. Cluster launches use
# this immutable environment and never resolve, copy, or compile packages.
RUN --mount=type=secret,id=netrc,target=/root/.netrc --mount=type=cache,target=/root/.cache/uv,id=alpagym-uv-cache,sharing=locked uv sync --project /alpagym --link-mode=copy --frozen --all-packages --no-editable \
    && uv pip install --link-mode=copy coverage

RUN test -n "$(find /usr -name cudnn.h -print -quit)" \
    && test -n "$(find /usr -name nccl.h -print -quit)"
