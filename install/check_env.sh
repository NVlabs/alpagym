#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pre-flight check for AlpaGym local-host runs. Validates that the host tools and
# headers documented in the README are present before `uv sync --all-packages`.
# Required checks that fail cause a non-zero exit; optional checks only warn.

set -uo pipefail

if [ -t 1 ]; then
  red=$'\033[31m'; yellow=$'\033[33m'; green=$'\033[32m'; bold=$'\033[1m'; reset=$'\033[0m'
else
  red=''; yellow=''; green=''; bold=''; reset=''
fi

errors=0
warnings=0

ok() { printf '  %s[ok]%s   %s\n' "$green" "$reset" "$1"; }

fail() {
  printf '  %s[fail]%s %s\n' "$red" "$reset" "$1"
  printf '         %sfix:%s %s\n' "$bold" "$reset" "$2"
  errors=$((errors + 1))
}

warn() {
  printf '  %s[warn]%s %s\n' "$yellow" "$reset" "$1"
  printf '         %shint:%s %s\n' "$bold" "$reset" "$2"
  warnings=$((warnings + 1))
}

printf '%sChecking AlpaGym local-host environment...%s\n\n' "$bold" "$reset"

# --- uv (required, >= 0.10.0) ------------------------------------------------
min_uv="0.10.0"
if ! command -v uv >/dev/null 2>&1; then
  fail "uv not found on PATH" \
    "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
else
  uv_version="$(uv --version 2>/dev/null | awk '{print $2}')"
  if [ -z "$uv_version" ]; then
    warn "could not parse 'uv --version' output" "Verify uv is functional: uv --version"
  elif [ "$(printf '%s\n%s\n' "$min_uv" "$uv_version" | sort -V | head -n1)" != "$min_uv" ]; then
    fail "uv $uv_version is older than the required $min_uv" \
      "Upgrade uv: python3 -m pip install --user --upgrade uv (or: uv self update)"
  else
    ok "uv $uv_version (>= $min_uv)"
  fi
fi

# --- git-lfs (required) ------------------------------------------------------
if ! command -v git-lfs >/dev/null 2>&1; then
  fail "git-lfs not found on PATH" \
    "sudo apt-get install -y git-lfs && git lfs install"
elif ! git config --get-regexp '^filter\.lfs\.' >/dev/null 2>&1; then
  fail "git-lfs is installed but not initialized" \
    "Run: git lfs install"
else
  ok "git-lfs installed and initialized"
fi

# --- redis-server (required, native dep used by Cosmos-RL) -------------------
if ! command -v redis-server >/dev/null 2>&1; then
  fail "redis-server not found on PATH" \
    "sudo apt-get install -y redis-server"
else
  ok "redis-server ($(redis-server --version | awk '{print $3}'))"
fi

# --- docker + docker compose (required for local AlpaSim Wizard runs) --------
if ! command -v docker >/dev/null 2>&1; then
  fail "docker not found on PATH" \
    "Install Docker Engine, then verify with: docker version"
elif ! docker compose version >/dev/null 2>&1; then
  fail "'docker compose' plugin not available" \
    "Install the Docker Compose v2 plugin (docker-compose-plugin)"
else
  ok "docker + docker compose available"
fi

# --- CUDA headers: cudnn.h and nccl.h (required by the model stack) ----------
header_dirs=(/usr/include /usr/include/x86_64-linux-gnu /usr/local/cuda/include)
find_header() {
  local name="$1" dir
  for dir in "${header_dirs[@]}"; do
    [ -f "$dir/$name" ] && { printf '%s/%s' "$dir" "$name"; return 0; }
  done
  return 1
}

if cudnn_path="$(find_header cudnn.h)"; then
  ok "cudnn.h found ($cudnn_path)"
else
  fail "cudnn.h not found" \
    "sudo apt-get install -y libcudnn9-dev-cuda-12 (see README for the cuda-keyring step)"
fi

if nccl_path="$(find_header nccl.h)"; then
  ok "nccl.h found ($nccl_path)"
else
  fail "nccl.h not found" \
    "sudo apt-get install -y libnccl-dev=2.26.2-1+cuda12.8 libnccl2=2.26.2-1+cuda12.8"
fi

# --- Optional checks (warn only) ---------------------------------------------
printf '\n%sOptional (policy/integration dependent):%s\n' "$bold" "$reset"

if GIT_TERMINAL_PROMPT=0 git ls-remote --quiet https://github.com/chenyx09/alpasim.git >/dev/null 2>&1; then
  ok "Pinned humanoid AlpaSim fork reachable"
else
  warn "could not authenticate to github.com/chenyx09/alpasim.git" \
    "Configure GitHub HTTPS credentials; uv sync and managed humanoid runs use this private fork"
fi

if ! command -v hf >/dev/null 2>&1 && command -v huggingface-cli >/dev/null 2>&1; then
  warn "only the legacy 'huggingface-cli' is installed (the 'hf' command is missing)" \
    "Upgrade huggingface_hub to get the 'hf' CLI: uv pip install -U 'huggingface_hub>=0.34'"
fi

if [ -n "${HF_TOKEN:-}" ] || (command -v hf >/dev/null 2>&1 && hf auth whoami >/dev/null 2>&1); then
  ok "Hugging Face authentication present"
else
  warn "no Hugging Face authentication detected" \
    "Run 'hf auth login' or export HF_TOKEN=... if your policy needs gated HF artifacts"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  ok "WANDB_API_KEY is set"
else
  warn "WANDB_API_KEY not set" \
    "Export WANDB_API_KEY in your shell profile to log runs to Weights & Biases"
fi

# --- Summary -----------------------------------------------------------------
printf '\n'
if [ "$errors" -gt 0 ]; then
  printf '%s%d required check(s) failed, %d warning(s).%s Fix the items above, then re-run.\n' \
    "$red" "$errors" "$warnings" "$reset"
  exit 1
fi

printf '%sAll required checks passed%s (%d warning(s)). You can run: uv sync --all-packages\n' \
  "$green" "$reset" "$warnings"
