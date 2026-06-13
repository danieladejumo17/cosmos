#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Set up the environment to run transformers_cosmos1_reasoning.py (Cosmos-Reason1
# anomaly reasoning with the Hugging Face transformers library).
#
# Installs into the repo's .venv (shared with setup_reasoner.sh). torch,
# transformers, and opencv are usually already present from that setup; this
# script adds the transformers video-inference deps (qwen-vl-utils, accelerate)
# and fills in torch/transformers/opencv if the venv is fresh.
#
# Usage:
#   ./setup_cosmos1.sh
#   source .venv/bin/activate
#   python transformers_cosmos1_reasoning.py --dataset generated_vids --exp_name smoke
#
# The torch backend is auto-selected from the host CUDA driver; override with e.g.
#   TORCH_BACKEND=cu128 ./setup_cosmos1.sh

set -euo pipefail

find_repo_root() {
  local dir
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [[ "${dir}" != "/" ]]; do
    if [[ -f "${dir}/README.md" && -d "${dir}/cookbooks" ]]; then
      printf '%s\n' "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  printf '%s\n' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
}

COSMOS_ROOT="$(find_repo_root)"

# Pick the torch backend from the host CUDA driver major version (override via env).
detect_cuda_major() {
  local ver
  ver="$(nvidia-smi 2>/dev/null | grep -oiE 'CUDA Version: [0-9]+' | grep -oE '[0-9]+' | head -1)"
  printf '%s\n' "${ver:-13}"
}
CUDA_MAJOR="$(detect_cuda_major)"
[[ "${CUDA_MAJOR}" == "13" ]] && DEFAULT_BACKEND="cu130" || DEFAULT_BACKEND="cu128"

export VENV_DIR="${VENV_DIR:-${COSMOS_ROOT}/.venv}"
export TORCH_BACKEND="${TORCH_BACKEND:-${DEFAULT_BACKEND}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

echo "COSMOS_ROOT:     ${COSMOS_ROOT}"
echo "VENV_DIR:        ${VENV_DIR}"
echo "Host CUDA major: ${CUDA_MAJOR}  ->  torch-backend=${TORCH_BACKEND}"
echo "HF_HOME:         ${HF_HOME}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN: <set>"
else
  echo "HF_TOKEN: <unset> (export it if nvidia/Cosmos-Reason1-7B download is gated)"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# Headless servers need these system graphics libraries for video decoding.
install_system_libs() {
  local pkgs=(libxcb1 libgl1 libglib2.0-0)
  command -v apt-get >/dev/null 2>&1 || return 0
  local sudo=""
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 && sudo="sudo" || {
      echo "Skipping system library install (need root). If imports fail, install: ${pkgs[*]}" >&2
      return 0
    }
  fi
  echo "Installing system graphics libraries: ${pkgs[*]}"
  ${sudo} apt-get update -y || true
  ${sudo} apt-get install -y "${pkgs[@]}" || \
    echo "Warning: failed to install system libraries; install manually if imports fail: ${pkgs[*]}" >&2
}

install_system_libs

# Create the venv if it does not exist yet.
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  echo "Using existing venv: ${VENV_DIR}"
else
  echo "Creating venv: ${VENV_DIR}"
  uv venv --python 3.13 --seed --managed-python "${VENV_DIR}"
fi

# Install the transformers video-inference stack. torch/transformers/opencv are
# unconstrained so an existing satisfying install (e.g. from setup_reasoner.sh) is
# kept; only the missing pieces (qwen-vl-utils, accelerate) get added.
echo "Installing transformers inference deps (torch-backend=${TORCH_BACKEND})"
uv pip install --python "${VENV_DIR}/bin/python" --torch-backend="${TORCH_BACKEND}" \
  torch torchvision transformers accelerate qwen-vl-utils opencv-python-headless

echo
echo "Setup complete. Run Cosmos-Reason1 inference with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python ${COSMOS_ROOT}/transformers_cosmos1_reasoning.py --dataset generated_vids --exp_name smoke"
echo
echo "(nvidia/Cosmos-Reason1-7B weights download on first run; export HF_TOKEN if gated.)"
