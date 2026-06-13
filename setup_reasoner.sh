#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Set up the environment to run vllm_cosmos_reasoning.py (Cosmos3 Reasoner anomaly
# reasoning behind a vLLM OpenAI-compatible server).
#
# This mirrors cell 4 of cookbooks/cosmos3/reasoner/run_with_vllm.ipynb: it clones
# the Cosmos Framework into packages/cosmos3 and installs vLLM + the cosmos3
# packages (and the openai client used by the script) into a uv-managed virtual
# environment (.venv at the repo root).
#
# Usage:
#   ./setup_reasoner.sh
#   source .venv/bin/activate
#   python vllm_cosmos_reasoning.py --dataset generated_vids --exp_name smoke --model nano
#
# The vLLM wheel and torch backend are paired to the host CUDA driver and are
# auto-selected, but can be overridden, e.g. for a CUDA 12.8 driver:
#   TORCH_BACKEND=cu128 VLLM_VERSION=0.19.1 ./setup_reasoner.sh

set -euo pipefail

# Resolve the repo root by walking up from this script until we find the
# checkout markers (README.md + cookbooks/).
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

# Pick the vLLM/torch-backend pairing from the host CUDA driver major version
# (CUDA 13 -> cu130/vllm 0.21.0, otherwise cu128/vllm 0.19.1). Override via env.
detect_cuda_major() {
  local ver
  ver="$(nvidia-smi 2>/dev/null | grep -oiE 'CUDA Version: [0-9]+' | grep -oE '[0-9]+' | head -1)"
  printf '%s\n' "${ver:-13}"
}
CUDA_MAJOR="$(detect_cuda_major)"
if [[ "${CUDA_MAJOR}" == "13" ]]; then
  DEFAULT_BACKEND="cu130"; DEFAULT_VLLM="0.21.0"
else
  DEFAULT_BACKEND="cu128"; DEFAULT_VLLM="0.19.1"
fi

# Environment variables (existing values win).
export COSMOS3_REPO="${COSMOS3_REPO:-${COSMOS_ROOT}/packages/cosmos3}"
export COSMOS3_GIT_URL="${COSMOS3_GIT_URL:-https://github.com/NVIDIA/cosmos-framework.git}"
export VENV_DIR="${VENV_DIR:-${COSMOS_ROOT}/.venv}"
export TORCH_BACKEND="${TORCH_BACKEND:-${DEFAULT_BACKEND}}"
export VLLM_VERSION="${VLLM_VERSION:-${DEFAULT_VLLM}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

echo "COSMOS_ROOT:     ${COSMOS_ROOT}"
echo "COSMOS3_REPO:    ${COSMOS3_REPO}"
echo "VENV_DIR:        ${VENV_DIR}"
echo "Host CUDA major: ${CUDA_MAJOR}  ->  torch-backend=${TORCH_BACKEND}, vllm==${VLLM_VERSION}"
echo "HF_HOME:         ${HF_HOME}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN: <set>"
else
  echo "HF_TOKEN: <unset> (required for gated nvidia/Cosmos3-Nano / Cosmos3-Super download)"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# Headless servers need these system graphics libraries for video decoding.
install_system_libs() {
  local pkgs=(libxcb1 libgl1 libglib2.0-0)
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi
  local sudo=""
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      sudo="sudo"
    else
      echo "Skipping system library install (need root). If imports fail, install: ${pkgs[*]}" >&2
      return 0
    fi
  fi
  echo "Installing system graphics libraries: ${pkgs[*]}"
  ${sudo} apt-get update -y || true
  ${sudo} apt-get install -y "${pkgs[@]}" || \
    echo "Warning: failed to install system libraries; install manually if imports fail: ${pkgs[*]}" >&2
}

install_system_libs

# 1. Clone (or reuse) the Cosmos Framework checkout (provides the cosmos3 packages).
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$(dirname "${COSMOS3_REPO}")"
if [[ -d "${COSMOS3_REPO}/.git" ]]; then
  echo "Using existing framework checkout: ${COSMOS3_REPO}"
else
  echo "Cloning ${COSMOS3_GIT_URL} into ${COSMOS3_REPO}"
  git clone "${COSMOS3_GIT_URL}" "${COSMOS3_REPO}"
fi

# 2. Create the virtual environment.
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  echo "Using existing venv: ${VENV_DIR}"
else
  echo "Creating venv: ${VENV_DIR}"
  uv venv --python 3.13 --seed --managed-python "${VENV_DIR}"
fi

# 3. Install vLLM, the cosmos3 packages (from the checkout), and the openai client.
echo "Installing vLLM (${VLLM_VERSION}, ${TORCH_BACKEND}) + cosmos3 packages + openai"
uv pip install --python "${VENV_DIR}/bin/python" --torch-backend="${TORCH_BACKEND}" \
  "vllm==${VLLM_VERSION}" \
  "${COSMOS3_REPO}/packages/transformers-cosmos3" \
  "${COSMOS3_REPO}/packages/vllm-cosmos3" \
  openai

echo
echo "Setup complete. Run the reasoner with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python ${COSMOS_ROOT}/vllm_cosmos_reasoning.py --dataset generated_vids --exp_name smoke --model nano"
echo
echo "(Cosmos3-Nano weights download on first launch; export HF_TOKEN if the model is gated.)"
