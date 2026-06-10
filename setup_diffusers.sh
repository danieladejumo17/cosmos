#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Set up the Cosmos3 audiovisual Diffusers environment.
#
# This mirrors the "Configure Paths and Environment" and "Install Diffusers
# Dependencies" steps of run_with_diffusers.ipynb: it exports the same
# environment variables (honoring any you have already set) and installs the
# diffusers dependencies into a uv-managed Python 3.13 virtual environment.
#
# Usage:
#   ./setup_diffusers.sh             # create venv + install deps
#   source ./setup_diffusers.sh      # also export the env vars into your shell
#
# Override any default by exporting the variable before running, e.g.:
#   COSMOS3_TORCH_BACKEND=cu128 ./setup_diffusers.sh

set -euo pipefail

# Resolve the repo root by walking up from this script until we find the
# checkout markers (README.md + cookbooks/), matching find_repo_root() in the
# notebook.
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
  # Fall back to the script directory if no marker is found.
  printf '%s\n' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
}

COSMOS_ROOT="$(find_repo_root)"
COSMOS3_AUDIOVISUAL_ROOT="${COSMOS_ROOT}/cookbooks/cosmos3/generator/audiovisual"

# Environment variables (same defaults as the notebook; existing values win).
export COSMOS3_DIFFUSERS_VENV="${COSMOS3_DIFFUSERS_VENV:-${COSMOS_ROOT}/.venv-cosmos3-diffusers}"
export COSMOS3_TORCH_BACKEND="${COSMOS3_TORCH_BACKEND:-cu130}"
export COSMOS3_AUDIOVISUAL_OUTPUT_ROOT="${COSMOS3_AUDIOVISUAL_OUTPUT_ROOT:-${COSMOS3_AUDIOVISUAL_ROOT}/outputs/notebooks}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "COSMOS_ROOT: ${COSMOS_ROOT}"
for key in \
  COSMOS3_DIFFUSERS_VENV \
  COSMOS3_TORCH_BACKEND \
  COSMOS3_AUDIOVISUAL_OUTPUT_ROOT \
  UV_CACHE_DIR \
  UV_LINK_MODE \
  HF_HOME \
  HF_HUB_DISABLE_XET \
  CUDA_VISIBLE_DEVICES; do
  printf '%s: %s\n' "${key}" "${!key}"
done
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN: <set>"
else
  echo "HF_TOKEN: <unset>"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  # When sourced, return instead of killing the parent shell.
  return 1 2>/dev/null || exit 1
fi

# Headless servers need these system graphics libraries for the pipeline import.
# Best effort: only attempt when running as root or with passwordless sudo.
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
      echo "Skipping system library install (need root). If you hit 'libxcb.so.1' errors, install: ${pkgs[*]}" >&2
      return 0
    fi
  fi
  echo "Installing system graphics libraries: ${pkgs[*]}"
  ${sudo} apt-get update -y || true
  ${sudo} apt-get install -y "${pkgs[@]}" || \
    echo "Warning: failed to install system libraries; install manually if imports fail: ${pkgs[*]}" >&2
}

install_system_libs

echo "Creating virtual environment at: ${COSMOS3_DIFFUSERS_VENV}"
uv venv "${COSMOS3_DIFFUSERS_VENV}" --python 3.13 --seed --managed-python --allow-existing

# shellcheck disable=SC1091
source "${COSMOS3_DIFFUSERS_VENV}/bin/activate"

echo "Installing Diffusers dependencies (torch backend: ${COSMOS3_TORCH_BACKEND})"
uv pip install --torch-backend="${COSMOS3_TORCH_BACKEND}" \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate \
  av \
  cosmos_guardrail \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  ipykernel \
  torch \
  torchvision \
  transformers

"${COSMOS3_DIFFUSERS_VENV}/bin/python" -m ipykernel install --user \
  --name cosmos3-diffusers \
  --display-name "Cosmos3 Diffusers (Python 3.13)"

echo
echo "Installed dependencies into: ${COSMOS3_DIFFUSERS_VENV}"
echo "Jupyter kernel registered: Cosmos3 Diffusers (Python 3.13)"
echo "Activate the venv with: source ${COSMOS3_DIFFUSERS_VENV}/bin/activate"
