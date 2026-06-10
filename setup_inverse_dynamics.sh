#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Set up the Cosmos Framework environment for action inverse-dynamics inference.
#
# This mirrors steps 3-4 of run_id_with_cosmos_framework.ipynb: it clones the
# Cosmos Framework into packages/cosmos3 and installs its dependencies into a
# uv-managed virtual environment (.venv inside the checkout). It also ensures
# huggingface_hub is available in that venv for the action-file upload step.
#
# Usage:
#   ./setup_inverse_dynamics.sh
#
# Override any default by exporting the variable before running, e.g.:
#   COSMOS3_UV_GROUP=cu128-train ./setup_inverse_dynamics.sh   # CUDA 12.x driver

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
  printf '%s\n' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
}

COSMOS_ROOT="$(find_repo_root)"

# Environment variables (same defaults as the notebook; existing values win).
export COSMOS3_REPO="${COSMOS3_REPO:-${COSMOS_ROOT}/packages/cosmos3}"
export COSMOS3_GIT_URL="${COSMOS3_GIT_URL:-https://github.com/NVIDIA/cosmos-framework.git}"
export COSMOS3_UV_GROUP="${COSMOS3_UV_GROUP:-cu130-train}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "COSMOS_ROOT:      ${COSMOS_ROOT}"
echo "COSMOS3_REPO:     ${COSMOS3_REPO}"
echo "COSMOS3_GIT_URL:  ${COSMOS3_GIT_URL}"
echo "COSMOS3_UV_GROUP: ${COSMOS3_UV_GROUP}"
echo "HF_HOME:          ${HF_HOME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN: <set>"
else
  echo "HF_TOKEN: <unset> (required for gated nvidia/Cosmos3-Nano download and dataset upload)"
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

# 1. Clone (or reuse) the Cosmos Framework checkout.
mkdir -p "$(dirname "${COSMOS3_REPO}")"
if [[ -d "${COSMOS3_REPO}/.git" ]]; then
  echo "Using existing framework checkout: ${COSMOS3_REPO}"
else
  echo "Cloning ${COSMOS3_GIT_URL} into ${COSMOS3_REPO}"
  git clone "${COSMOS3_GIT_URL}" "${COSMOS3_REPO}"
fi

# 2. Install dependencies (heavier "train" group, matching the notebook audit).
export GIT_LFS_SKIP_SMUDGE=1
cd "${COSMOS3_REPO}"
echo "Installing Cosmos Framework dependencies (uv group: ${COSMOS3_UV_GROUP})"
uv sync --all-extras --group="${COSMOS3_UV_GROUP}"

# 3. Ensure huggingface_hub is available in the framework venv (used by the
#    run script to upload action files). It is usually pulled in transitively,
#    but install it explicitly to be safe.
if ! "${COSMOS3_REPO}/.venv/bin/python" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "Installing huggingface_hub into the framework venv"
  uv pip install huggingface_hub
fi

echo
echo "Installed Cosmos Framework into: ${COSMOS3_REPO}"
echo "Run inverse dynamics with:"
echo "  HF_TOKEN=<token> ${COSMOS3_REPO}/.venv/bin/python ${COSMOS_ROOT}/run_inverse_dynamics.py"
