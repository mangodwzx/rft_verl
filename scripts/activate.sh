#!/usr/bin/env bash
# Source this file in every new training/evaluation shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source scripts/activate.sh' >&2
  exit 1
fi
export RFT_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RFT_RUNTIME="${RFT_RUNTIME:-$(dirname -- "$(dirname -- "$RFT_PROJECT_ROOT")")/rft_runtime}"
export XDG_CACHE_HOME="$RFT_RUNTIME/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$RFT_RUNTIME/python"
export UV_PYTHON_BIN_DIR="$RFT_RUNTIME/bin"
export HF_HOME="$RFT_RUNTIME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$RFT_RUNTIME/tmp"
export RAY_TMPDIR="$TMPDIR"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nvidia"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PATH="$RFT_RUNTIME/uv-pypi/bin:$RFT_RUNTIME/bin:$PATH"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" || return 1

# The locked stack uses CUDA 13. Apply forward compatibility only on the
# supported older driver families, before Python/Ray loads any CUDA libraries.
_rft_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
_rft_driver_major="${_rft_driver%%.*}"
_rft_compat="$RFT_PROJECT_ROOT/vendor/cuda-compat-13-0/usr/local/cuda-13.0/compat"
if [[ "$_rft_driver_major" == 535 || "$_rft_driver_major" == 570 ]]; then
  if [[ ! -f "$_rft_compat/libcuda.so.1" ]]; then
    echo 'CUDA 13 on this driver needs local compatibility libraries. Run: bash scripts/install_cuda_compat.sh' >&2
    unset _rft_driver _rft_driver_major _rft_compat
    return 1
  fi
  if [[ "${LD_LIBRARY_PATH:-}" != "$_rft_compat" && "${LD_LIBRARY_PATH:-}" != "$_rft_compat:"* ]]; then
    export LD_LIBRARY_PATH="$_rft_compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  echo "Using local CUDA 13 compatibility libraries with driver $_rft_driver."
fi
unset _rft_driver _rft_driver_major _rft_compat
source "$RFT_PROJECT_ROOT/vendor/verl/.venv/bin/activate" || return 1
