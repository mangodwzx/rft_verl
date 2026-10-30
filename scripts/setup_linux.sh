#!/usr/bin/env bash
# Installs dependencies only when explicitly invoked. Does not launch experiments.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_COMMIT="3c6d9b43b67c74ec343d77874cd8fa6e841523da"
VERL_DIR="$ROOT/vendor/verl"
command -v git >/dev/null
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || { echo 'Training stack requires Linux/WSL2.' >&2; exit 1; }
mkdir -p "$ROOT/vendor"
if [[ ! -e "$VERL_DIR" ]]; then
  git clone https://github.com/verl-project/verl.git "$VERL_DIR"
  git -C "$VERL_DIR" checkout --detach "$VERL_COMMIT"
else
  [[ "$(git -C "$VERL_DIR" rev-parse HEAD)" == "$VERL_COMMIT" ]] || {
    echo 'Existing vendor/verl is a different revision. Move it aside or select the pinned commit explicitly.' >&2; exit 1;
  }
fi
[[ -z "$(git -C "$VERL_DIR" status --porcelain)" ]] || { echo 'Pinned verl checkout has local changes.' >&2; exit 1; }
(
  cd "$VERL_DIR"
  uv sync --frozen --python 3.12 --extra fsdp --extra vllm --extra math
)
# Use the SAME interpreter for the package, driver, Ray workers and generation.
uv pip install --python "$VERL_DIR/.venv/bin/python" -e "$ROOT[static,plots]"
uv pip freeze --python "$VERL_DIR/.venv/bin/python" > "$ROOT/vendor/environment.freeze.txt"
echo "Activate with: source '$VERL_DIR/.venv/bin/activate'"
echo 'For A800 with a 570 driver and this CUDA 13 stack: bash scripts/install_cuda_compat.sh; source scripts/activate.sh'
echo 'Setup complete. No data preparation, generation, or training was started.'
