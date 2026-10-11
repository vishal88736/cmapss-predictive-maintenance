#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
training_device=cuda
case "${1:-}" in
  ""|--cuda) ;;
  --cpu) training_device=cpu ;;
  -h|--help)
    printf '%s\n' 'Usage: bash scripts/setup_env.sh [--cuda|--cpu]' \
      'CUDA is the default. Use --cpu explicitly for an environment without GPU training.'
    exit 0 ;;
  *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
  printf '%s\n' 'Pass at most one setup option.' >&2
  exit 2
fi
if [ ! -x .venv/bin/python ]; then
  "${GRACE_PYTHON:-python3}" -m venv .venv
fi
# CUDA wheels can exceed a memory-backed /tmp. Keep installation staging on disk.
mkdir -p .venv/.install-tmp
export TMPDIR="$PWD/.venv/.install-tmp"
if [ "$training_device" = cuda ]; then
  if ! .venv/bin/python -c 'import sys,torch; version=tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]); sys.exit(0 if torch.version.cuda is not None and (2,6) <= version < (3,0) else 1)' 2>/dev/null; then
    cuda_wheel="${GRACE_CUDA_WHEEL:-cu130}"
    .venv/bin/python -m pip install --no-cache-dir --upgrade \
      "torch==${GRACE_TORCH_VERSION:-2.14.1}+$cuda_wheel" \
      --index-url "https://download.pytorch.org/whl/$cuda_wheel"
  fi
  .venv/bin/python scripts/check_cuda.py
else
  .venv/bin/python -m pip install --no-cache-dir --upgrade \
    "torch==${GRACE_TORCH_VERSION:-2.14.1}+cpu" --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python -m ipykernel install --prefix "$PWD/.venv" --name grace-rul --display-name 'Python (GRACE-RUL)'
printf 'Environment ready for %s. Start Jupyter with .venv/bin/python -m jupyterlab.\n' "$training_device"
