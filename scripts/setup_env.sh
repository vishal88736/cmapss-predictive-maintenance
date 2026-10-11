#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  "${GRACE_PYTHON:-python3}" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix "$PWD/.venv" --name grace-rul --display-name 'Python (GRACE-RUL)'
printf '%s\n' 'Environment ready. Start Jupyter with .venv/bin/python -m jupyterlab.'
