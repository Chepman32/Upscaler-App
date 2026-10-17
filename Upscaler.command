#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  bash ./setup.command
fi
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec python app.py "$@"
