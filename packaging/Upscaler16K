#!/bin/bash
# Лаунчер .app: при первом запуске ставит окружение (в ~/Library/Application Support), далее просто запускает GUI.
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
SUPPORT="$HOME/Library/Application Support/Upscaler16K"
mkdir -p "$SUPPORT"
exec >>"$SUPPORT/launcher.log" 2>&1
echo "--- $(date) launch $*"
export UPSCALER_MODELS_DIR="$SUPPORT/models"
export PYTORCH_ENABLE_MPS_FALLBACK=1
if [ -f "$SUPPORT/.ready-1.0.0" ] && [ -x "$SUPPORT/venv/bin/python" ]; then
  cd "$APP_DIR"
  exec "$SUPPORT/venv/bin/python" "$APP_DIR/app.py" "$@"
fi
open -a Terminal "$APP_DIR/first_run.command"
