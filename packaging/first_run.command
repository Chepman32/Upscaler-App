#!/bin/bash
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_BUNDLE="$(cd "$APP_DIR/../../.." && pwd)"
SUPPORT="$HOME/Library/Application Support/Upscaler16K"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
fail() { echo; echo "✖ Установка не удалась (см. сообщение выше). Проверьте интернет и откройте приложение снова."; read -r -p "Enter — закрыть…" _; exit 1; }
echo "== Upscaler 16K: первая установка =="
echo "Скачиваются нейросетевые компоненты (≈2 ГБ), 5–15 минут. После этого приложение запустится само, интернет больше не нужен."
mkdir -p "$SUPPORT" || fail
if ! command -v uv >/dev/null 2>&1; then
  echo "Устанавливаю uv…"; curl -LsSf https://astral.sh/uv/install.sh | sh || fail
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
export UV_PYTHON_INSTALL_DIR="$SUPPORT/python"
uv venv --python 3.12 --allow-existing "$SUPPORT/venv" || fail
uv pip install --python "$SUPPORT/venv/bin/python" -r "$APP_DIR/requirements.txt" || fail
cd "$APP_DIR"
UPSCALER_MODELS_DIR="$SUPPORT/models" "$SUPPORT/venv/bin/python" -m upscaler.models || fail
touch "$SUPPORT/.ready-1.0.0"
echo; echo "✔ Готово. Запускаю приложение…"
open "$APP_BUNDLE"
sleep 2
osascript -e 'tell application "Terminal" to close (every window whose name contains "first_run")' >/dev/null 2>&1
exit 0
