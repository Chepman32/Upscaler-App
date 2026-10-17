#!/bin/bash
# Одноразовая установка: окружение + зависимости + веса моделей. После неё интернет не нужен.
set -e
cd "$(dirname "$0")"
echo "== Upscaler 16K: установка =="

if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Устанавливаю uv (менеджер Python)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r requirements.txt
python -m upscaler.models
echo
echo "Готово. Запуск: дважды щёлкните Upscaler.command"
read -r -p "Нажмите Enter, чтобы закрыть…" _
