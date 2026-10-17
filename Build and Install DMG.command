#!/bin/bash
# Собирает Upscaler16K-1.0.0.dmg (в папке dist/) и ставит приложение в /Applications, затем запускает его.
cd "$(dirname "$0")"
chmod +x packaging/*.sh packaging/Upscaler16K packaging/first_run.command
if bash packaging/build_dmg.sh --install; then
  echo; echo "✔ DMG собран и приложение установлено."
  open "/Applications/Upscaler 16K.app"
  open -R dist/Upscaler16K-*.dmg
else
  echo; echo "✖ Сборка не удалась."; read -r -p "Enter — закрыть…" _
fi
