#!/bin/bash
# Сборка "Upscaler 16K.app" и Upscaler16K-<версия>.dmg. Запускать на macOS.  Опция --install: поставить в /Applications.
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; VER=1.0.3
OUT="$ROOT/dist"; STAGE="$OUT/stage"; APP="$STAGE/Upscaler 16K.app"
DMG="$OUT/Upscaler16K-$VER.dmg"
rm -rf "$OUT"; mkdir -p "$STAGE"
# настоящий Mach-O-апплет (osacompile есть в любой macOS) вместо shell-скрипта: LaunchServices на Apple Silicon требует его
osacompile -o "$APP" packaging/launcher.applescript
PL="$APP/Contents/Info.plist"; PB=/usr/libexec/PlistBuddy
setp() { "$PB" -c "Set :$1 '$3'" "$PL" 2>/dev/null || "$PB" -c "Add :$1 $2 '$3'" "$PL"; }
setp CFBundleName string "Upscaler 16K"
setp CFBundleDisplayName string "Upscaler 16K"
setp CFBundleIdentifier string "local.upscaler16k"
setp CFBundleShortVersionString string "$VER"
setp CFBundleVersion string "$VER"
"$PB" -c "Delete :CFBundleIconName" "$PL" 2>/dev/null || true
ICON="$("$PB" -c "Print :CFBundleIconFile" "$PL")"
cp packaging/AppIcon.icns "$APP/Contents/Resources/$ICON.icns"
rm -f "$APP/Contents/Resources/Assets.car"
mkdir -p "$APP/Contents/Resources/app"
cp packaging/Upscaler16K "$APP/Contents/Resources/app/launch.sh"
cp packaging/first_run.command "$APP/Contents/Resources/app/first_run.command"
cp app.py upscale_cli.py fix_colors.py requirements.txt "$APP/Contents/Resources/app/"
cp -R upscaler "$APP/Contents/Resources/app/upscaler"
find "$APP" -name __pycache__ -type d -prune -exec rm -rf {} +
chmod +x "$APP/Contents/Resources/app/launch.sh" "$APP/Contents/Resources/app/first_run.command"
codesign --force --deep -s - "$APP"
ln -s /Applications "$STAGE/Applications"
# 1) собираем RW-образ, кладём иконку тома (.VolumeIcon.icns + флаг "custom icon"), 2) сжимаем в UDZO
RW="$OUT/rw.dmg"; RWMNT="$OUT/rwmnt"
hdiutil create -volname "Upscaler 16K" -srcfolder "$STAGE" -ov -format UDRW -fs HFS+ "$RW" >/dev/null
mkdir -p "$RWMNT"
hdiutil attach "$RW" -nobrowse -quiet -mountpoint "$RWMNT"
cp packaging/AppIcon.icns "$RWMNT/.VolumeIcon.icns"
xattr -wx com.apple.FinderInfo "0000000000000000040000000000000000000000000000000000000000000000" "$RWMNT" || true
chflags hidden "$RWMNT/.VolumeIcon.icns" 2>/dev/null || true
sync; hdiutil detach "$RWMNT" -quiet || { sleep 2; hdiutil detach "$RWMNT" -force -quiet; }
hdiutil convert "$RW" -format UDZO -imagekey zlib-level=9 -ov -o "$DMG" >/dev/null
rm -f "$RW"; rmdir "$RWMNT" 2>/dev/null || true
# иконка самого файла .dmg (хранится в метаданных файла на этом Mac)
osascript -l JavaScript -e 'ObjC.import("AppKit"); $.NSWorkspace.sharedWorkspace.setIconForFileOptions($.NSImage.alloc.initWithContentsOfFile("'"$ROOT/packaging/AppIcon.icns"'"), "'"$DMG"'", 0)' >/dev/null 2>&1 \
  || echo "⚠ не удалось поставить иконку на файл .dmg"
echo "DMG: $DMG ($(du -h "$DMG" | cut -f1))"
if [ "$1" = "--install" ]; then
  MNT="$(mktemp -d)/mnt"; mkdir -p "$MNT"
  hdiutil attach "$DMG" -nobrowse -quiet -mountpoint "$MNT"
  rm -rf "/Applications/Upscaler 16K.app"
  ditto "$MNT/Upscaler 16K.app" "/Applications/Upscaler 16K.app"
  hdiutil detach "$MNT" -quiet
  xattr -dr com.apple.quarantine "/Applications/Upscaler 16K.app" 2>/dev/null || true
  echo "Установлено: /Applications/Upscaler 16K.app"
fi
