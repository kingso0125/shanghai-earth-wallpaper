#!/bin/zsh
set -euo pipefail

root="${0:A:h:h}"
source_plist="$root/shortcuts/更新上海实时地球.plist"
output="$root/web/更新上海实时地球.shortcut"
unsigned="$(mktemp -t shanghai-earth-wallpaper).shortcut"
trap 'rm -f "$unsigned"' EXIT

plutil -lint "$source_plist"
cp "$source_plist" "$unsigned"
plutil -convert binary1 "$unsigned"
shortcuts sign --mode anyone --input "$unsigned" --output "$output"
chmod 644 "$output"
shasum -a 256 "$output"
