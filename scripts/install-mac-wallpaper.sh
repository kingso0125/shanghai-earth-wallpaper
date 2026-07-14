#!/bin/zsh
set -euo pipefail

repo="${0:A:h:h}"
agent="$HOME/Library/LaunchAgents/com.kingso.earthwall.mac.plist"
saver="$HOME/Library/Screen Savers/Earthwall.saver"
logs="$HOME/Library/Logs/EarthwallMac"
uid="$(id -u)"

mkdir -p "${agent:h}" "${saver:h}" "$logs"
rm -rf "$saver"
mkdir -p "$saver/Contents/MacOS"
/bin/cp "$repo/macos/EarthwallScreenSaver-Info.plist" "$saver/Contents/Info.plist"
/usr/bin/clang -fobjc-arc -dynamiclib \
  -framework Cocoa -framework ScreenSaver \
  "$repo/macos/EarthwallScreenSaverView.m" \
  -o "$saver/Contents/MacOS/Earthwall"
/usr/bin/codesign --force --sign - "$saver"

cat > "$agent" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.kingso.earthwall.mac</string>
  <key>ProgramArguments</key>
  <array><string>$repo/scripts/mac-hourly.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>StartCalendarInterval</key>
  <dict><key>Minute</key><integer>28</integer></dict>
  <key>StandardOutPath</key><string>$logs/hourly.log</string>
  <key>StandardErrorPath</key><string>$logs/hourly-error.log</string>
</dict>
</plist>
PLIST

/bin/launchctl bootout "gui/$uid/com.kingso.earthwall.mac" 2>/dev/null || true
/bin/launchctl bootstrap "gui/$uid" "$agent"
/bin/launchctl enable "gui/$uid/com.kingso.earthwall.mac"

# Legacy ScreenSaver.framework remains the compatible local-image path.
/usr/bin/defaults -currentHost write com.apple.screensaver moduleDict -dict \
  moduleName Earthwall path "$saver" type -int 0
/usr/bin/defaults -currentHost write com.apple.screensaver moduleName -string Earthwall
/usr/bin/defaults -currentHost write com.apple.screensaver modulePath -string "$saver"

"$repo/scripts/mac-hourly.sh"
/bin/launchctl kickstart -k "gui/$uid/com.kingso.earthwall.mac"
