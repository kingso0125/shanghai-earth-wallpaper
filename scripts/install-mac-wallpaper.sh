#!/bin/zsh
set -euo pipefail

repo="${0:A:h:h}"
agent="$HOME/Library/LaunchAgents/com.kingso.earthwall.mac.plist"
logs="$HOME/Library/Logs/EarthwallMac"
uid="$(id -u)"

mkdir -p "${agent:h}" "$logs"

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

"$repo/scripts/mac-hourly.sh"
PYTHONPATH="$repo/src" /usr/bin/python3 -m earthwall.mac_lock \
  "$HOME/Library/Application Support/EarthwallMac/current/mac-lock.jpg"
/usr/bin/killall WallpaperAgent ScreenSaverEngine 2>/dev/null || true
/bin/launchctl kickstart -k "gui/$uid/com.kingso.earthwall.mac"
