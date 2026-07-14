#!/bin/zsh
set -euo pipefail

repo="${0:A:h:h}"
state="${EARTHWALL_MAC_STATE:-$HOME/Library/Application Support/EarthwallMac}"
python="${EARTHWALL_PYTHON:-/Library/Frameworks/Python.framework/Versions/Current/bin/python3}"
cache="$state/cache"
current="$state/current"
wallpapers="$state/wallpapers"
mkdir -p "$cache" "$current" "$wallpapers"

# Read-only source synchronization: the server already validates the freshest
# phone observation. Mac still renders its own independent composition.
source_host="${EARTHWALL_SOURCE_HOST:-root@47.116.45.167}"
latest_visible="$(/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=8 "$source_host" \
  "ls -1t /var/cache/earthwall/himawari-*-visible.png 2>/dev/null | head -1" 2>/dev/null || true)"
if [[ -n "$latest_visible" ]]; then
  latest_infrared="${latest_visible%-visible.png}-infrared.png"
  /usr/bin/scp -q -o BatchMode=yes -o ConnectTimeout=8 \
    "$source_host:$latest_visible" "$source_host:$latest_infrared" \
    "$source_host:/srv/earthwall/current/manifest.json" "$cache/" 2>/dev/null || true
  [[ -f "$cache/manifest.json" ]] && /bin/mv -f "$cache/manifest.json" "$cache/server-manifest.json"
fi

work="$(mktemp -d "$state/render.XXXXXX")"
trap 'rm -rf "$work"' EXIT

cd "$repo"
PYTHONPATH=src "$python" -m earthwall.mac_cli \
  --cache "$cache" \
  --output "$work" \
  --latitude 31.2304 \
  --longitude 121.4737 \
  --location-name Shanghai > "$work/render.json"
PYTHONPATH=src "$python" -m earthwall.mac_qa "$work" > "$work/qa.json"

for file in mac-lock.jpg mac-home.jpg mac-manifest.json qa.json; do
  /usr/bin/install -m 0644 "$work/$file" "$current/.$file.new"
  /bin/mv -f "$current/.$file.new" "$current/$file"
done

stamp="$(date +%Y%m%dT%H%M%S)"
desktop="$wallpapers/mac-home-$stamp.jpg"
/bin/cp "$current/mac-home.jpg" "$desktop"

EARTHWALL_DESKTOP="$desktop" /usr/bin/osascript <<'APPLESCRIPT'
set imagePath to system attribute "EARTHWALL_DESKTOP"
tell application "System Events"
    repeat with currentDesktop in desktops
        set picture of currentDesktop to POSIX file imagePath
    end repeat
end tell
APPLESCRIPT

# Keep one day of versioned desktop images and two days of source cache.
/usr/bin/find "$wallpapers" -type f -name 'mac-home-*.jpg' -mtime +1 -delete
/usr/bin/find "$cache" -type f \( -name '*-visible.png' -o -name '*-infrared.png' \) -mtime +2 -delete
