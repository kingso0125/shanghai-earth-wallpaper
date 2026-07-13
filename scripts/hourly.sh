#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
lock_dir="${TMPDIR:-/tmp}/earthwall-hourly.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "another earthwall render is running" >&2
  exit 0
fi
trap 'rmdir "$lock_dir"' EXIT INT TERM

PYTHONPATH=src python3 -m earthwall.cli --cache cache --output output/current
PYTHONPATH=src python3 -m earthwall.qa output/current

