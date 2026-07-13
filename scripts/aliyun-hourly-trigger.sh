#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPOSITORY="git@github.com:kingso0125/shanghai-earth-wallpaper.git"
readonly WORK_DIR="/opt/shanghai-earth-wallpaper-scheduler"
readonly KEY_FILE="/etc/shanghai-earth-wallpaper/deploy_key"

export GIT_SSH_COMMAND="/usr/bin/ssh -i ${KEY_FILE} -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes"

if [[ ! -d "${WORK_DIR}/.git" ]]; then
  /usr/bin/mkdir -p "${WORK_DIR}"
  /usr/bin/git -C "${WORK_DIR}" init --quiet
  /usr/bin/git -C "${WORK_DIR}" remote add origin "${REPOSITORY}"
fi

/usr/bin/git -C "${WORK_DIR}" fetch --quiet --depth=1 origin main
/usr/bin/git -C "${WORK_DIR}" checkout --quiet --detach FETCH_HEAD
/usr/bin/git -C "${WORK_DIR}" \
  -c user.name="Shanghai Earth Scheduler" \
  -c user.email="scheduler@localhost" \
  commit --quiet --allow-empty -m "chore hourly scheduler heartbeat $(/usr/bin/date --iso-8601=seconds)"
/usr/bin/git -C "${WORK_DIR}" push --quiet --force origin HEAD:refs/heads/scheduler

printf 'scheduler heartbeat pushed at %s\n' "$(/usr/bin/date --iso-8601=seconds)"
