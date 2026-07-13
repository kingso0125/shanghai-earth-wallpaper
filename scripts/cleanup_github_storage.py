#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")
EARTH_CACHE_PREFIX = "earthwall-cache-"


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _is_before_today(item: dict, today) -> bool:
    return _timestamp(item["created_at"]).astimezone(SHANGHAI).date() < today


def select_cache_ids(caches: Iterable[dict], today) -> list[int]:
    earth_caches = [item for item in caches if item["key"].startswith(EARTH_CACHE_PREFIX)]
    if not earth_caches:
        return []
    latest_id = max(earth_caches, key=lambda item: _timestamp(item["created_at"]))["id"]
    return [
        item["id"]
        for item in earth_caches
        if item["id"] != latest_id and _is_before_today(item, today)
    ]


def select_artifact_ids(artifacts: Iterable[dict], today) -> list[int]:
    return [item["id"] for item in artifacts if _is_before_today(item, today)]


class GitHubAPI:
    def __init__(self, repository: str, token: str):
        self.base = f"repos/{repository}"
        self.env = os.environ.copy()
        self.env["GH_TOKEN"] = token

    def _request(self, path: str, method: str = "GET") -> str:
        command = ["gh", "api"]
        if method != "GET":
            command.extend(["--method", method])
        command.append(f"{self.base}{path}")
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=self.env,
        ).stdout

    def list_all(self, path: str, key: str) -> list[dict]:
        items = []
        page = 1
        while True:
            separator = "&" if "?" in path else "?"
            batch = json.loads(self._request(f"{path}{separator}per_page=100&page={page}"))[key]
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1

    def delete(self, path: str) -> None:
        self._request(path, method="DELETE")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Delete yesterday's GitHub Earth caches and artifacts")
    parser.add_argument("--execute", action="store_true", help="perform deletions instead of a dry run")
    args = parser.parse_args(argv)

    repository = os.environ.get("GITHUB_REPOSITORY", "kingso0125/shanghai-earth-wallpaper")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        parser.error("GITHUB_TOKEN is required")

    api = GitHubAPI(repository, token)
    caches = api.list_all("/actions/caches", "actions_caches")
    artifacts = api.list_all("/actions/artifacts", "artifacts")
    today = datetime.now(SHANGHAI).date()
    cache_ids = select_cache_ids(caches, today)
    artifact_ids = select_artifact_ids(artifacts, today)

    if args.execute:
        for cache_id in cache_ids:
            api.delete(f"/actions/caches/{cache_id}")
        for artifact_id in artifact_ids:
            api.delete(f"/actions/artifacts/{artifact_id}")

    selected_caches = [item for item in caches if item["id"] in set(cache_ids)]
    selected_artifacts = [item for item in artifacts if item["id"] in set(artifact_ids)]
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "dry-run",
                "today_shanghai": str(today),
                "deleted_cache_ids": cache_ids,
                "deleted_artifact_ids": artifact_ids,
                "estimated_bytes_freed": sum(
                    item.get("size_in_bytes", 0) for item in selected_caches + selected_artifacts
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
