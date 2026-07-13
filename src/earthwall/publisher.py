from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path

from .location import Location
from .qa import audit
from .render import render_pair
from .sources import acquire


class Publisher:
    def __init__(self, root: Path, cache: Path, lock_file: Path):
        self.root = root
        self.cache = cache
        self.lock_file = lock_file

    def publish(self, location: Location) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "releases").mkdir(exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        with self.lock_file.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            staging = Path(
                tempfile.mkdtemp(prefix=".render-", dir=self.root / "releases")
            )
            try:
                observation = acquire(self.cache)
                manifest = render_pair(
                    observation,
                    staging,
                    target_latitude=location.latitude,
                    target_longitude=location.longitude,
                    target_name=location.name,
                )
                quality = audit(staging)
                if not quality["passed"]:
                    raise RuntimeError("wallpaper quality check failed: " + "; ".join(quality["failures"]))
                (staging / "qa.json").write_text(
                    json.dumps(quality, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

                release_name = manifest["rendered_utc"].replace(":", "").replace(".", "-")
                release = self.root / "releases" / release_name
                os.replace(staging, release)
                temporary_link = self.root / ".current-next"
                temporary_link.unlink(missing_ok=True)
                temporary_link.symlink_to(Path("releases") / release.name)
                os.replace(temporary_link, self.root / "current")
                self._prune(release)
                return manifest
            finally:
                if staging.exists():
                    shutil.rmtree(staging)

    def _prune(self, current: Path) -> None:
        releases = sorted(
            (path for path in (self.root / "releases").iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for stale in releases[2:]:
            if stale != current:
                shutil.rmtree(stale)
