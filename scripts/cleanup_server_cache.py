#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


PATTERNS = (
    "himawari-*-visible.png",
    "himawari-*-infrared.png",
    "cira-*-geocolor.png",
)


def cleanup(root: Path, *, now: datetime | None = None, dry_run: bool = False) -> dict:
    now = now or datetime.now().astimezone()
    today = now.astimezone().date()
    candidates = {path for pattern in PATTERNS for path in root.glob(pattern)}
    keep: set[Path] = set()
    for pattern in PATTERNS:
        matches = list(root.glob(pattern))
        if matches:
            keep.add(max(matches, key=lambda path: path.stat().st_mtime_ns))

    deleted = []
    reclaimed = 0
    for path in sorted(candidates):
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=now.tzinfo).date()
        if path in keep or modified >= today:
            continue
        size = path.stat().st_size
        deleted.append(path.name)
        reclaimed += size
        if not dry_run:
            path.unlink(missing_ok=True)
    return {
        "dry_run": dry_run,
        "deleted_count": len(deleted),
        "reclaimed_bytes": reclaimed,
        "preserved_latest": sorted(path.name for path in keep),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prune previous-day Earthwall satellite cache")
    parser.add_argument("root", nargs="?", type=Path, default=Path("/var/cache/earthwall"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(cleanup(args.root, dry_run=args.dry_run), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
