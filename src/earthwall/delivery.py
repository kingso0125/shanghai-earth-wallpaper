"""Publish one checked server release, keeping the phone and server identical."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .location import Location
from .qa import audit
from .sources import _request, sha256
from .target import SERVER_MANIFEST


def copy_server_release(output: Path, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    manifest = json.loads(_request(SERVER_MANIFEST, timeout=20))
    rendered = datetime.fromisoformat(manifest["rendered_utc"].replace("Z", "+00:00"))
    if rendered.tzinfo is None:
        raise ValueError("server release has no timezone")
    age = (now - rendered).total_seconds()
    if not 0 <= age <= 85 * 60:
        raise ValueError(f"server release is stale or future-dated: {age:.0f}s")
    target = manifest["target"]
    Location(float(target["latitude"]), float(target["longitude"]), target["name"])
    version = rendered.astimezone(UTC).isoformat().replace("+00:00", "Z").replace(":", "").replace(".", "-")
    base_url = SERVER_MANIFEST.rsplit("/", 1)[0] + "/releases/" + version
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".delivery-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for name in ("lock", "home"):
            artifact = manifest["artifacts"][name]
            if artifact["file"] != f"{name}.jpg":
                raise ValueError("unexpected server artifact filename")
            path = staging / artifact["file"]
            path.write_bytes(_request(f"{base_url}/{name}.jpg", timeout=30))
            if sha256(path) != artifact["sha256"]:
                raise ValueError(f"{name} download checksum mismatch")
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        quality = audit(staging)
        if not quality["passed"]:
            raise ValueError("server release quality check failed: " + "; ".join(quality["failures"]))
        (staging / "qa.json").write_text(json.dumps(quality, indent=2) + "\n")
        output.mkdir(parents=True, exist_ok=True)
        for name in ("lock.jpg", "home.jpg", "qa.json", "manifest.json"):
            shutil.copy2(staging / name, output / name)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = copy_server_release(args.output)
    print(json.dumps({"delivery": "verified-server-release", "rendered_utc": manifest["rendered_utc"], "target": manifest["target"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
