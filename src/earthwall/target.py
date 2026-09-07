"""Share the phone's accepted city with Pages and Mac without a GPS history."""
import json
from pathlib import Path
from .location import Location
from .sources import _request

SERVER_MANIFEST = "https://47.116.45.167:8443/earthwall/manifest.json"


def follow_server(cache: Path, fallback: Location) -> Location:
    cached = cache / "accepted-target.json"
    try:
        manifest = json.loads(_request(SERVER_MANIFEST, timeout=10))
        target = manifest["target"]
        location = Location(float(target["latitude"]), float(target["longitude"]), str(target["name"]))
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_suffix(".tmp")
        temporary.write_text(json.dumps(target, ensure_ascii=False))
        temporary.replace(cached)
        return location
    except (OSError, KeyError, TypeError, ValueError):
        try:
            target = json.loads(cached.read_text())
            return Location(float(target["latitude"]), float(target["longitude"]), str(target["name"]))
        except (OSError, KeyError, TypeError, ValueError):
            return fallback
