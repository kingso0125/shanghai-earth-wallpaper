from __future__ import annotations

import hashlib
import io
import json
import re
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from .config import (
    BASE_LAYER,
    GIBS_CAPABILITIES,
    GIBS_ENDPOINT,
    IR_LAYER,
    LIGHTS_LAYER,
    TERRAIN_LAYER,
    VISIBLE_LAYER,
)


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    visible: Path
    infrared: Path
    geocolor: Path | None
    base: Path
    lights: Path
    status: str
    source: str = "CIRA SLIDER / KMA GK2A"
    satellite_longitude: float = 140.7
    terrain: Path | None = None


def _request(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "earthwall/0.1"})
    system_bundle = Path("/etc/ssl/cert.pem")
    context = (
        ssl.create_default_context(cafile=str(system_bundle))
        if system_bundle.exists()
        else ssl.create_default_context()
    )
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read()


def _atomic_download(url: str, destination: Path, expected_size=(4096, 2048)) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_bytes(_request(url))
    with Image.open(tmp) as image:
        image.verify()
    with Image.open(tmp) as image:
        if image.size != expected_size:
            raise ValueError(f"unexpected image size {image.size} from {url}")
    tmp.replace(destination)


def _wms_url(layer: str, *, timestamp: datetime | None, image_format: str) -> str:
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": image_format,
        "TRANSPARENT": "true" if image_format == "image/png" else "false",
        "SRS": "EPSG:4326",
        "BBOX": "-180,-90,180,90",
        "WIDTH": "4096",
        "HEIGHT": "2048",
    }
    if timestamp is not None:
        params["TIME"] = timestamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    query = urllib.parse.urlencode(params, safe="/:,")
    return f"{GIBS_ENDPOINT}?{query}"


def _layer_default_time(capabilities: str, layer: str) -> datetime:
    pattern = rf"<Name>{re.escape(layer)}</Name>.*?<Dimension[^>]+default=\"([^\"]+)\""
    match = re.search(pattern, capabilities, re.DOTALL)
    if not match:
        raise ValueError(f"no default time for {layer}")
    return datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))


def latest_common_time(capabilities: str) -> datetime:
    return min(
        _layer_default_time(capabilities, VISIBLE_LAYER),
        _layer_default_time(capabilities, IR_LAYER),
    )


def _valid_image(path: Path, expected_size=(4096, 2048)) -> bool:
    try:
        with Image.open(path) as image:
            return image.size == expected_size
    except (FileNotFoundError, OSError):
        return False


def _newest_cached_pair(cache: Path) -> tuple[datetime, Path, Path] | None:
    pairs = []
    for visible in cache.glob("himawari-*-visible.png"):
        stamp = visible.name.removeprefix("himawari-").removesuffix("-visible.png")
        infrared = cache / f"himawari-{stamp}-infrared.png"
        if _valid_image(visible) and _valid_image(infrared):
            try:
                timestamp = datetime.strptime(stamp, "%Y%m%dT%H%MZ").replace(tzinfo=UTC)
            except ValueError:
                continue
            pairs.append((timestamp, visible, infrared))
    return max(pairs, default=None, key=lambda item: item[0])


def _newest_cached_geocolor(cache: Path) -> tuple[datetime, Path, str] | None:
    candidates = []
    for path in cache.glob("cira-*-geocolor.png"):
        match = re.fullmatch(
            r"cira-(?:(gk2a|himawari)-)?(\d{8}T\d{4}Z)-geocolor\.png",
            path.name,
        )
        if match is None:
            continue
        satellite = match.group(1) or "himawari"
        stamp = match.group(2)
        if not _valid_image(path, expected_size=(2752, 2752)):
            continue
        try:
            timestamp = datetime.strptime(stamp, "%Y%m%dT%H%MZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        candidates.append((timestamp, path, satellite))
    return max(
        candidates,
        default=None,
        key=lambda item: (item[0], item[2] == "himawari"),
    )


def _acquire_gibs_layers(
    cache: Path,
    base: Path,
    lights: Path,
    terrain: Path | None,
) -> Observation:
    capabilities = _request(GIBS_CAPABILITIES).decode("utf-8")
    timestamp = latest_common_time(capabilities)
    stamp = timestamp.strftime("%Y%m%dT%H%MZ")
    visible = cache / f"himawari-{stamp}-visible.png"
    infrared = cache / f"himawari-{stamp}-infrared.png"
    if not _valid_image(visible):
        _atomic_download(
            _wms_url(VISIBLE_LAYER, timestamp=timestamp, image_format="image/png"),
            visible,
        )
    if not _valid_image(infrared):
        _atomic_download(
            _wms_url(IR_LAYER, timestamp=timestamp, image_format="image/png"),
            infrared,
        )
    return Observation(
        timestamp,
        visible,
        infrared,
        None,
        base,
        lights,
        "fresh",
        "NASA GIBS / JMA Himawari-9",
        140.7,
        terrain,
    )


CIRA_LATEST = "https://rammb-slider.cira.colostate.edu/data/json/{satellite}/full_disk/geocolor/latest_times.json"
CIRA_DATA = (
    "https://slider.cira.colostate.edu/data/rammb-slider6/slider/"
    "data_location_01"
)
CIRA_SOURCES = (
    ("himawari", "CIRA SLIDER / JMA Himawari-9", 140.7),
    ("gk2a", "CIRA SLIDER / KMA GK2A", 128.2),
)


def _acquire_cira_geocolor(cache: Path, satellite: str) -> tuple[datetime, Path]:
    payload = json.loads(_request(CIRA_LATEST.format(satellite=satellite)))
    now = datetime.now(UTC)
    last_error: Exception | None = None
    for raw_timestamp in payload["timestamps_int"][:12]:
        timestamp_text = str(raw_timestamp)
        timestamp = datetime.strptime(timestamp_text, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        if not (timestamp <= now and (now - timestamp).total_seconds() <= 3 * 3600):
            continue
        destination = cache / (
            f"cira-{satellite}-{timestamp.strftime('%Y%m%dT%H%MZ')}-geocolor.png"
        )
        if _valid_image(destination, expected_size=(2752, 2752)):
            return timestamp, destination

        date_path = timestamp.strftime("%Y/%m/%d")
        stamp = timestamp.strftime("%Y%m%d%H%M%S")
        canvas = Image.new("RGB", (2752, 2752))
        try:
            for row in range(4):
                for column in range(4):
                    url = (
                        f"{CIRA_DATA}/{date_path}/{satellite}/full_disk/geocolor/"
                        f"{stamp}/02/{row:03d}_{column:03d}.png"
                    )
                    tile = Image.open(io.BytesIO(_request(url))).convert("RGB")
                    if tile.size != (688, 688):
                        raise ValueError(f"unexpected CIRA tile size {tile.size}")
                    canvas.paste(tile, (column * 688, row * 688))
        except Exception as error:
            last_error = error
            continue

        tmp = destination.with_suffix(".tmp.png")
        canvas.save(tmp, optimize=True)
        tmp.replace(destination)
        old_files = sorted(
            cache.glob(f"cira-{satellite}-*-geocolor.png"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old in old_files[3:]:
            old.unlink(missing_ok=True)
        return timestamp, destination

    if last_error is not None:
        raise last_error
    raise ValueError(f"no recent CIRA {satellite} observation")


def acquire(cache: Path) -> Observation:
    cache.mkdir(parents=True, exist_ok=True)
    base = cache / "blue-marble.jpg"
    lights = cache / "city-lights.jpg"
    terrain = cache / "terrain-relief.jpg"
    if not _valid_image(base):
        _atomic_download(_wms_url(BASE_LAYER, timestamp=None, image_format="image/jpeg"), base)
    if not _valid_image(lights):
        _atomic_download(_wms_url(LIGHTS_LAYER, timestamp=None, image_format="image/jpeg"), lights)
    if not _valid_image(terrain):
        try:
            _atomic_download(
                _wms_url(TERRAIN_LAYER, timestamp=None, image_format="image/jpeg"),
                terrain,
            )
        except Exception:
            terrain = None

    # Separate JMA visible/IR layers are the primary source. They let the
    # renderer put city emission below live cloud and avoid GeoColor's synthetic
    # night palette becoming a flat grey/purple cloud sheet.
    try:
        return _acquire_gibs_layers(cache, base, lights, terrain)
    except Exception:
        pass

    # CIRA remains the high-resolution fallback when NASA GIBS is unavailable.
    for satellite, source, satellite_longitude in CIRA_SOURCES:
        try:
            timestamp, geocolor = _acquire_cira_geocolor(cache, satellite)
            return Observation(
                timestamp=timestamp,
                visible=geocolor,
                infrared=geocolor,
                geocolor=geocolor,
                base=base,
                lights=lights,
                status="fresh",
                source=source,
                satellite_longitude=satellite_longitude,
                terrain=terrain,
            )
        except Exception:
            pass

    cached_geocolor = _newest_cached_geocolor(cache)
    if cached_geocolor is not None:
        timestamp, geocolor, satellite = cached_geocolor
        if 0 <= (datetime.now(UTC) - timestamp).total_seconds() <= 3 * 3600:
            return Observation(
                timestamp,
                geocolor,
                geocolor,
                geocolor,
                base,
                lights,
                "cached",
                f"CIRA SLIDER / {'KMA GK2A' if satellite == 'gk2a' else 'JMA Himawari-9'} (cached)",
                128.2 if satellite == "gk2a" else 140.7,
                terrain,
            )

    cached = _newest_cached_pair(cache)
    if cached is not None:
        timestamp, visible, infrared = cached
        return Observation(
            timestamp, visible, infrared, None, base, lights, "cached",
            terrain=terrain,
        )
    geocolor_cached = _newest_cached_geocolor(cache)
    if geocolor_cached is None:
        raise RuntimeError("no live or cached satellite observation is available")
    timestamp, geocolor, satellite = geocolor_cached
    return Observation(
        timestamp, geocolor, geocolor, geocolor, base, lights, "cached",
        f"CIRA SLIDER / {'KMA GK2A' if satellite == 'gk2a' else 'JMA Himawari-9'} (cached)",
        128.2 if satellite == "gk2a" else 140.7,
        terrain,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
