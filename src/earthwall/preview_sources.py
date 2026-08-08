from __future__ import annotations

import urllib.parse
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from .config import (
    BASE_LAYER,
    GIBS_ENDPOINT,
    IR_LAYER,
    LIGHTS_LAYER,
    TERRAIN_LAYER,
    VISIBLE_LAYER,
)
from .sources import Observation, _request


PREVIEW_SIZE = (8192, 4096)


def _wms_url(layer: str, timestamp: datetime | None) -> str:
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/png",
        "TRANSPARENT": "true",
        "SRS": "EPSG:4326",
        "BBOX": "-180,-90,180,90",
        "WIDTH": str(PREVIEW_SIZE[0]),
        "HEIGHT": str(PREVIEW_SIZE[1]),
    }
    if timestamp is not None:
        params["TIME"] = timestamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{GIBS_ENDPOINT}?{urllib.parse.urlencode(params, safe='/:,')}"


def _valid(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            return image.size == PREVIEW_SIZE
    except (FileNotFoundError, OSError):
        return False


def _download(layer: str, timestamp: datetime | None, destination: Path) -> Path:
    if _valid(destination):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(_request(_wms_url(layer, timestamp), timeout=240))
    with Image.open(temporary) as image:
        image.verify()
    with Image.open(temporary) as image:
        if image.size != PREVIEW_SIZE:
            raise ValueError(f"unexpected preview asset size {image.size}")
    temporary.replace(destination)
    return destination


def upgrade_v2_observation(cache: Path, observation: Observation) -> Observation:
    """Upgrade an observation to the approved 8K V2 rendering assets."""
    v2_cache = cache / "cinematic-v2"
    base = _download(BASE_LAYER, None, v2_cache / "blue-marble-8k.png")
    lights = _download(LIGHTS_LAYER, None, v2_cache / "city-lights-8k.png")
    terrain = _download(TERRAIN_LAYER, None, v2_cache / "terrain-relief-8k.png")
    if observation.geocolor is not None:
        return replace(observation, base=base, lights=lights, terrain=terrain)

    stamp = observation.timestamp.astimezone(UTC).strftime("%Y%m%dT%H%MZ")
    visible = _download(
        VISIBLE_LAYER,
        observation.timestamp,
        v2_cache / f"himawari-{stamp}-visible-8k.png",
    )
    infrared = _download(
        IR_LAYER,
        observation.timestamp,
        v2_cache / f"himawari-{stamp}-infrared-8k.png",
    )
    return replace(
        observation,
        visible=visible,
        infrared=infrared,
        base=base,
        lights=lights,
        terrain=terrain,
    )


def upgrade_preview_observation(cache: Path, observation: Observation) -> Observation:
    return upgrade_v2_observation(cache, observation)
