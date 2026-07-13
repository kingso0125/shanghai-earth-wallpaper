from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PRESETS, RenderPreset
from .geometry import camera_grid, sample_equirectangular, sample_geostationary_focus_plate
from .lighting import daylight, sun_vector
from .sources import Observation, sha256
from .style import atmosphere, smoothstep, space_background


ACKNOWLEDGEMENT = (
    "Satellite imagery: Korea Meteorological Administration (KMA), "
    "Japan Meteorological Agency (JMA), NOAA/NESDIS, "
    "and Colorado State University/CIRA, processed through CIRA SLIDER. "
    "Static Earth and city-light imagery: NASA Earth Observatory/GIBS."
)


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0


def _grade_earth(rgb: np.ndarray) -> np.ndarray:
    luminance = rgb[..., :3].mean(axis=-1, keepdims=True)
    rgb = luminance + (rgb[..., :3] - luminance) * 1.08
    rgb = np.power(np.clip(rgb * 1.18 + 0.015, 0.0, 1.0), 0.86)
    rgb[..., 0] *= 0.98
    rgb[..., 1] *= 1.03
    rgb[..., 2] *= 1.08
    return np.clip(rgb, 0.0, 1.0)


def _cloud_alpha(visible: np.ndarray, infrared: np.ndarray, base: np.ndarray, day: np.ndarray):
    vis_luma = visible[..., :3].mean(axis=-1)
    base_luma = base[..., :3].mean(axis=-1)
    vis_signal = smoothstep(0.055, 0.42, vis_luma - base_luma * 0.31)
    vis_signal *= visible[..., 3]

    ir_rgb = infrared[..., :3]
    ir_luma = ir_rgb.mean(axis=-1)
    ir_sat = ir_rgb.max(axis=-1) - ir_rgb.min(axis=-1)
    ir_signal = np.maximum(smoothstep(0.10, 0.58, ir_sat), smoothstep(0.59, 0.88, ir_luma))
    ir_signal *= infrared[..., 3]
    cloud = vis_signal * day + ir_signal * (1.0 - day)
    return np.clip(np.power(cloud, 0.82) * 0.92, 0.0, 0.94)


def render_one(observation: Observation, preset: RenderPreset, destination: Path) -> None:
    base_map = _load(observation.base)
    lights_map = _load(observation.lights)

    lat, lon, mask, sz, vectors = camera_grid(preset)
    base = sample_equirectangular(base_map, lat, lon)
    lights = sample_equirectangular(lights_map, lat, lon)
    day = daylight(vectors, observation.timestamp)

    base_earth = _grade_earth(base)
    illumination = 0.25 + 0.75 * np.power(day[..., None], 0.54)
    base_earth *= illumination

    light_strength = np.power(lights[..., :3].mean(axis=-1), 1.45) * (1.0 - day) * 1.55
    light_color = np.array([1.0, 0.66, 0.28], dtype=np.float32)
    base_earth += light_strength[..., None] * light_color

    if observation.geocolor is not None:
        geocolor_map = _load(observation.geocolor)
        satellite, source_valid = sample_geostationary_focus_plate(
            geocolor_map, preset, observation.satellite_longitude
        )
        earth = np.power(
            np.clip(satellite[..., :3] * 1.08 + 0.012, 0.0, 1.0), 0.90
        )
        earth = np.where(source_valid[..., None], earth, base_earth)
    else:
        visible_map = _load(observation.visible)
        infrared_map = _load(observation.infrared)
        visible = sample_equirectangular(visible_map, lat, lon)
        infrared = sample_equirectangular(infrared_map, lat, lon)
        earth = base_earth
        cloud_alpha = _cloud_alpha(visible, infrared, base, day)
        cloud_light = 0.42 + 0.58 * np.power(day, 0.38)
        cloud_color = np.stack(
            [0.92 * cloud_light, 0.97 * cloud_light, 1.0 * cloud_light], axis=-1
        )
        earth = earth * (1.0 - cloud_alpha[..., None] * 0.78) + cloud_color * cloud_alpha[..., None]

    rim, halo = atmosphere(mask.astype(np.float32), sz, preset.size)
    earth += rim[..., None] * np.array([0.20, 0.58, 0.88], dtype=np.float32) * 0.50
    earth = np.clip(earth, 0.0, 1.0)

    output = space_background(preset.size, asset=Path("assets/space-background.jpg"))
    output += halo[..., None] * np.array([0.12, 0.48, 0.76], dtype=np.float32) * 0.56
    output = output * (1.0 - mask[..., None]) + earth * mask[..., None]
    output = np.clip(np.power(output, 0.97), 0.0, 1.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(output * 255), "RGB").save(destination, quality=96)


def render_pair(observation: Observation, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for preset in PRESETS:
        path = output / f"{preset.name}.jpg"
        render_one(observation, preset, path)
        artifacts[preset.name] = {"file": path.name, "sha256": sha256(path), "size": preset.size}

    sun = sun_vector(observation.timestamp)
    manifest = {
        "observation_utc": observation.timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "rendered_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": observation.source,
        "source_status": observation.status,
        "render_mode": (
            "fused_geostationary_plate_shanghai_meridian"
            if observation.geocolor is not None
            else "equirectangular_cloud_fallback"
        ),
        "observation_asset_sha256": sha256(
            observation.geocolor or observation.visible
        ),
        "target": {"name": "Shanghai", "latitude": 31.2304, "longitude": 121.4737},
        "view_center": {"latitude": 0.0, "longitude": 121.4737},
        "sun_vector": [round(float(value), 7) for value in sun],
        "acknowledgement": ACKNOWLEDGEMENT,
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
