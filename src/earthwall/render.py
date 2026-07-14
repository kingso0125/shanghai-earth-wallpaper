from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .config import SHANGHAI, RenderPreset, presets_for_location
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
    rgb[..., 0] *= 1.03
    rgb[..., 1] *= 1.01
    rgb[..., 2] *= 0.99
    return np.clip(rgb, 0.0, 1.0)


def _grade_geocolor(rgb: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Keep daytime GeoColor intact while neutralizing synthetic IR purple at night."""
    graded = np.power(np.clip(rgb[..., :3] * 1.08 + 0.012, 0.0, 1.0), 0.90)
    graded *= np.array([1.035, 1.01, 0.975], dtype=np.float32)
    graded = np.clip(graded, 0.0, 1.0)
    daytime = smoothstep(0.08, 0.72, day)[..., None]
    cloud_luminance = np.sum(
        graded * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
    )
    cloud_chroma = graded.max(axis=-1) - graded.min(axis=-1)
    cloud_highlight = (
        smoothstep(0.56, 0.92, cloud_luminance)
        * (1.0 - smoothstep(0.10, 0.28, cloud_chroma))
    )[..., None]
    # GeoColor already contains the observed cloud field. Roll off only its
    # neutral daytime highlights so dense systems keep texture instead of
    # clipping into an artificially opaque white mass.
    graded *= 1.0 - cloud_highlight * daytime * 0.14
    initial_luminance = np.sum(
        graded * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
        keepdims=True,
    )
    magenta_excess = np.minimum(graded[..., 0], graded[..., 2]) - graded[..., 1]
    magenta_mix = smoothstep(0.015, 0.075, magenta_excess)[..., None] * 0.94
    infrared_neutral = initial_luminance * np.array([0.94, 1.0, 1.06], dtype=np.float32)
    graded = graded * (1.0 - magenta_mix) + infrared_neutral * magenta_mix
    luminance = np.sum(
        graded * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
        keepdims=True,
    )
    neutral = np.repeat(luminance, 3, axis=-1)
    night_grade = neutral * np.array([0.88, 0.90, 0.92], dtype=np.float32)
    night_grade += (graded - neutral) * 0.03
    night_mix = smoothstep(0.06, 0.62, np.clip(1.0 - day, 0.0, 1.0))[..., None]
    return np.clip(graded * (1.0 - night_mix) + night_grade * night_mix, 0.0, 1.0)


def _night_cloud_alpha(satellite: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Extract neutral IR cloud texture without mistaking warm city lights for clouds."""
    rgb = satellite[..., :3]
    luminance = np.sum(
        rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1
    )
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    neutral_brightness = np.clip(luminance - chroma * 0.70, 0.0, 1.0)
    night = 1.0 - smoothstep(0.08, 0.72, day)
    return smoothstep(0.11, 0.66, neutral_brightness) * night


def _feather_coverage(alpha: np.ndarray, radius: float = 96.0) -> np.ndarray:
    """Hide rectangular WMS coverage edges without changing cloud texture."""
    image = Image.fromarray(np.uint8(np.clip(alpha, 0.0, 1.0) * 255), "L")
    blurred = (
        np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
        / 255.0
    )
    # Fade inward from the valid footprint. Using the raw blur would spread a
    # translucent rectangular haze outside the satellite coverage boundary.
    return smoothstep(0.54, 0.98, blurred)


def _city_light_signal(lights: np.ndarray) -> np.ndarray:
    """Isolate measured VIIRS light emission from the blue night-map background."""
    rgb = lights[..., :3]
    warm_radiance = np.clip(
        (rgb[..., 0] + rgb[..., 1]) * 0.5 - rgb[..., 2] * 0.56, 0.0, 1.0
    )
    core = np.power(smoothstep(0.045, 0.76, warm_radiance), 1.28)
    glow_image = Image.fromarray(np.uint8(np.clip(core, 0.0, 1.0) * 255), "L")
    glow = np.asarray(glow_image.filter(ImageFilter.GaussianBlur(1.8)), dtype=np.float32) / 255.0
    return np.clip(core * 0.70 + glow * 0.38, 0.0, 1.0)


def _blend_city_lights(
    earth: np.ndarray,
    lights: np.ndarray,
    day: np.ndarray,
    cloud_alpha: np.ndarray,
) -> np.ndarray:
    """Add real VIIRS lights on the night side, dimmed by the live cloud field."""
    night = smoothstep(0.14, 0.88, 1.0 - day)
    cloud_visibility = np.clip(1.0 - cloud_alpha * 0.86, 0.08, 1.0)
    strength = _city_light_signal(lights) * night * cloud_visibility
    light_color = np.array([1.0, 0.69, 0.32], dtype=np.float32)
    overlay = np.clip(strength[..., None] * light_color * 0.86, 0.0, 0.88)
    return np.clip(1.0 - (1.0 - earth) * (1.0 - overlay), 0.0, 1.0)


def _cloud_alpha(visible: np.ndarray, infrared: np.ndarray, base: np.ndarray, day: np.ndarray):
    vis_luma = visible[..., :3].mean(axis=-1)
    base_luma = base[..., :3].mean(axis=-1)
    vis_signal = smoothstep(0.055, 0.42, vis_luma - base_luma * 0.31)
    vis_signal *= _feather_coverage(visible[..., 3])

    ir_rgb = infrared[..., :3]
    ir_luma = ir_rgb.mean(axis=-1)
    ir_sat = ir_rgb.max(axis=-1) - ir_rgb.min(axis=-1)
    ir_signal = np.maximum(smoothstep(0.10, 0.58, ir_sat), smoothstep(0.59, 0.88, ir_luma))
    ir_signal *= _feather_coverage(infrared[..., 3])
    cloud = vis_signal * day + ir_signal * (1.0 - day)
    return np.clip(np.power(cloud, 0.82) * 0.92, 0.0, 0.94)


def _fallback_cloud_appearance(
    visible: np.ndarray, cloud_alpha: np.ndarray, day: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve observed cloud optical depth instead of painting a flat white veil."""
    reflectance = smoothstep(0.06, 0.74, visible[..., :3].mean(axis=-1))
    reflected_detail = np.power(reflectance, 0.72)

    day_tone = 0.40 + 0.48 * reflected_detail
    night_tone = 0.25 + 0.46 * np.power(cloud_alpha, 0.84)
    day_mix = smoothstep(0.08, 0.72, day)
    tone = night_tone * (1.0 - day_mix) + day_tone * day_mix

    day_color = np.array([1.02, 1.00, 0.97], dtype=np.float32)
    night_color = np.array([0.88, 0.92, 0.98], dtype=np.float32)
    color_balance = (
        night_color[None, None, :] * (1.0 - day_mix[..., None])
        + day_color[None, None, :] * day_mix[..., None]
    )
    cloud_color = np.clip(tone[..., None] * color_balance, 0.0, 0.92)

    # Thin clouds remain translucent; bright convective tops become denser and
    # brighter. Both are driven by the satellite reflectance texture.
    daytime_strength = 0.34 + 0.52 * reflected_detail
    nighttime_strength = 0.66 + 0.12 * cloud_alpha
    strength = nighttime_strength * (1.0 - day_mix) + daytime_strength * day_mix
    mix = np.clip(cloud_alpha * strength, 0.0, 0.82)
    return cloud_color, mix


def render_one(observation: Observation, preset: RenderPreset, destination: Path) -> None:
    base_map = _load(observation.base)
    lights_map = _load(observation.lights)

    lat, lon, mask, sz, vectors = camera_grid(preset)
    base = sample_equirectangular(base_map, lat, lon)
    lights = sample_equirectangular(lights_map, lat, lon)
    day = daylight(vectors, observation.timestamp)

    base_earth = _grade_earth(base)
    illumination = 0.28 + 0.72 * np.power(day[..., None], 0.54)
    base_earth *= illumination

    cloud_alpha = np.zeros_like(day)

    if observation.geocolor is not None:
        geocolor_map = _load(observation.geocolor)
        satellite, source_valid = sample_geostationary_focus_plate(
            geocolor_map, preset, observation.satellite_longitude
        )
        day_earth = _grade_geocolor(satellite, day)
        cloud_alpha = _night_cloud_alpha(satellite, day)
        cloud_luminance = np.sum(
            satellite[..., :3]
            * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
            axis=-1,
        )
        cloud_brightness = 0.28 + 0.62 * smoothstep(0.08, 0.78, cloud_luminance)
        cloud_color = cloud_brightness[..., None] * np.array(
            [0.92, 0.91, 0.90], dtype=np.float32
        )
        night_luminance = np.sum(
            base_earth * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
            axis=-1,
            keepdims=True,
        )
        night_surface = night_luminance + (base_earth - night_luminance) * 0.58
        ocean_weight = smoothstep(
            0.008,
            0.075,
            base_earth[..., 2] - np.maximum(base_earth[..., 0], base_earth[..., 1]),
        )[..., None]
        land_tint = np.array([1.08, 1.00, 0.91], dtype=np.float32)
        ocean_tint = np.array([0.96, 1.00, 1.07], dtype=np.float32)
        night_surface *= land_tint * (1.0 - ocean_weight) + ocean_tint * ocean_weight
        night_surface = np.power(np.clip(night_surface * 1.24 + 0.008, 0.0, 1.0), 0.90)
        night_earth = night_surface * (1.0 - cloud_alpha[..., None] * 0.70)
        night_earth += cloud_color * cloud_alpha[..., None] * 0.70
        day_mix = smoothstep(0.08, 0.72, day)[..., None]
        earth = night_earth * (1.0 - day_mix) + day_earth * day_mix
        earth = np.where(source_valid[..., None], earth, base_earth)
    else:
        visible_map = _load(observation.visible)
        infrared_map = _load(observation.infrared)
        visible = sample_equirectangular(visible_map, lat, lon)
        infrared = sample_equirectangular(infrared_map, lat, lon)
        earth = base_earth
        cloud_alpha = _cloud_alpha(visible, infrared, base, day)
        cloud_color, cloud_mix = _fallback_cloud_appearance(visible, cloud_alpha, day)
        earth = earth * (1.0 - cloud_mix[..., None]) + cloud_color * cloud_mix[..., None]

    earth = _blend_city_lights(earth, lights, day, cloud_alpha)

    rim, halo = atmosphere(mask.astype(np.float32), sz, preset.size)
    earth += rim[..., None] * np.array([0.20, 0.58, 0.88], dtype=np.float32) * 0.50
    earth = np.clip(earth, 0.0, 1.0)

    output = space_background(preset.size, asset=Path("assets/space-background.jpg"))
    output += halo[..., None] * np.array([0.12, 0.48, 0.76], dtype=np.float32) * 0.56
    output = output * (1.0 - mask[..., None]) + earth * mask[..., None]
    output = np.clip(np.power(output, 0.97), 0.0, 1.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(output * 255), "RGB").save(destination, quality=96)


def render_pair(
    observation: Observation,
    output: Path,
    target_latitude: float = SHANGHAI[0],
    target_longitude: float = SHANGHAI[1],
    target_name: str = "Shanghai",
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    presets = presets_for_location(target_latitude, target_longitude)
    for preset in presets:
        path = output / f"{preset.name}.jpg"
        render_one(observation, preset, path)
        artifacts[preset.name] = {
            "file": path.name,
            "sha256": sha256(path),
            "size": preset.size,
            "view_center": {
                "latitude": preset.target_lat,
                "longitude": preset.target_lon,
            },
        }

    sun = sun_vector(observation.timestamp)
    manifest = {
        "observation_utc": observation.timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "rendered_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": observation.source,
        "source_status": observation.status,
        "render_mode": (
            "fused_geostationary_plate_location_centered"
            if observation.geocolor is not None
            else "equirectangular_cloud_fallback"
        ),
        "observation_asset_sha256": sha256(
            observation.geocolor or observation.visible
        ),
        "target": {
            "name": target_name,
            "latitude": target_latitude,
            "longitude": target_longitude,
        },
        "view_center": {
            "lock": artifacts["lock"]["view_center"],
            "home": artifacts["home"]["view_center"],
        },
        "sun_vector": [round(float(value), 7) for value in sun],
        "acknowledgement": ACKNOWLEDGEMENT,
        "night_lights": {
            "source": "NASA GIBS VIIRS_CityLights_2012",
            "mode": "night-side, cloud-occluded",
            "temporal_model": "observed static baseline; sun and cloud masks are current",
        },
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
