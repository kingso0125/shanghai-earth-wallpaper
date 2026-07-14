from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .config import SHANGHAI, RenderPreset, mac_presets_for_location, presets_for_location
from .geometry import camera_grid, sample_equirectangular, sample_geostationary_focus_plate
from .lighting import daylight, sun_vector
from .sources import Observation, sha256
from .style import atmosphere, smoothstep, space_background


ACKNOWLEDGEMENT = (
    "Satellite imagery: Korea Meteorological Administration (KMA), "
    "Japan Meteorological Agency (JMA), NOAA/NESDIS, "
    "and Colorado State University/CIRA, processed through CIRA SLIDER. "
    "Static Earth and city-light imagery: NASA Earth Observatory/GIBS. "
    "Terrain relief: NASA/METI ASTER GDEM through NASA GIBS."
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


def _apple_natural_grade(
    rgb: np.ndarray, day: np.ndarray, sz: np.ndarray
) -> np.ndarray:
    """Apply a warm, filmic display transform without changing observed geography."""
    luminance = np.sum(
        rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
        keepdims=True,
    )
    rgb = luminance + (rgb - luminance) * 1.03
    rgb = np.power(np.clip(rgb, 0.0, 1.0), 0.93)

    daylight_mix = smoothstep(0.05, 0.78, day)[..., None]
    night_balance = np.array([1.025, 1.005, 0.985], dtype=np.float32)
    day_balance = np.array([1.045, 1.018, 0.965], dtype=np.float32)
    rgb *= night_balance * (1.0 - daylight_mix) + day_balance * daylight_mix

    # Retain a physically legible day/night separation after the display tone map.
    # Daylight opens the midtones; the night side remains deep enough for VIIRS
    # lights to read clearly without inventing any illumination.
    exposure = 0.74 + daylight_mix * 0.34
    rgb *= exposure
    rgb += daylight_mix * rgb * (1.0 - rgb) * 0.08

    display_luminance = np.sum(
        rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
        keepdims=True,
    )
    shadow_lift = (1.0 - smoothstep(0.24, 0.58, display_luminance)) * daylight_mix * 0.055
    rgb += (1.0 - rgb) * shadow_lift
    highlight_rolloff = smoothstep(0.58, 0.92, display_luminance) * daylight_mix * 0.10
    rgb *= 1.0 - highlight_rolloff

    # Preserve the observed ocean mask, then restore the richer blue response
    # that is lost when satellite RGB is compressed for display.
    ocean = smoothstep(
        0.012,
        0.115,
        rgb[..., 2] - np.maximum(rgb[..., 0], rgb[..., 1]),
    )[..., None]
    ocean *= daylight_mix
    ocean_balance = np.array([0.86, 1.30, 1.34], dtype=np.float32)
    rgb *= 1.0 - ocean + ocean * ocean_balance
    rgb += ocean * np.array([0.0, 0.050, 0.080], dtype=np.float32)

    vegetation = smoothstep(
        0.012,
        0.125,
        rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]),
    )[..., None]
    vegetation *= daylight_mix * (1.0 - ocean)
    vegetation_balance = np.array([0.88, 1.10, 1.13], dtype=np.float32)
    rgb *= 1.0 - vegetation + vegetation * vegetation_balance
    rgb += vegetation * np.array([0.0, 0.012, 0.022], dtype=np.float32)

    warm_difference = rgb[..., 0] - rgb[..., 2]
    gold_difference = rgb[..., 1] - rgb[..., 2]
    desert = (
        smoothstep(0.055, 0.24, warm_difference)
        * smoothstep(0.018, 0.16, gold_difference)
        * smoothstep(0.20, 0.62, display_luminance[..., 0])
    )[..., None]
    desert *= daylight_mix * (1.0 - ocean) * (1.0 - vegetation)
    desert_balance = np.array([1.08, 1.09, 0.82], dtype=np.float32)
    rgb *= 1.0 - desert + desert * desert_balance
    rgb += desert * np.array([0.018, 0.024, 0.0], dtype=np.float32)

    final_luminance = np.sum(
        rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
        keepdims=True,
    )
    final_chroma = rgb.max(axis=-1, keepdims=True) - rgb.min(axis=-1, keepdims=True)
    saturation = 1.0 + smoothstep(0.025, 0.18, final_chroma) * daylight_mix * 0.12
    rgb = final_luminance + (rgb - final_luminance) * saturation

    daylight_haze = daylight_mix * 0.025
    haze_tone = np.array([0.56, 0.70, 0.75], dtype=np.float32)
    rgb = rgb * (1.0 - daylight_haze) + haze_tone * daylight_haze

    # A small view-angle haze creates depth at the limb instead of a hard blue outline.
    limb = np.power(np.clip(1.0 - sz, 0.0, 1.0), 2.4)[..., None]
    haze_color = np.array([0.31, 0.53, 0.67], dtype=np.float32)
    rgb = rgb * (1.0 - limb * 0.10) + haze_color * limb * 0.10
    return np.clip(rgb, 0.0, 1.0)


def _grade_geocolor(rgb: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Keep daytime GeoColor intact while neutralizing synthetic IR purple at night."""
    graded = np.power(np.clip(rgb[..., :3] * 1.035 + 0.010, 0.0, 1.0), 0.95)
    luminance = np.sum(
        graded * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
        keepdims=True,
    )
    graded = luminance + (graded - luminance) * 0.94
    graded *= np.array([1.035, 1.015, 0.975], dtype=np.float32)
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
    graded *= 1.0 - cloud_highlight * daytime * 0.19
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


def _day_cloud_alpha(satellite: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Estimate neutral daytime cloud coverage only for local detail enhancement."""
    rgb = satellite[..., :3]
    luminance = np.sum(
        rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1
    )
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    neutral = 1.0 - smoothstep(0.09, 0.26, chroma)
    daylight = smoothstep(0.08, 0.72, day)
    return smoothstep(0.30, 0.82, luminance) * neutral * daylight


def _sharpen_cloud_texture(
    earth: np.ndarray, cloud_alpha: np.ndarray, day: np.ndarray
) -> np.ndarray:
    """Sharpen observed cloud texture locally without changing cloud geometry."""
    image = Image.fromarray(np.uint8(np.clip(earth, 0.0, 1.0) * 255), "RGB")
    blurred = np.asarray(
        image.filter(ImageFilter.GaussianBlur(1.25)), dtype=np.float32
    ) / 255.0
    detail = earth - blurred
    daylight_strength = 0.16 + 0.84 * smoothstep(0.08, 0.70, day)
    strength = (
        smoothstep(0.12, 0.78, cloud_alpha) * daylight_strength
    )[..., None] * 0.52
    return np.clip(earth + detail * strength, 0.0, 1.0)


def _apply_terrain_relief(
    earth: np.ndarray,
    relief: np.ndarray,
    cloud_alpha: np.ndarray,
    day: np.ndarray,
) -> np.ndarray:
    """Add restrained ASTER terrain shading without embossing oceans or clouds."""
    gray = np.sum(
        relief[..., :3] * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
    )
    gray_image = Image.fromarray(np.uint8(np.clip(gray, 0.0, 1.0) * 255), "L")
    fine_local = np.asarray(
        gray_image.filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float32
    ) / 255.0
    broad_local = np.asarray(
        gray_image.filter(ImageFilter.GaussianBlur(11.0)), dtype=np.float32
    ) / 255.0
    terrain_detail = np.clip(
        (gray - fine_local) * 2.0 + (gray - broad_local) * 1.35,
        -0.18,
        0.18,
    )

    land = smoothstep(0.50, 0.64, gray)
    land_image = Image.fromarray(np.uint8(land * 255), "L")
    land_core = np.asarray(
        land_image.filter(ImageFilter.MinFilter(11)), dtype=np.float32
    ) / 255.0
    cloud_visibility = 1.0 - smoothstep(0.10, 0.72, cloud_alpha)
    light_visibility = 0.34 + 0.66 * smoothstep(0.04, 0.72, day)
    strength = land_core * cloud_visibility * light_visibility
    return np.clip(
        earth * (1.0 + terrain_detail[..., None] * strength[..., None]),
        0.0,
        1.0,
    )


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
    core = np.power(smoothstep(0.040, 0.70, warm_radiance), 1.22)
    glow_image = Image.fromarray(np.uint8(np.clip(core, 0.0, 1.0) * 255), "L")
    local = np.asarray(
        glow_image.filter(ImageFilter.GaussianBlur(1.15)), dtype=np.float32
    ) / 255.0
    crisp_core = np.clip(core + (core - local) * 0.55, 0.0, 1.0)
    return np.clip(crisp_core * 0.90 + local * 0.18, 0.0, 1.0)


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
    overlay = np.clip(strength[..., None] * light_color * 1.02, 0.0, 0.92)
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
    # Visible cloud reflectance fades rapidly near dusk. Infrared remains valid
    # around the clock, so retain it as a restrained daytime coverage floor.
    day_cloud = np.maximum(vis_signal, ir_signal * 0.72)
    cloud = day_cloud * day + ir_signal * (1.0 - day)
    return np.clip(np.power(cloud, 0.82) * 0.92, 0.0, 0.94)


def _fallback_cloud_appearance(
    visible: np.ndarray,
    infrared: np.ndarray,
    cloud_alpha: np.ndarray,
    day: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve observed cloud optical depth instead of painting a flat white veil."""
    reflectance = smoothstep(0.06, 0.74, visible[..., :3].mean(axis=-1))
    reflected_detail = np.power(reflectance, 0.72)

    ir_rgb = infrared[..., :3]
    ir_luma = ir_rgb.mean(axis=-1)
    ir_chroma = ir_rgb.max(axis=-1) - ir_rgb.min(axis=-1)
    thermal = np.maximum(
        smoothstep(0.10, 0.62, ir_chroma),
        smoothstep(0.55, 0.91, ir_luma),
    )
    thermal_image = Image.fromarray(np.uint8(thermal * 255), "L")
    thermal_local = np.asarray(
        thermal_image.filter(ImageFilter.GaussianBlur(1.15)), dtype=np.float32
    ) / 255.0
    thermal_detail = np.clip(thermal + (thermal - thermal_local) * 0.42, 0.0, 1.0)
    observed_detail = np.maximum(reflected_detail, thermal_detail * 0.82)

    day_tone = 0.38 + 0.50 * observed_detail
    night_tone = 0.12 + 0.30 * observed_detail
    day_mix = smoothstep(0.08, 0.72, day)
    tone = night_tone * (1.0 - day_mix) + day_tone * day_mix

    day_color = np.array([1.02, 1.00, 0.97], dtype=np.float32)
    night_color = np.array([0.66, 0.73, 0.82], dtype=np.float32)
    color_balance = (
        night_color[None, None, :] * (1.0 - day_mix[..., None])
        + day_color[None, None, :] * day_mix[..., None]
    )
    cloud_color = np.clip(tone[..., None] * color_balance, 0.0, 0.92)

    # Thin clouds remain translucent; bright convective tops become denser and
    # brighter. Both are driven by the satellite reflectance texture.
    daytime_strength = 0.32 + 0.52 * observed_detail
    nighttime_strength = 0.30 + 0.30 * observed_detail
    strength = nighttime_strength * (1.0 - day_mix) + daytime_strength * day_mix
    alpha_image = Image.fromarray(np.uint8(np.clip(cloud_alpha, 0.0, 1.0) * 255), "L")
    softened_alpha = np.asarray(
        alpha_image.filter(ImageFilter.GaussianBlur(1.15)), dtype=np.float32
    ) / 255.0
    appearance_alpha = softened_alpha * (1.0 - day_mix) + cloud_alpha * day_mix
    mix = np.clip(appearance_alpha * strength, 0.0, 0.82)
    return cloud_color, mix


def render_one(
    observation: Observation,
    preset: RenderPreset,
    destination: Path,
    *,
    lighting_time: datetime | None = None,
    background_asset: Path | None = Path("assets/space-background.jpg"),
    jpeg_quality: int = 96,
) -> None:
    base_map = _load(observation.base)
    lights_map = _load(observation.lights)
    terrain_map = _load(observation.terrain) if observation.terrain else None

    lat, lon, mask, sz, vectors = camera_grid(preset)
    base = sample_equirectangular(base_map, lat, lon)
    lights = sample_equirectangular(lights_map, lat, lon)
    terrain = (
        sample_equirectangular(terrain_map, lat, lon)
        if terrain_map is not None
        else None
    )
    day = daylight(vectors, lighting_time or datetime.now(UTC))

    base_earth = _grade_earth(base)
    illumination = 0.28 + 0.72 * np.power(day[..., None], 0.54)
    base_earth *= illumination

    cloud_alpha = np.zeros_like(day)
    cloud_detail_alpha = np.zeros_like(day)

    if observation.geocolor is not None:
        geocolor_map = _load(observation.geocolor)
        satellite, source_valid = sample_geostationary_focus_plate(
            geocolor_map, preset, observation.satellite_longitude
        )
        day_earth = _grade_geocolor(satellite, day)
        cloud_alpha = _night_cloud_alpha(satellite, day)
        cloud_detail_alpha = np.maximum(
            cloud_alpha, _day_cloud_alpha(satellite, day)
        )
        cloud_luminance = np.sum(
            satellite[..., :3]
            * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
            axis=-1,
        )
        cloud_brightness = 0.15 + 0.31 * smoothstep(0.08, 0.78, cloud_luminance)
        cloud_color = cloud_brightness[..., None] * np.array(
            [0.68, 0.75, 0.84], dtype=np.float32
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
        night_earth = night_surface * (1.0 - cloud_alpha[..., None] * 0.52)
        night_earth += cloud_color * cloud_alpha[..., None] * 0.48
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
        cloud_detail_alpha = cloud_alpha
        cloud_color, cloud_mix = _fallback_cloud_appearance(
            visible, infrared, cloud_alpha, day
        )
        earth = earth * (1.0 - cloud_mix[..., None]) + cloud_color * cloud_mix[..., None]

    earth = _apple_natural_grade(earth, day, sz)
    if terrain is not None:
        earth = _apply_terrain_relief(earth, terrain, cloud_detail_alpha, day)
    earth = _sharpen_cloud_texture(earth, cloud_detail_alpha, day)
    earth = _blend_city_lights(earth, lights, day, cloud_alpha)

    rim, halo = atmosphere(mask.astype(np.float32), sz, preset.size)
    earth += rim[..., None] * np.array([0.25, 0.62, 0.84], dtype=np.float32) * 0.48
    earth = np.clip(earth, 0.0, 1.0)

    output = space_background(preset.size, asset=background_asset)
    output += halo[..., None] * np.array([0.18, 0.53, 0.76], dtype=np.float32) * 0.66
    output = output * (1.0 - mask[..., None]) + earth * mask[..., None]
    output = np.clip(np.power(output, 0.97), 0.0, 1.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(output * 255), "RGB").save(
        destination, quality=jpeg_quality, subsampling=0
    )


def render_pair(
    observation: Observation,
    output: Path,
    target_latitude: float = SHANGHAI[0],
    target_longitude: float = SHANGHAI[1],
    target_name: str = "Shanghai",
    lighting_time: datetime | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    lighting_time = lighting_time or datetime.now(UTC)
    artifacts = {}
    presets = presets_for_location(target_latitude, target_longitude)
    for preset in presets:
        path = output / f"{preset.name}.jpg"
        render_one(observation, preset, path, lighting_time=lighting_time)
        artifacts[preset.name] = {
            "file": path.name,
            "sha256": sha256(path),
            "size": preset.size,
            "view_center": {
                "latitude": preset.target_lat,
                "longitude": preset.target_lon,
            },
        }

    sun = sun_vector(lighting_time)
    manifest = {
        "observation_utc": observation.timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "lighting_utc": lighting_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
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


def render_mac_pair(
    observation: Observation,
    output: Path,
    target_latitude: float = SHANGHAI[0],
    target_longitude: float = SHANGHAI[1],
    target_name: str = "Shanghai",
    lighting_time: datetime | None = None,
) -> dict:
    """Render native-resolution Mac assets without touching the phone outputs."""
    output.mkdir(parents=True, exist_ok=True)
    lighting_time = lighting_time or datetime.now(UTC)
    artifacts = {}
    for preset in mac_presets_for_location(target_latitude, target_longitude):
        path = output / f"mac-{preset.name}.jpg"
        render_one(
            observation,
            preset,
            path,
            lighting_time=lighting_time,
            background_asset=None,
            jpeg_quality=98,
        )
        artifacts[preset.name] = {
            "file": path.name,
            "sha256": sha256(path),
            "size": preset.size,
            "view_center": {
                "latitude": preset.target_lat,
                "longitude": preset.target_lon,
            },
        }

    manifest = {
        "profile": "mac",
        "observation_utc": observation.timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "lighting_utc": lighting_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "rendered_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": observation.source,
        "source_status": observation.status,
        "render_mode": (
            "fused_geostationary_plate_location_centered"
            if observation.geocolor is not None
            else "equirectangular_cloud_fallback"
        ),
        "target": {
            "name": target_name,
            "latitude": target_latitude,
            "longitude": target_longitude,
        },
        "acknowledgement": ACKNOWLEDGEMENT,
        "artifacts": artifacts,
    }
    (output / "mac-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
