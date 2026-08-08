from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .config import SHANGHAI
from .geometry import sample_equirectangular, sample_geostationary_focus_plate
from .lighting import sun_vector
from .render import (
    ACKNOWLEDGEMENT,
    _city_light_signal,
    _cloud_alpha,
    _day_cloud_alpha,
    _night_cloud_alpha,
    _thermal_cloud_texture,
)
from .sources import Observation, sha256
from .style import smoothstep, space_background


@dataclass(frozen=True)
class V2Preset:
    name: str
    size: tuple[int, int]
    center_px: tuple[float, float]
    globe_radius_px: float
    camera_distance: float
    target_lat: float
    target_lon: float


V2_LOCK = V2Preset(
    "lock",
    (1320, 2868),
    (660.0, 1480.0),
    620.0,
    4.8,
    SHANGHAI[0] - 6.0,
    SHANGHAI[1],
)
V2_HOME = V2Preset(
    "home",
    (1320, 2868),
    (660.0, 2735.0),
    1510.0,
    2.75,
    5.0,
    SHANGHAI[1],
)
V2_MAC_LOCK = V2Preset(
    "lock",
    (2560, 1664),
    (1280.0, 900.0),
    560.0,
    4.8,
    SHANGHAI[0] - 6.0,
    SHANGHAI[1],
)
V2_MAC_HOME = V2Preset(
    "home",
    (2560, 1664),
    (1280.0, 1880.0),
    1290.0,
    2.75,
    5.0,
    SHANGHAI[1],
)


@dataclass(frozen=True)
class CloudProperties:
    opacity: np.ndarray
    radiance: np.ndarray


def preview_output(name: str) -> Path:
    return Path("output/preview-v2") / f"{name}.jpg"


def presets_for_location(latitude: float, longitude: float) -> tuple[V2Preset, ...]:
    return (
        replace(V2_LOCK, target_lat=np.clip(latitude - 6.0, -90.0, 90.0), target_lon=longitude),
        replace(V2_HOME, target_lon=longitude),
    )


def presets_for_mac_location(latitude: float, longitude: float) -> tuple[V2Preset, ...]:
    return (
        replace(V2_MAC_LOCK, target_lat=np.clip(latitude - 6.0, -90.0, 90.0), target_lon=longitude),
        replace(V2_MAC_HOME, target_lon=longitude),
    )


def _basis(latitude: float, longitude: float) -> tuple[np.ndarray, ...]:
    lat0 = np.deg2rad(latitude)
    lon0 = np.deg2rad(longitude)
    forward = np.array(
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        dtype=np.float32,
    )
    east = np.array([-np.sin(lon0), np.cos(lon0), 0.0], dtype=np.float32)
    north = np.array(
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        dtype=np.float32,
    )
    return forward, east, north


def perspective_camera_grid(preset: V2Preset):
    """Intersect perspective camera rays with a unit sphere.

    The projected silhouette keeps the requested pixel radius, while camera
    distance controls how much of the hemisphere is visible. This gives Home a
    lower-orbit feeling without altering geographic coordinates.
    """
    width, height = preset.size
    cx, cy = preset.center_px
    radius = preset.globe_radius_px
    yy, xx = np.mgrid[0:height, 0:width]
    screen_x = (xx.astype(np.float32) - cx) / radius
    screen_y = (yy.astype(np.float32) - cy) / radius
    rho2 = screen_x * screen_x + screen_y * screen_y
    visible = rho2 <= 1.0

    distance = np.float32(preset.camera_distance)
    slope = np.float32(1.0 / np.sqrt(distance * distance - 1.0))
    ray = np.stack(
        (screen_x * slope, -screen_y * slope, np.ones_like(screen_x)), axis=-1
    )
    ray /= np.maximum(np.linalg.norm(ray, axis=-1, keepdims=True), 1e-6)
    toward_center = ray[..., 2] * distance
    discriminant = np.clip(toward_center * toward_center - (distance * distance - 1.0), 0.0, None)
    travel = toward_center - np.sqrt(discriminant)
    point = ray * travel[..., None]
    normal_local = point - np.array([0.0, 0.0, distance], dtype=np.float32)
    normal_local /= np.maximum(np.linalg.norm(normal_local, axis=-1, keepdims=True), 1e-6)

    forward, east, north = _basis(preset.target_lat, preset.target_lon)
    vectors = (
        normal_local[..., 0, None] * east
        + normal_local[..., 1, None] * north
        - normal_local[..., 2, None] * forward
    )
    vectors /= np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-6)

    view_local = -point
    view = (
        view_local[..., 0, None] * east
        + view_local[..., 1, None] * north
        - view_local[..., 2, None] * forward
    )
    view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-6)
    view_cos = np.clip(np.sum(vectors * view, axis=-1), 0.0, 1.0)
    lat = np.arcsin(np.clip(vectors[..., 2], -1.0, 1.0))
    lon = np.arctan2(vectors[..., 1], vectors[..., 0])
    return lat, lon, visible, view_cos, vectors.astype(np.float32), view.astype(np.float32)


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0


def _blur_scalar(values: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.uint8(np.clip(values, 0.0, 1.0) * 255), "L")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def _blur_rgb(values: np.ndarray, radius: float) -> np.ndarray:
    image = Image.fromarray(np.uint8(np.clip(values, 0.0, 1.0) * 255), "RGB")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def _shift(values: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    source_x0 = max(0, -dx)
    source_x1 = values.shape[1] - max(0, dx)
    source_y0 = max(0, -dy)
    source_y1 = values.shape[0] - max(0, dy)
    target_x0 = max(0, dx)
    target_x1 = values.shape[1] - max(0, -dx)
    target_y0 = max(0, dy)
    target_y1 = values.shape[0] - max(0, -dy)
    if source_x1 > source_x0 and source_y1 > source_y0:
        shifted[target_y0:target_y1, target_x0:target_x1] = values[
            source_y0:source_y1, source_x0:source_x1
        ]
    return shifted


def _apple_night_ir_cloud(
    infrared: np.ndarray,
    surface: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the colour-enhanced Band 13 plate into continuous night clouds.

    GIBS encodes cold cloud tops with colour and warmer structures in grey.
    Preserve both as a continuous field; thresholding the palette creates the
    flat paper-like cloud patches that this profile is designed to avoid.
    """
    rgb = np.clip(infrared[..., :3], 0.0, 1.0)
    luminance = np.sum(
        rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1
    )
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    medium_luminance = _blur_scalar(luminance, 2.2)
    broad_luminance = _blur_scalar(luminance, 18.0)
    cold_anomaly = np.clip(medium_luminance - broad_luminance, 0.0, 0.24) / 0.24

    coloured_top = np.power(smoothstep(0.018, 0.42, _blur_scalar(chroma, 1.1)), 1.18)
    neutral_top = np.power(smoothstep(0.025, 0.62, cold_anomaly), 0.90)
    if surface is not None:
        # A single thermal band cannot intrinsically distinguish every cold
        # cloud from cold high terrain. Suppress weak neutral anomalies over
        # land while retaining coloured/strong cold cloud tops and all ocean
        # cloud structure. This prevents Siberia and the Tibetan Plateau from
        # becoming a false cloud sheet at night.
        surface_rgb = np.clip(surface[..., :3], 0.0, 1.0)
        blue_dominance = surface_rgb[..., 2] - np.maximum(
            surface_rgb[..., 0], surface_rgb[..., 1]
        )
        ocean = _blur_scalar(smoothstep(0.015, 0.060, blue_dominance), 1.2)
        land = 1.0 - ocean
        land_cloud_evidence = smoothstep(0.075, 0.34, cold_anomaly)
        coloured_top *= ocean + land * (0.06 + land_cloud_evidence * 0.94)
        neutral_top *= ocean + land * (0.035 + land_cloud_evidence * 0.965)
    raw = np.maximum(coloured_top * 0.88, neutral_top * 0.72)
    medium = _blur_scalar(raw, 1.5)
    broad = _blur_scalar(raw, 8.0)
    texture = np.clip(
        raw + (raw - medium) * 0.24 + (medium - broad) * 0.16,
        0.0,
        1.0,
    )
    alpha = np.clip(raw * 0.66 + medium * 0.18 + broad * 0.08, 0.0, 0.88)
    return alpha.astype(np.float32), texture.astype(np.float32)


def _linear(rgb: np.ndarray) -> np.ndarray:
    return np.power(np.clip(rgb, 0.0, 1.0), 2.2)


def _display(rgb: np.ndarray) -> np.ndarray:
    return np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2)


def aces_tonemap(rgb: np.ndarray) -> np.ndarray:
    rgb = np.maximum(rgb, 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e), 0.0, 1.0)


def _material_albedo(base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.clip(base[..., :3], 0.0, 1.0)
    luminance = np.sum(rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1)
    ocean = smoothstep(0.012, 0.115, rgb[..., 2] - np.maximum(rgb[..., 0], rgb[..., 1]))
    ocean = _blur_scalar(ocean, 1.2)
    land = 1.0 - ocean

    coast = np.clip(ocean * (1.0 - _blur_scalar(ocean, 16.0)) * 5.0, 0.0, 1.0)
    deep_ocean = np.array([0.045, 0.275, 0.455], dtype=np.float32)
    shallow_ocean = np.array([0.070, 0.465, 0.585], dtype=np.float32)
    ocean_tone = deep_ocean * (1.0 - coast[..., None]) + shallow_ocean * coast[..., None]
    ocean_tone *= (0.76 + luminance[..., None] * 0.72)
    rgb = rgb * (1.0 - ocean[..., None] * 0.82) + ocean_tone * ocean[..., None] * 0.82

    vegetation = smoothstep(0.018, 0.13, rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])) * land
    warm = rgb[..., 0] - rgb[..., 2]
    desert = smoothstep(0.045, 0.22, warm) * land * (1.0 - vegetation)
    teal_green = np.array([0.075, 0.325, 0.235], dtype=np.float32)
    golden_land = np.array([0.49, 0.335, 0.145], dtype=np.float32)
    rgb = rgb * (1.0 - vegetation[..., None] * 0.14) + teal_green * vegetation[..., None] * 0.14
    rgb = rgb * (1.0 - desert[..., None] * 0.15) + golden_land * desert[..., None] * 0.15

    lum = np.sum(rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1, keepdims=True)
    rgb = lum + (rgb - lum) * 1.025
    rgb *= np.array([1.025, 1.010, 0.975], dtype=np.float32)
    ocean_soft = _blur_rgb(np.clip(rgb, 0.0, 1.0), 0.85)
    rgb = rgb * (1.0 - ocean[..., None] * 0.34) + ocean_soft * ocean[..., None] * 0.34
    return np.clip(rgb, 0.0, 1.0), ocean, land


def _relief_normals(
    vectors: np.ndarray,
    relief: np.ndarray | None,
    land: np.ndarray,
    preset: V2Preset,
) -> np.ndarray:
    if relief is None:
        return vectors
    gray = np.sum(
        relief[..., :3] * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1
    )
    broad = _blur_scalar(gray, 11.0)
    height = np.clip(gray - broad, -0.16, 0.16)
    gradient_y, gradient_x = np.gradient(height)
    forward, east, north = _basis(preset.target_lat, preset.target_lon)
    strength = 22.0 * land * smoothstep(0.18, 0.88, np.abs(height) + 0.12)
    perturbed = (
        vectors
        - gradient_x[..., None] * east * strength[..., None]
        + gradient_y[..., None] * north * strength[..., None]
    )
    return perturbed / np.maximum(np.linalg.norm(perturbed, axis=-1, keepdims=True), 1e-6)


def _surface_radiance(
    albedo: np.ndarray,
    ocean: np.ndarray,
    normal: np.ndarray,
    geometric_normal: np.ndarray,
    view: np.ndarray,
    sun: np.ndarray,
    *,
    apple_night: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    solar_cos = np.sum(geometric_normal * sun, axis=-1)
    diffuse = np.power(np.clip(np.sum(normal * sun, axis=-1), 0.0, 1.0), 0.74)
    daylight = smoothstep(-0.08, 0.12, solar_cos)
    night = 1.0 - daylight
    base_light = 0.034 + daylight * 0.155 + diffuse * 1.24
    if apple_night:
        # Apple's Astronomy treatment keeps the unlit hemisphere legible.
        # This is a restrained ambient fill, not an invented light source: the
        # sun mask, terminator and all observed surface detail remain intact.
        base_light += night * 0.073
    radiance = _linear(albedo) * base_light[..., None]
    radiance += (
        ocean * daylight
    )[..., None] * np.array([0.020, 0.074, 0.124], dtype=np.float32)

    half_vector = sun + view
    half_vector /= np.maximum(np.linalg.norm(half_vector, axis=-1, keepdims=True), 1e-6)
    highlight = np.power(np.clip(np.sum(normal * half_vector, axis=-1), 0.0, 1.0), 92.0)
    grazing = np.power(1.0 - np.clip(np.sum(normal * view, axis=-1), 0.0, 1.0), 4.5)
    specular = ocean * daylight * (highlight * 1.65 + grazing * 0.055)
    radiance += specular[..., None] * np.array([0.40, 0.72, 0.91], dtype=np.float32)
    if apple_night:
        facing = np.clip(np.sum(geometric_normal * view, axis=-1), 0.0, 1.0)
        night_fill = night * (0.70 + facing * 0.30)
        radiance += night_fill[..., None] * np.array(
            [0.0024, 0.0027, 0.0031], dtype=np.float32
        )
        albedo_luminance = np.sum(
            albedo * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
            axis=-1,
        )
        material_midtones = np.power(smoothstep(0.08, 0.58, albedo_luminance), 0.78)
        radiance += (night * material_midtones)[..., None] * np.array(
            [0.0300, 0.0282, 0.0260], dtype=np.float32
        )
        radiance += (ocean * night)[..., None] * np.array(
            [0.0008, 0.0038, 0.0054], dtype=np.float32
        )
    return radiance.astype(np.float32), solar_cos.astype(np.float32)


def cloud_optical_properties(alpha: np.ndarray, day: np.ndarray) -> CloudProperties:
    alpha = np.clip(alpha, 0.0, 1.0)
    optical_depth = -np.log(np.maximum(1.0 - alpha * 0.88, 0.04))
    opacity = np.clip(1.0 - np.exp(-optical_depth * 0.52), 0.0, 0.72)
    daylight = np.power(smoothstep(0.0, 0.66, day), 0.58)
    density_light = 0.42 + 0.58 * np.sqrt(alpha)
    brightness = (0.036 + daylight * 0.93) * density_light
    cool = np.array([0.085, 0.110, 0.145], dtype=np.float32)
    warm = np.array([0.98, 0.955, 0.90], dtype=np.float32)
    tone = cool * (1.0 - daylight[..., None]) + warm * daylight[..., None]
    return CloudProperties(opacity.astype(np.float32), (tone * brightness[..., None]).astype(np.float32))


def cloud_material(
    alpha: np.ndarray,
    observed: np.ndarray,
    solar_cos: np.ndarray,
    *,
    apple_night: bool = False,
) -> CloudProperties:
    """Build a display-calibrated cloud material from real satellite fields.

    Visible imagery supplies daylight structure; the alpha field supplies
    optical depth and night-side occlusion. Colour remains nearly neutral so
    oceans do not tint clouds blue through the composite.
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    observed = np.clip(observed, 0.0, 1.0)
    daylight_curve = smoothstep(-0.08, 0.32, solar_cos)
    daylight = np.power(daylight_curve, 1.05 if apple_night else 0.62)

    medium = _blur_scalar(observed, 1.7)
    broad = _blur_scalar(observed, 10.0)
    volume = np.clip((medium - broad) * 0.24, -0.055, 0.060)
    reflectance = np.clip(np.power(smoothstep(0.035, 0.96, observed), 0.82) + volume, 0.0, 1.0)

    day_srgb = np.clip(0.48 + reflectance * 0.39, 0.46, 0.875)
    day_tone = np.array([1.000, 0.985, 0.955], dtype=np.float32)
    day_radiance = _linear(day_srgb[..., None] * day_tone)
    day_radiance *= (0.80 + daylight[..., None] * 0.24)

    if apple_night:
        # Night clouds need their own dim skylight material.  Keeping them
        # slightly brighter than the ground reveals cloud tops while their
        # alpha still occludes city lights underneath.
        fine_alpha = _blur_scalar(alpha, 1.1)
        broad_alpha = _blur_scalar(alpha, 7.0)
        optical_relief = np.clip(
            (alpha - fine_alpha) * 0.50 + (fine_alpha - broad_alpha) * 0.34,
            -0.11,
            0.11,
        )
        thin = smoothstep(0.045, 0.38, alpha)
        body = smoothstep(0.24, 0.76, alpha)
        top = smoothstep(0.58, 0.92, alpha)
        cloud_detail = np.clip(
            observed + (observed - medium) * 0.18 + (medium - broad) * 0.12,
            0.0,
            1.0,
        )
        cloud_peak = np.power(cloud_detail, 0.88)
        illuminated_relief = np.clip(optical_relief, 0.0, 0.11) / 0.11
        night_srgb = np.clip(
            0.096
            + cloud_peak * 0.176
            + body * cloud_peak * 0.012
            + top * cloud_peak * 0.016
            + illuminated_relief * 0.020,
            0.086,
            0.278,
        )
        night_tone = np.array([0.88, 0.89, 0.91], dtype=np.float32)
    else:
        night_srgb = np.clip(0.028 + reflectance * 0.085, 0.025, 0.115)
        night_tone = np.array([0.82, 0.87, 0.94], dtype=np.float32)
    night_radiance = _linear(night_srgb[..., None] * night_tone)
    radiance = night_radiance * (1.0 - daylight[..., None]) + day_radiance * daylight[..., None]

    if apple_night:
        cloud_detail = np.clip(observed, 0.0, 1.0)
        textured_density = np.power(cloud_detail, 1.42)
        veil_opacity = smoothstep(0.05, 0.82, alpha) * (
            0.012 + textured_density * 0.300 + illuminated_relief * 0.060
        )
        core_opacity = (
            smoothstep(0.54, 0.90, alpha)
            * smoothstep(0.52, 0.92, cloud_detail)
            * 0.125
        )
        night_opacity = veil_opacity + core_opacity
        density = smoothstep(0.055, 0.88, alpha)
        day_opacity = density * (0.28 + reflectance * 0.56)
        opacity = night_opacity * (1.0 - daylight) + day_opacity * daylight
    else:
        density = smoothstep(0.055, 0.88, alpha)
        texture_weight = 0.28 + reflectance * 0.56
        opacity = density * texture_weight
        opacity *= 0.88 + daylight * 0.12
    opacity = np.clip(
        _blur_scalar(opacity, 2.0 if apple_night else 0.48),
        0.0,
        0.82,
    )
    return CloudProperties(opacity.astype(np.float32), radiance.astype(np.float32))


def _projected_sun(preset: V2Preset, sun: np.ndarray) -> tuple[float, float]:
    _forward, east, north = _basis(preset.target_lat, preset.target_lon)
    x = float(np.dot(sun, east))
    y = float(-np.dot(sun, north))
    length = max(np.hypot(x, y), 1e-6)
    return x / length, y / length


def _cloud_layer(
    surface: np.ndarray,
    cloud_alpha: np.ndarray,
    cloud_texture: np.ndarray,
    solar_cos: np.ndarray,
    view_cos: np.ndarray,
    sun: np.ndarray,
    preset: V2Preset,
    *,
    apple_night: bool = False,
) -> np.ndarray:
    sun_x, sun_y = _projected_sun(preset, sun)
    observed = np.clip(cloud_texture, 0.0, 1.0)
    soft = _blur_scalar(cloud_alpha * (0.55 + observed * 0.45), 1.8)
    # The stylised night profile keeps cloud shadows directly beneath the
    # observed cloud footprint. A screen-space offset creates a visible double
    # edge in the full-globe Lock view and reads as projection misalignment.
    offset = 0 if apple_night else (5 if preset.name == "lock" else 8)
    cast = _shift(soft, int(round(-sun_x * offset)), int(round(-sun_y * offset)))
    exposed = np.clip(cast - cloud_alpha * 0.46, 0.0, 1.0)
    daylight = smoothstep(-0.04, 0.40, solar_cos)
    shadow = smoothstep(0.04, 0.70, exposed) * daylight * 0.145
    surface = surface * (1.0 - shadow[..., None])

    material = cloud_material(cloud_alpha, observed, solar_cos, apple_night=apple_night)
    cloud_radiance = material.radiance
    cloud_opacity = material.opacity
    if apple_night:
        # Night clouds receive weak skylight and should sit just above the
        # local ground value. This removes dark cut-out shapes while keeping
        # their real IR-derived structure and optical depth.
        local_surface = _blur_rgb(surface, 2.4)
        cloud_volume = np.power(smoothstep(0.08, 0.88, observed), 0.86)
        minimum_night_cloud = local_surface * (
            1.045 + cloud_volume[..., None] * 0.39
        ) + (
            0.22 + cloud_volume[..., None] * 0.78
        ) * np.array([0.0038, 0.0044, 0.0053], dtype=np.float32)
        cloud_night = 1.0 - daylight
        lifted_night = np.maximum(material.radiance, minimum_night_cloud)
        limb = np.power(np.clip(1.0 - view_cos, 0.0, 1.0), 1.65)
        edge_fade = smoothstep(0.020, 0.145, view_cos)
        density = smoothstep(0.07, 0.82, cloud_alpha)
        limb_opacity = _blur_scalar(material.opacity, 3.8)
        spherical_opacity = (
            material.opacity * (1.0 - limb * 0.62)
            + limb_opacity * limb * 0.62
        )
        # Longer optical paths near the limb make the layer read as a shell,
        # while the final few pixels fade into the atmospheric rim instead of
        # forming a flat strip across the top of the globe.
        cloud_opacity = np.clip(
            spherical_opacity * (1.0 + limb * 0.52) * edge_fade,
            0.0,
            0.68,
        )
        limb_skylight = (
            limb * density * cloud_night
        )[..., None] * np.array([0.0028, 0.0040, 0.0056], dtype=np.float32)
        cloud_radiance = (
            material.radiance * daylight[..., None]
            + lifted_night * cloud_night[..., None]
        ) + limb_skylight
    composite = (
        surface * (1.0 - cloud_opacity[..., None])
        + cloud_radiance * cloud_opacity[..., None]
    )
    if apple_night:
        # A very low-energy silver lining restores the observed fine structure
        # without turning the full IR footprint into a flat grey veil.
        local = _blur_scalar(observed, 2.2)
        fine_relief = np.clip(observed - local * 0.72, 0.0, 1.0)
        cloud_night = 1.0 - daylight
        detail = (
            fine_relief
            * smoothstep(0.05, 0.78, cloud_alpha)
            * cloud_night
        )
        composite += detail[..., None] * np.array(
            [0.0038, 0.0042, 0.0048], dtype=np.float32
        )
    return composite


def atmosphere_scattering(
    visible: np.ndarray,
    view_cos: np.ndarray,
    vectors: np.ndarray,
    sun: np.ndarray,
    size: tuple[int, int],
    *,
    apple_night: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    solar = np.sum(vectors * sun, axis=-1)
    if apple_night:
        sunward = 0.055 + 0.945 * smoothstep(-0.48, 0.42, solar)
    else:
        sunward = 0.40 + 0.60 * smoothstep(-0.45, 0.65, solar)
    limb = np.power(np.clip(1.0 - view_cos, 0.0, 1.0), 2.65)
    rim = limb * visible * sunward

    mask_image = Image.fromarray(np.uint8(visible * 255), "L")
    near_radius = size[0] * (0.0115 if apple_night else 0.0062)
    far_radius = size[0] * (0.032 if apple_night else 0.017)
    near = np.asarray(
        mask_image.filter(ImageFilter.GaussianBlur(max(5, near_radius))), dtype=np.float32
    ) / 255.0
    far = np.asarray(
        mask_image.filter(ImageFilter.GaussianBlur(max(12, far_radius))), dtype=np.float32
    ) / 255.0
    near_weight = 0.66 if apple_night else 0.78
    halo = np.clip(
        (near - visible) * near_weight + (far - visible) * (1.0 - near_weight),
        0.0,
        1.0,
    )
    halo *= sunward
    return rim.astype(np.float32), halo.astype(np.float32)


def _edge_alpha(preset: V2Preset) -> np.ndarray:
    width, height = preset.size
    yy, xx = np.mgrid[0:height, 0:width]
    rho = np.sqrt(
        ((xx.astype(np.float32) - preset.center_px[0]) / preset.globe_radius_px) ** 2
        + ((yy.astype(np.float32) - preset.center_px[1]) / preset.globe_radius_px) ** 2
    )
    feather_px = 8.0 if preset.name == "lock" else 11.0
    feather = feather_px / preset.globe_radius_px
    return (1.0 - smoothstep(1.0 - feather, 1.0 + feather, rho)).astype(np.float32)


def _soft_bloom(linear_rgb: np.ndarray) -> np.ndarray:
    luminance = np.sum(
        linear_rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1
    )
    highlights = np.clip((luminance - 0.82) * 0.24, 0.0, 0.16)
    image = Image.fromarray(np.uint8(highlights * 255 / 0.16), "L")
    bloom = np.asarray(image.filter(ImageFilter.GaussianBlur(8.0)), dtype=np.float32) / 255.0 * 0.16
    return linear_rgb + bloom[..., None] * np.array([0.92, 0.96, 1.0], dtype=np.float32) * 0.11


def _apple_night_midtone_grade(earth: np.ndarray, night: np.ndarray) -> np.ndarray:
    """Open night midtones while retaining black oceans and light highlights."""
    luminance = np.sum(
        earth * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1
    )
    toe = smoothstep(0.0018, 0.011, luminance)
    highlight_rolloff = 1.0 - smoothstep(0.060, 0.27, luminance)
    lift = night * toe * highlight_rolloff
    return earth + lift[..., None] * np.array(
        [0.0040, 0.0045, 0.0048], dtype=np.float32
    )


def _soften_surface_limb(surface: np.ndarray, view_cos: np.ndarray) -> np.ndarray:
    """Suppress high-frequency terrain leakage at the grazing-angle rim."""
    limb = 1.0 - smoothstep(0.025, 0.18, view_cos)
    luminance = np.sum(
        surface * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1
    )
    broad_luminance = _blur_scalar(luminance, 8.0)
    highlight_excess = np.clip(luminance - broad_luminance * 1.04, 0.0, None)
    target_luminance = np.maximum(luminance - highlight_excess * limb * 0.82, 0.0)
    scale = target_luminance / np.maximum(luminance, 1e-5)
    compressed = surface * scale[..., None]
    smoothed = _blur_rgb(compressed, 8.0)
    blend = limb * 0.82
    return compressed * (1.0 - blend[..., None]) + smoothed * blend[..., None]


def render_preview_one(
    observation: Observation,
    preset: V2Preset,
    destination: Path,
    *,
    lighting_time: datetime,
    background_asset: Path | None = Path("assets/space-background.jpg"),
    output_size: tuple[int, int] | None = None,
    apple_night: bool = False,
) -> None:
    lat, lon, visible, view_cos, vectors, view = perspective_camera_grid(preset)
    # Clouds and terrain must share the exact same geographic projection.
    # Apparent altitude belongs in shading/edge softness, not a larger sphere:
    # a larger sphere shifts coastlines and becomes obvious in the full-globe
    # Lock composition.
    cloud_preset = preset
    cloud_lat, cloud_lon, cloud_visible, cloud_vectors = lat, lon, visible, vectors
    base_map = _load(observation.base)
    base = sample_equirectangular(base_map, lat, lon)
    cloud_base = (
        sample_equirectangular(base_map, cloud_lat, cloud_lon)
        if apple_night
        else base
    )
    del base_map
    lights_map = _load(observation.lights)
    lights = sample_equirectangular(lights_map, lat, lon)
    del lights_map
    if observation.terrain:
        relief_map = _load(observation.terrain)
        relief = sample_equirectangular(relief_map, lat, lon)
        del relief_map
    else:
        relief = None

    albedo, ocean, land = _material_albedo(base)
    normal = _relief_normals(vectors, relief, land, preset)
    sun = sun_vector(lighting_time).astype(np.float32)
    cloud_solar_cos = np.sum(cloud_vectors * sun, axis=-1).astype(np.float32)
    earth, solar_cos = _surface_radiance(
        albedo,
        ocean,
        normal,
        vectors,
        view,
        sun,
        apple_night=apple_night,
    )
    if apple_night:
        earth = _soften_surface_limb(earth, view_cos)

    if observation.geocolor is None:
        visible_map = _load(observation.visible)
        cloud_visible_image = sample_equirectangular(visible_map, cloud_lat, cloud_lon)
        del visible_map
        infrared_map = _load(observation.infrared)
        cloud_infrared = sample_equirectangular(infrared_map, cloud_lat, cloud_lon)
        del infrared_map
        cloud_alpha = _cloud_alpha(
            cloud_visible_image,
            cloud_infrared,
            cloud_base,
            np.clip(cloud_solar_cos, 0.0, 1.0),
        )
        apple_thermal_texture = None
        if apple_night:
            night_alpha, apple_thermal_texture = _apple_night_ir_cloud(
                cloud_infrared,
                cloud_base,
            )
            night_weight = 1.0 - smoothstep(
                0.02,
                0.30,
                np.clip(cloud_solar_cos, 0.0, 1.0),
            )
            cloud_alpha = (
                cloud_alpha * (1.0 - night_weight)
                + night_alpha * night_weight
            )
        cloud_alpha *= cloud_visible
        visible_luminance = np.sum(
            cloud_visible_image[..., :3]
            * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
            axis=-1,
        )
        daytime_texture = np.power(smoothstep(0.025, 0.96, visible_luminance), 0.90)
        daytime_fine = _blur_scalar(daytime_texture, 1.4)
        daytime_broad = _blur_scalar(daytime_texture, 10.0)
        daytime_texture = np.clip(
            daytime_texture
            + (daytime_texture - daytime_fine) * 0.10
            + (daytime_fine - daytime_broad) * 0.16,
            0.0,
            1.0,
        )
        thermal_texture = (
            apple_thermal_texture
            if apple_thermal_texture is not None
            else _thermal_cloud_texture(cloud_infrared)
        )
        day_texture_mix = smoothstep(0.02, 0.30, np.clip(cloud_solar_cos, 0.0, 1.0))
        cloud_texture = np.clip(
            daytime_texture * day_texture_mix
            + thermal_texture * (1.0 - day_texture_mix),
            0.0,
            1.0,
        )
    else:
        geocolor = _load(observation.geocolor)
        satellite, valid = sample_geostationary_focus_plate(
            geocolor, cloud_preset, observation.satellite_longitude
        )
        day = np.clip(cloud_solar_cos, 0.0, 1.0)
        cloud_alpha = np.maximum(_day_cloud_alpha(satellite, day), _night_cloud_alpha(satellite, day))
        cloud_alpha *= valid * cloud_visible
        cloud_texture = np.clip(satellite[..., :3].mean(axis=-1), 0.0, 1.0)

    night = 1.0 - smoothstep(-0.08, 0.14, solar_cos)
    raw_light_signal = np.clip(_city_light_signal(lights), 0.0, 1.0)
    if apple_night:
        crisp_lights = np.power(raw_light_signal, 1.38 if preset.name == "lock" else 1.48)
        local_lights = _blur_scalar(crisp_lights, 1.7 if preset.name == "lock" else 1.3)
        broad_lights = _blur_scalar(crisp_lights, 4.5 if preset.name == "lock" else 3.7)
        light_signal = crisp_lights * 0.67 + local_lights * 0.25 + broad_lights * 0.08
        light_signal *= 1.0 - np.clip(light_signal, 0.0, 1.0) * 0.25
        light_signal *= np.clip(1.0 - cloud_alpha * 0.70, 0.24, 1.0)
        light_strength = 0.40 if preset.name == "lock" else 0.36
        light_tone = np.array([1.36, 0.56, 0.14], dtype=np.float32)
    else:
        light_exponent = 1.32
        light_strength = 1.72
        light_tone = np.array([2.35, 0.88, 0.20], dtype=np.float32)
        light_signal = np.power(raw_light_signal, light_exponent)
    earth += (
        light_signal * night * light_strength
    )[..., None] * light_tone
    earth = _cloud_layer(
        earth,
        cloud_alpha,
        cloud_texture,
        cloud_solar_cos,
        view_cos,
        sun,
        preset,
        apple_night=apple_night,
    )
    if apple_night:
        earth = _apple_night_midtone_grade(earth, night)

    daylight = smoothstep(-0.05, 0.32, solar_cos)
    aerial = daylight * (
        0.018 + np.power(np.clip(1.0 - view_cos, 0.0, 1.0), 1.7) * 0.15
    )
    aerial_tone = np.array([0.075, 0.245, 0.390], dtype=np.float32)
    earth = earth * (1.0 - aerial[..., None]) + aerial_tone * aerial[..., None]

    rim, halo = atmosphere_scattering(
        visible,
        view_cos,
        vectors,
        sun,
        preset.size,
        apple_night=apple_night,
    )
    atmosphere_color = np.array([0.20, 0.67, 1.16], dtype=np.float32)
    if apple_night:
        daylight_atmosphere = smoothstep(-0.08, 0.32, solar_cos)
        night_atmosphere_color = np.array([0.43, 0.80, 1.07], dtype=np.float32)
        atmosphere_tone = (
            night_atmosphere_color
            + daylight_atmosphere[..., None]
            * (atmosphere_color - night_atmosphere_color)
        )
        rim_strength = 0.225 + daylight_atmosphere * 0.035
        night_fraction = float(night[visible].mean())
        halo_strength = 0.215 - night_fraction * 0.015
        halo_color = (
            night_atmosphere_color * night_fraction
            + atmosphere_color * (1.0 - night_fraction)
        )
    else:
        atmosphere_tone = atmosphere_color
        rim_strength = 0.20
        halo_strength = 0.17
        halo_color = atmosphere_color
    earth += rim[..., None] * atmosphere_tone * np.asarray(rim_strength)[..., None]
    edge_alpha = _edge_alpha(preset)

    background_srgb = space_background(preset.size, asset=background_asset)
    output = _linear(background_srgb)
    output += halo[..., None] * halo_color * halo_strength
    # Perspective rays outside ``visible`` do not intersect the globe.  Never
    # composite their sampled texture: around the Arctic it previously leaked
    # bright ice into the exterior feather and looked like a detached cloud
    # strip.  Preserve the already-rendered atmosphere there instead.
    composite_earth = np.where(visible[..., None], earth, output)
    output = (
        output * (1.0 - edge_alpha[..., None])
        + composite_earth * edge_alpha[..., None]
    )
    output = _soft_bloom(output)
    output = aces_tonemap(output * 1.16)

    luminance = np.sum(
        output * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1, keepdims=True
    )
    shadows = 1.0 - smoothstep(0.10, 0.52, luminance)
    highlights = smoothstep(0.42, 0.90, luminance)
    output *= (
        1.0
        + shadows * np.array([-0.010, 0.002, 0.018], dtype=np.float32)
        + highlights * np.array([0.018, 0.006, -0.016], dtype=np.float32)
    )
    output = _display(np.clip(output, 0.0, 1.0))

    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.uint8(output * 255), "RGB")
    if output_size is not None and image.size != output_size:
        image = image.resize(output_size, Image.Resampling.LANCZOS)
    image.save(destination, quality=97, subsampling=0)


def render_preview_pair(
    observation: Observation,
    output: Path,
    *,
    latitude: float = SHANGHAI[0],
    longitude: float = SHANGHAI[1],
    lighting_time: datetime | None = None,
    apple_night: bool = False,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    lighting_time = lighting_time or datetime.now(UTC)
    artifacts: dict[str, dict] = {}
    for preset in presets_for_location(latitude, longitude):
        destination = output / f"{preset.name}.jpg"
        render_preview_one(
            observation,
            preset,
            destination,
            lighting_time=lighting_time,
            apple_night=apple_night,
        )
        artifacts[preset.name] = {
            "file": destination.name,
            "sha256": sha256(destination),
            "size": preset.size,
            "camera_distance": preset.camera_distance,
            "view_center": {
                "latitude": float(preset.target_lat),
                "longitude": float(preset.target_lon),
            },
        }
    manifest = {
        "preview_only": True,
        "renderer": "cinematic-earth-v2-apple-night-study" if apple_night else "cinematic-earth-v2",
        "observation_utc": observation.timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "lighting_utc": lighting_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "rendered_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": observation.source,
        "source_status": observation.status,
        "target": {"latitude": latitude, "longitude": longitude},
        "style": "apple-night-study" if apple_night else "production-v2",
        "artifacts": artifacts,
        "production_paths_untouched": ["output/current/lock.jpg", "output/current/home.jpg"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _production_manifest(
    observation: Observation,
    artifacts: dict,
    *,
    lighting_time: datetime,
    latitude: float,
    longitude: float,
    target_name: str,
    profile: str,
) -> dict:
    sun = sun_vector(lighting_time)
    manifest = {
        "profile": profile,
        "renderer": "cinematic-earth-v2",
        "style": "apple-night-v24",
        "preview_only": False,
        "observation_utc": observation.timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "lighting_utc": lighting_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "rendered_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": observation.source,
        "source_status": observation.status,
        "render_mode": "cinematic_v2_real_observation_layers",
        "observation_asset_sha256": sha256(observation.geocolor or observation.visible),
        "target": {
            "name": target_name,
            "latitude": latitude,
            "longitude": longitude,
        },
        "view_center": {
            name: artifact["view_center"] for name, artifact in artifacts.items()
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
    return manifest


def render_production_pair(
    observation: Observation,
    output: Path,
    *,
    target_latitude: float = SHANGHAI[0],
    target_longitude: float = SHANGHAI[1],
    target_name: str = "Shanghai",
    lighting_time: datetime | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    lighting_time = lighting_time or datetime.now(UTC)
    artifacts: dict[str, dict] = {}
    for preset in presets_for_location(target_latitude, target_longitude):
        path = output / f"{preset.name}.jpg"
        working_scale = 0.68
        working = replace(
            preset,
            size=(round(preset.size[0] * working_scale), round(preset.size[1] * working_scale)),
            center_px=(
                preset.center_px[0] * working_scale,
                preset.center_px[1] * working_scale,
            ),
            globe_radius_px=preset.globe_radius_px * working_scale,
        )
        render_preview_one(
            observation,
            working,
            path,
            lighting_time=lighting_time,
            output_size=preset.size,
            apple_night=True,
        )
        artifacts[preset.name] = {
            "file": path.name,
            "sha256": sha256(path),
            "size": preset.size,
            "camera_distance": preset.camera_distance,
            "working_scale": working_scale,
            "view_center": {
                "latitude": float(preset.target_lat),
                "longitude": float(preset.target_lon),
            },
        }
    manifest = _production_manifest(
        observation,
        artifacts,
        lighting_time=lighting_time,
        latitude=target_latitude,
        longitude=target_longitude,
        target_name=target_name,
        profile="phone",
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def render_production_mac_pair(
    observation: Observation,
    output: Path,
    *,
    target_latitude: float = SHANGHAI[0],
    target_longitude: float = SHANGHAI[1],
    target_name: str = "Shanghai",
    lighting_time: datetime | None = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    lighting_time = lighting_time or datetime.now(UTC)
    artifacts: dict[str, dict] = {}
    for preset in presets_for_mac_location(target_latitude, target_longitude):
        path = output / f"mac-{preset.name}.jpg"
        render_preview_one(
            observation,
            preset,
            path,
            lighting_time=lighting_time,
            apple_night=True,
        )
        artifacts[preset.name] = {
            "file": path.name,
            "sha256": sha256(path),
            "size": preset.size,
            "camera_distance": preset.camera_distance,
            "view_center": {
                "latitude": float(preset.target_lat),
                "longitude": float(preset.target_lon),
            },
        }
    manifest = _production_manifest(
        observation,
        artifacts,
        lighting_time=lighting_time,
        latitude=target_latitude,
        longitude=target_longitude,
        target_name=target_name,
        profile="mac",
    )
    (output / "mac-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
