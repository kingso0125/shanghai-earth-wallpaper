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
) -> tuple[np.ndarray, np.ndarray]:
    solar_cos = np.sum(geometric_normal * sun, axis=-1)
    diffuse = np.power(np.clip(np.sum(normal * sun, axis=-1), 0.0, 1.0), 0.74)
    daylight = smoothstep(-0.08, 0.12, solar_cos)
    base_light = 0.034 + daylight * 0.155 + diffuse * 1.24
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
) -> CloudProperties:
    """Build a display-calibrated cloud material from real satellite fields.

    Visible imagery supplies daylight structure; the alpha field supplies
    optical depth and night-side occlusion. Colour remains nearly neutral so
    oceans do not tint clouds blue through the composite.
    """
    alpha = np.clip(alpha, 0.0, 1.0)
    observed = np.clip(observed, 0.0, 1.0)
    daylight = np.power(smoothstep(-0.08, 0.32, solar_cos), 0.62)

    medium = _blur_scalar(observed, 1.7)
    broad = _blur_scalar(observed, 10.0)
    volume = np.clip((medium - broad) * 0.24, -0.055, 0.060)
    reflectance = np.clip(np.power(smoothstep(0.035, 0.96, observed), 0.82) + volume, 0.0, 1.0)

    day_srgb = np.clip(0.48 + reflectance * 0.39, 0.46, 0.875)
    day_tone = np.array([1.000, 0.985, 0.955], dtype=np.float32)
    day_radiance = _linear(day_srgb[..., None] * day_tone)
    day_radiance *= (0.80 + daylight[..., None] * 0.24)

    night_srgb = np.clip(0.028 + reflectance * 0.085, 0.025, 0.115)
    night_tone = np.array([0.82, 0.87, 0.94], dtype=np.float32)
    night_radiance = _linear(night_srgb[..., None] * night_tone)
    radiance = night_radiance * (1.0 - daylight[..., None]) + day_radiance * daylight[..., None]

    density = smoothstep(0.055, 0.88, alpha)
    texture_weight = 0.28 + reflectance * 0.56
    opacity = density * texture_weight
    opacity *= 0.88 + daylight * 0.12
    opacity = np.clip(_blur_scalar(opacity, 0.48), 0.0, 0.82)
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
    sun: np.ndarray,
    preset: V2Preset,
) -> np.ndarray:
    sun_x, sun_y = _projected_sun(preset, sun)
    observed = np.clip(cloud_texture, 0.0, 1.0)
    soft = _blur_scalar(cloud_alpha * (0.55 + observed * 0.45), 1.8)
    offset = 5 if preset.name == "lock" else 8
    cast = _shift(soft, int(round(-sun_x * offset)), int(round(-sun_y * offset)))
    exposed = np.clip(cast - cloud_alpha * 0.46, 0.0, 1.0)
    daylight = smoothstep(-0.04, 0.40, solar_cos)
    shadow = smoothstep(0.04, 0.70, exposed) * daylight * 0.145
    surface = surface * (1.0 - shadow[..., None])

    material = cloud_material(cloud_alpha, observed, solar_cos)
    return surface * (1.0 - material.opacity[..., None]) + material.radiance * material.opacity[..., None]


def atmosphere_scattering(
    visible: np.ndarray,
    view_cos: np.ndarray,
    vectors: np.ndarray,
    sun: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    solar = np.sum(vectors * sun, axis=-1)
    sunward = 0.40 + 0.60 * smoothstep(-0.45, 0.65, solar)
    limb = np.power(np.clip(1.0 - view_cos, 0.0, 1.0), 2.65)
    rim = limb * visible * sunward

    mask_image = Image.fromarray(np.uint8(visible * 255), "L")
    near = np.asarray(
        mask_image.filter(ImageFilter.GaussianBlur(max(5, size[0] * 0.0062))), dtype=np.float32
    ) / 255.0
    far = np.asarray(
        mask_image.filter(ImageFilter.GaussianBlur(max(12, size[0] * 0.017))), dtype=np.float32
    ) / 255.0
    halo = np.clip((near - visible) * 0.78 + (far - visible) * 0.22, 0.0, 1.0)
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


def render_preview_one(
    observation: Observation,
    preset: V2Preset,
    destination: Path,
    *,
    lighting_time: datetime,
    background_asset: Path | None = Path("assets/space-background.jpg"),
    output_size: tuple[int, int] | None = None,
) -> None:
    lat, lon, visible, view_cos, vectors, view = perspective_camera_grid(preset)
    base_map = _load(observation.base)
    base = sample_equirectangular(base_map, lat, lon)
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
    earth, solar_cos = _surface_radiance(albedo, ocean, normal, vectors, view, sun)

    if observation.geocolor is None:
        visible_map = _load(observation.visible)
        cloud_visible = sample_equirectangular(visible_map, lat, lon)
        del visible_map
        infrared_map = _load(observation.infrared)
        cloud_infrared = sample_equirectangular(infrared_map, lat, lon)
        del infrared_map
        cloud_alpha = _cloud_alpha(
            cloud_visible,
            cloud_infrared,
            base,
            np.clip(solar_cos, 0.0, 1.0),
        )
        visible_luminance = np.sum(
            cloud_visible[..., :3]
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
        thermal_texture = _thermal_cloud_texture(cloud_infrared)
        day_texture_mix = smoothstep(0.02, 0.30, np.clip(solar_cos, 0.0, 1.0))
        cloud_texture = np.clip(
            daytime_texture * day_texture_mix
            + thermal_texture * (1.0 - day_texture_mix),
            0.0,
            1.0,
        )
    else:
        geocolor = _load(observation.geocolor)
        satellite, valid = sample_geostationary_focus_plate(
            geocolor, preset, observation.satellite_longitude
        )
        day = np.clip(solar_cos, 0.0, 1.0)
        cloud_alpha = np.maximum(_day_cloud_alpha(satellite, day), _night_cloud_alpha(satellite, day))
        cloud_alpha *= valid
        cloud_texture = np.clip(satellite[..., :3].mean(axis=-1), 0.0, 1.0)

    night = 1.0 - smoothstep(-0.08, 0.14, solar_cos)
    light_signal = np.power(np.clip(_city_light_signal(lights), 0.0, 1.0), 1.32)
    earth += (
        light_signal * night * 1.72
    )[..., None] * np.array([2.35, 0.88, 0.20], dtype=np.float32)
    earth = _cloud_layer(earth, cloud_alpha, cloud_texture, solar_cos, sun, preset)

    daylight = smoothstep(-0.05, 0.32, solar_cos)
    aerial = daylight * (
        0.018 + np.power(np.clip(1.0 - view_cos, 0.0, 1.0), 1.7) * 0.15
    )
    aerial_tone = np.array([0.075, 0.245, 0.390], dtype=np.float32)
    earth = earth * (1.0 - aerial[..., None]) + aerial_tone * aerial[..., None]

    rim, halo = atmosphere_scattering(visible, view_cos, vectors, sun, preset.size)
    atmosphere_color = np.array([0.20, 0.67, 1.16], dtype=np.float32)
    earth += rim[..., None] * atmosphere_color * 0.20
    edge_alpha = _edge_alpha(preset)

    background_srgb = space_background(preset.size, asset=background_asset)
    output = _linear(background_srgb)
    output += halo[..., None] * atmosphere_color * 0.17
    output = output * (1.0 - edge_alpha[..., None]) + earth * edge_alpha[..., None]
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
        "renderer": "cinematic-earth-v2",
        "observation_utc": observation.timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "lighting_utc": lighting_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "rendered_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": observation.source,
        "source_status": observation.status,
        "target": {"latitude": latitude, "longitude": longitude},
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
        render_preview_one(observation, preset, path, lighting_time=lighting_time)
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
