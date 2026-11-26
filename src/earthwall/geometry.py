from __future__ import annotations

import numpy as np

from .config import RenderPreset


def camera_grid(preset: RenderPreset):
    width, height = preset.size
    cx, cy = preset.center_px
    radius = preset.globe_radius_px
    yy, xx = np.mgrid[0:height, 0:width]
    sx = (xx.astype(np.float32) - cx) / radius
    sy = (yy.astype(np.float32) - cy) / radius
    rho2 = sx * sx + sy * sy
    visible = rho2 <= 1.0
    sz = np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0)).astype(np.float32)

    lat0 = np.deg2rad(preset.target_lat)
    lon0 = np.deg2rad(preset.target_lon)
    forward = np.array(
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        dtype=np.float32,
    )
    east = np.array([-np.sin(lon0), np.cos(lon0), 0.0], dtype=np.float32)
    north = np.array(
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        dtype=np.float32,
    )
    vectors = (
        sx[..., None] * east
        - sy[..., None] * north
        + sz[..., None] * forward
    )
    lat = np.arcsin(np.clip(vectors[..., 2], -1.0, 1.0))
    lon = np.arctan2(vectors[..., 1], vectors[..., 0])
    return lat, lon, visible, sz, vectors


def sample_equirectangular(image: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    height, width = image.shape[:2]
    xf = ((lon + np.pi) / (2.0 * np.pi) * width) % width
    yf = np.clip((np.pi / 2.0 - lat) / np.pi * (height - 1), 0, height - 1)
    # Float32 modulo can round a tiny negative longitude to exactly ``width``.
    # Wrap the integer index too, keeping interpolation weights unwrapped.
    floor_x = np.floor(xf)
    x0 = floor_x.astype(np.int32) % width
    y0 = np.floor(yf).astype(np.int32)
    x1 = (x0 + 1) % width
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (xf - floor_x)[..., None].astype(np.float32)
    wy = (yf - y0)[..., None].astype(np.float32)
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def sample_himawari_plate(image: np.ndarray, preset: RenderPreset):
    """Scale the native full-disk observation without separating clouds from Earth."""
    target_width, target_height = preset.size
    yy, xx = np.mgrid[0:target_height, 0:target_width]
    sx = (xx.astype(np.float32) - preset.center_px[0]) / preset.globe_radius_px
    sy = (yy.astype(np.float32) - preset.center_px[1]) / preset.globe_radius_px

    height, width = image.shape[:2]
    source_radius = min(width, height) * 0.493
    xf = (width - 1) / 2.0 + sx * source_radius
    yf = (height - 1) / 2.0 + sy * source_radius
    valid = (sx * sx + sy * sy <= 1.0) & (
        (xf >= 0) & (xf < width - 1) & (yf >= 0) & (yf < height - 1)
    )
    xf = np.clip(xf, 0, width - 1.001)
    yf = np.clip(yf, 0, height - 1.001)
    x0 = np.floor(xf).astype(np.int32)
    y0 = np.floor(yf).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (xf - x0)[..., None].astype(np.float32)
    wy = (yf - y0)[..., None].astype(np.float32)
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy, valid


def sample_geostationary_focus_plate(
    image: np.ndarray, preset: RenderPreset, satellite_longitude: float
):
    """Recenter a fused full-disk plate while preserving its circular boundary."""
    target_width, target_height = preset.size
    yy, xx = np.mgrid[0:target_height, 0:target_width]
    wx = (xx.astype(np.float32) - preset.center_px[0]) / preset.globe_radius_px
    wy = (yy.astype(np.float32) - preset.center_px[1]) / preset.globe_radius_px
    valid = wx * wx + wy * wy <= 1.0

    satellite_longitude = np.deg2rad(satellite_longitude)
    target_latitude = np.deg2rad(preset.target_lat)
    target_longitude = np.deg2rad(preset.target_lon)
    vector = np.array(
        [
            np.cos(target_latitude) * np.cos(target_longitude),
            np.cos(target_latitude) * np.sin(target_longitude),
            np.sin(target_latitude),
        ],
        dtype=np.float64,
    )
    forward = np.array(
        [np.cos(satellite_longitude), np.sin(satellite_longitude), 0.0], dtype=np.float64
    )
    east = np.array(
        [-np.sin(satellite_longitude), np.cos(satellite_longitude), 0.0], dtype=np.float64
    )
    front = float(vector @ forward)
    east_component = float(vector @ east)
    north_component = float(vector[2])
    satellite_radius = 42164.0 / 6378.137
    max_scan = np.arcsin(1.0 / satellite_radius)
    focus_x = np.arctan2(east_component, satellite_radius - front) / max_scan
    focus_y = -np.arctan2(
        north_component,
        np.sqrt((satellite_radius - front) ** 2 + east_component**2),
    ) / max_scan

    # A disk automorphism moves Shanghai to (0, 0) without exposing satellite
    # blind zones. It transforms the already fused cloud/surface pixels together.
    denominator_real = 1.0 + focus_x * wx + focus_y * wy
    denominator_imag = focus_x * wy - focus_y * wx
    numerator_real = wx + focus_x
    numerator_imag = wy + focus_y
    denominator_squared = denominator_real**2 + denominator_imag**2
    zx = (numerator_real * denominator_real + numerator_imag * denominator_imag) / denominator_squared
    zy = (numerator_imag * denominator_real - numerator_real * denominator_imag) / denominator_squared

    height, width = image.shape[:2]
    source_radius = min(width, height) * 0.493
    xf = (width - 1) / 2.0 + zx * source_radius
    yf = (height - 1) / 2.0 + zy * source_radius
    xf = np.clip(xf, 0, width - 1.001)
    yf = np.clip(yf, 0, height - 1.001)
    x0 = np.floor(xf).astype(np.int32)
    y0 = np.floor(yf).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    blend_x = (xf - x0)[..., None].astype(np.float32)
    blend_y = (yf - y0)[..., None].astype(np.float32)
    top = image[y0, x0] * (1.0 - blend_x) + image[y0, x1] * blend_x
    bottom = image[y1, x0] * (1.0 - blend_x) + image[y1, x1] * blend_x
    return top * (1.0 - blend_y) + bottom * blend_y, valid


def sample_cira_geocolor(image, latitude, longitude, satellite_longitude):
    """Sample the CIRA Himawari/GK2A full disk at the terrain's geodetic coordinates.

    SLIDER full_disk lat_lon_query specifies the zoom-0 radii on a 688 px tile.
    Invert its scan-to-geodetic equations; both instruments use a y-sweep grid.
    """
    equator, polar, distance = 6378.1, 6356.8, 42171.7
    axis_ratio = (polar / equator) ** 2
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    delta = longitude - np.float32(np.deg2rad(satellite_longitude))
    radius = equator / np.sqrt(1 - (1 - axis_ratio) * sin_lat**2)
    toward = distance - radius * cos_lat * np.cos(delta)
    east = radius * cos_lat * np.sin(delta)
    north = radius * axis_ratio * sin_lat
    scan_x = np.arctan2(east, toward)
    scan_y = np.arctan2(north, np.hypot(toward, east))
    height, width = image.shape[:2]
    xf = (width - 1) / 2 + scan_x * (width / 688 * 337 / .150618)
    yf = (height - 1) / 2 - scan_y * (height / 688 * 336 / .150485)
    # The satellite must be above the local ellipsoid horizon. The far side
    # also maps to in-bounds pixels, so rectangular bounds alone are insufficient.
    horizon = cos_lat * np.cos(delta) - equator**2 / (radius * distance)
    valid = (horizon > 0) & (xf >= 0) & (xf < width - 1) & (yf >= 0) & (yf < height - 1)
    # GeoColor is strongly limb-brightened. Fade the last few scan rows
    # continuously instead of cutting a bright crescent into the basemap.
    zenith_cos = horizon * distance / np.sqrt(toward**2 + east**2 + north**2)
    t = np.clip((zenith_cos - .035) / .24, 0, 1)
    return _sample_disk_pixels(image, xf, yf), valid * t**2 * (3 - 2 * t)


def sample_geostationary_disk(
    image: np.ndarray, surface_vectors: np.ndarray, satellite_longitude: float
):
    longitude = np.deg2rad(satellite_longitude)
    forward = np.array([np.cos(longitude), np.sin(longitude), 0.0], dtype=np.float32)
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.0], dtype=np.float32)
    north = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    front = np.sum(surface_vectors * forward, axis=-1)
    sx = np.sum(surface_vectors * east, axis=-1)
    north_component = np.sum(surface_vectors * north, axis=-1)
    height, width = image.shape[:2]
    radius = min(width, height) * 0.493
    satellite_radius = 42164.0 / 6378.137
    horizon = 1.0 / satellite_radius
    max_scan = np.arcsin(horizon)
    toward_satellite = satellite_radius - front
    scan_x = np.arctan2(sx, toward_satellite)
    scan_y = np.arctan2(
        north_component,
        np.sqrt(toward_satellite * toward_satellite + sx * sx),
    )
    xf = width / 2.0 + scan_x / max_scan * radius
    yf = height / 2.0 - scan_y / max_scan * radius
    valid = (front > horizon) & (xf >= 0) & (xf < width - 1) & (yf >= 0) & (yf < height - 1)
    return _sample_disk_pixels(image, xf, yf), valid, front


def _sample_disk_pixels(image, xf, yf):
    height, width = image.shape[:2]
    xf = np.clip(xf, 0, width - 1.001)
    yf = np.clip(yf, 0, height - 1.001)
    x0 = np.floor(xf).astype(np.int32)
    y0 = np.floor(yf).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (xf - x0)[..., None].astype(np.float32)
    wy = (yf - y0)[..., None].astype(np.float32)
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return sampled
