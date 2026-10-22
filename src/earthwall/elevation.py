"""World-space terrain normals from NOAA ETOPO heights, in meters."""
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import sample_equirectangular


ELEVATION = Path(__file__).parent / "data" / "earth-elevation-5400.png"
RELIEF_EXAGGERATION = 12.0


def terrain_normals(
    path: Path, latitude: np.ndarray, longitude: np.ndarray,
    vectors: np.ndarray, land: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        meters = np.asarray(image, dtype=np.uint16)[..., None]
    height, width = meters.shape[:2]
    dlat, dlon = np.pi / height, 2 * np.pi / width

    def sample(lat, lon):
        return sample_equirectangular(meters, lat, lon)[..., 0]

    center = sample(latitude, longitude)
    east_slope = (sample(latitude, longitude + dlon) - sample(latitude, longitude - dlon))
    east_slope /= 2 * 6371008.8 * dlon * np.maximum(np.cos(latitude), .08)
    north_slope = (sample(np.clip(latitude + dlat, -np.pi / 2, np.pi / 2), longitude)
                   - sample(np.clip(latitude - dlat, -np.pi / 2, np.pi / 2), longitude))
    north_slope /= 2 * 6371008.8 * dlat
    gain = RELIEF_EXAGGERATION * land
    east_slope = np.clip(east_slope * gain, -.9, .9)
    north_slope = np.clip(north_slope * gain, -.9, .9)
    east = np.stack((-np.sin(longitude), np.cos(longitude), np.zeros_like(longitude)), axis=-1)
    north = np.stack((-np.sin(latitude) * np.cos(longitude),
                      -np.sin(latitude) * np.sin(longitude), np.cos(latitude)), axis=-1)
    normal = vectors - east_slope[..., None] * east - north_slope[..., None] * north
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    return normal.astype(np.float32), center.astype(np.float32)
