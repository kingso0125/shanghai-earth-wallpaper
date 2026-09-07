"""Decode the documented NASA GIBS enhanced IR palette before reprojection.

Source: https://gibs.earthdata.nasa.gov/colormaps/v1.3/Clean_Longwave_Infrared_Window_Band.xml
This browse palette is lossy: extreme cold grey colours overlap warm greys.
We conservatively choose the warm branch for ambiguous greys. These values
are a cloud appearance proxy, not a quantitative cloud-height/opacity product.
"""
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

@lru_cache(maxsize=1)
def _lookup() -> np.ndarray:
    entries = ET.parse(Path(__file__).parent / "data/infrared-colormap.xml")
    palette, temperatures = [], []
    for entry in entries.iter("ColorMapEntry"):
        if entry.get("transparent") == "true":
            continue
        values = entry.attrib["sourceValue"][1:-1].split(",")
        temperature = sum(map(float, values)) / 2
        rgb = list(map(int, entry.attrib["rgb"].split(",")))
        if max(rgb) == min(rgb) and temperature < -70:
            continue
        palette.append(rgb)
        temperatures.append(temperature)
    palette = np.array(palette, dtype=np.float32)
    temperatures = np.array(temperatures, dtype=np.float32)
    grid = np.indices((64, 64, 64), dtype=np.float32).reshape(3, -1).T * 4 + 1.5
    result = np.empty(len(grid), dtype=np.float32)
    for start in range(0, len(grid), 2048):
        distance = np.sum((grid[start:start + 2048, None] - palette[None]) ** 2, axis=-1)
        result[start:start + 2048] = temperatures[np.argmin(distance, axis=-1)]
    return result.reshape(64, 64, 64)


def coldness(rgb: np.ndarray) -> np.ndarray:
    """0..1 proxy increasing continuously as cloud-top temperature falls."""
    quantized = np.clip(rgb[..., :3], 0, 255).astype(np.uint8) >> 2
    temperature = _lookup()[quantized[..., 0], quantized[..., 1], quantized[..., 2]]
    return np.clip((40.0 - temperature) / 140.0, 0, 1).astype(np.float32)


def sample_coldness(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    # Decode only the native texels used by this view. Decoding the complete 8K
    # plate creates several unnecessary 128 MB arrays on the 2 GB renderer.
    # We still interpolate temperature, not discontinuous palette RGB hues.
    height, width = rgb.shape[:2]
    xf = ((lon + np.pi) / (2.0 * np.pi) * width) % width
    yf = np.clip((np.pi / 2.0 - lat) / np.pi * (height - 1), 0, height - 1)
    x0, y0 = np.floor(xf).astype(np.int32), np.floor(yf).astype(np.int32)
    x1, y1 = (x0 + 1) % width, np.minimum(y0 + 1, height - 1)
    wx, wy = (xf - x0).astype(np.float32), (yf - y0).astype(np.float32)
    top = coldness(rgb[y0, x0]) * (1.0 - wx) + coldness(rgb[y0, x1]) * wx
    bottom = coldness(rgb[y1, x0]) * (1.0 - wx) + coldness(rgb[y1, x1]) * wx
    return top * (1.0 - wy) + bottom * wy
