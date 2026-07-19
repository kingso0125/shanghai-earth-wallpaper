from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path


def smoothstep(low, high, value):
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    scale = max(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
    )
    left = (resized.width - target_width) // 2
    top = (resized.height - target_height) // 2
    return resized.crop((left, top, left + target_width, top + target_height))


def space_background(
    size: tuple[int, int], seed: int = 1214737, asset: Path | None = None
) -> np.ndarray:
    if asset is not None and asset.exists():
        generated = _cover(Image.open(asset).convert("RGB"), size)
        rgb = np.asarray(generated, dtype=np.float32) / 255.0
        # The generated plate supplies texture only; keep the lock-screen safe area restrained.
        luminance = rgb.mean(axis=-1, keepdims=True)
        rgb = luminance + (rgb - luminance) * 0.68
        return np.clip(np.power(rgb, 1.12) * 0.48, 0.0, 0.18)

    width, height = size
    rng = np.random.default_rng(seed)
    small = rng.random((max(8, height // 40), max(8, width // 40))).astype(np.float32)
    nebula = Image.fromarray(np.uint8(small * 255), "L").resize(size, Image.Resampling.BICUBIC)
    nebula = nebula.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 15))
    haze = np.asarray(nebula, dtype=np.float32) / 255.0
    haze = np.clip((haze - 0.42) * 0.16, 0.0, 0.035)
    background = np.zeros((height, width, 3), dtype=np.float32)
    background[..., 0] = 0.006 + haze * 0.95
    background[..., 1] = 0.008 + haze * 0.66
    background[..., 2] = 0.012 + haze * 0.45

    stars = Image.new("RGB", size)
    draw = ImageDraw.Draw(stars)
    for _ in range(max(18, width * height // 140000)):
        x = int(rng.integers(16, max(17, width - 16)))
        y = int(rng.integers(16, max(17, height - 16)))
        radius = float(rng.choice([0.5, 0.7, 0.9, 1.3], p=[0.44, 0.34, 0.17, 0.05]))
        tint = tuple(
            int(value)
            for value in rng.choice([(155, 187, 218), (225, 221, 195), (191, 170, 211)])
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=tint)
    stars = stars.filter(ImageFilter.GaussianBlur(0.35))
    star_array = np.asarray(stars, dtype=np.float32) / 255.0
    return np.clip(background + star_array * 0.72, 0.0, 1.0)


def atmosphere(mask: np.ndarray, sz: np.ndarray, size: tuple[int, int]):
    edge = np.clip(1.0 - sz, 0.0, 1.0)
    rim = (
        np.power(edge, 4.8) * 0.62
        + np.power(edge, 12.0) * 0.38
    ) * mask
    mask_image = Image.fromarray(np.uint8(mask * 255), "L")
    near_blur = mask_image.filter(
        ImageFilter.GaussianBlur(radius=max(8, size[0] * 0.013))
    )
    far_blur = mask_image.filter(
        ImageFilter.GaussianBlur(radius=max(18, size[0] * 0.034))
    )
    near = np.clip(np.asarray(near_blur, dtype=np.float32) / 255.0 - mask, 0.0, 1.0)
    far = np.clip(np.asarray(far_blur, dtype=np.float32) / 255.0 - mask, 0.0, 1.0)
    halo = np.clip(near * 0.74 + far * 0.26, 0.0, 1.0)
    return rim.astype(np.float32), halo.astype(np.float32)
