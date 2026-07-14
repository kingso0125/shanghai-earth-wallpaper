from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .config import RenderPreset, presets_for_location
from .geometry import camera_grid
from .lighting import daylight


def _metrics(path: Path, preset: RenderPreset, observation: datetime) -> dict:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if (rgb.shape[1], rgb.shape[0]) != preset.size:
        raise ValueError(f"{path} has incorrect dimensions")
    luminance = rgb.mean(axis=-1)
    yy, xx = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]]
    cx, cy = preset.center_px
    earth_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= preset.globe_radius_px**2
    earth_values = luminance[earth_mask]
    _, _, visible, _, vectors = camera_grid(preset)
    day_fraction = float(daylight(vectors, observation)[visible].mean())
    gx = np.abs(np.diff(luminance, axis=1)).mean()
    gy = np.abs(np.diff(luminance, axis=0)).mean()
    safe_height = 560 if preset.name == "lock" else 470
    return {
        "safe_area_mean": float(luminance[:safe_height].mean()),
        "earth_mean": float(earth_values.mean()),
        "earth_p95": float(np.percentile(earth_values, 95)),
        "earth_clipped_fraction": float((earth_values > 0.985).mean()),
        "detail_gradient": float(gx + gy),
        "day_fraction": day_fraction,
        "minimum_expected_brightness": 0.17 + day_fraction * 0.08,
    }


def audit(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    observation = datetime.fromisoformat(manifest["observation_utc"].replace("Z", "+00:00"))
    rendered = datetime.fromisoformat(manifest["rendered_utc"].replace("Z", "+00:00"))
    age_hours = (rendered.astimezone(UTC) - observation.astimezone(UTC)).total_seconds() / 3600
    target = manifest.get("target", {})
    lock, home = presets_for_location(
        float(target.get("latitude", 31.2304)),
        float(target.get("longitude", 121.4737)),
    )
    result = {
        "observation_age_hours": age_hours,
        "lock": _metrics(directory / "lock.jpg", lock, observation),
        "home": _metrics(directory / "home.jpg", home, observation),
    }
    failures = []
    if "CIRA SLIDER" in manifest["source"] and manifest.get("render_mode") not in {
        "fused_geostationary_plate_shanghai_meridian",
        "fused_geostationary_plate_location_meridian",
        "fused_geostationary_plate_location_centered",
    }:
        failures.append("CIRA observation was not transformed as one fused Earth/cloud plate")
    if manifest["source_status"] == "fresh" and not 0 <= age_hours <= 3.0:
        failures.append(f"fresh observation age is {age_hours:.2f} hours")
    for name in ("lock", "home"):
        metrics = result[name]
        if metrics["safe_area_mean"] > 0.055:
            failures.append(f"{name} safe area is too bright")
        if not metrics["minimum_expected_brightness"] <= metrics["earth_mean"] <= 0.68:
            failures.append(f"{name} Earth brightness is outside target")
        if metrics["earth_clipped_fraction"] > 0.12:
            failures.append(f"{name} has excessive highlight clipping")
        if metrics["detail_gradient"] < 0.010:
            failures.append(f"{name} lacks visible detail")
    result["passed"] = not failures
    result["failures"] = failures
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit generated Earth wallpapers")
    parser.add_argument("directory", nargs="?", type=Path, default=Path("output/current"))
    args = parser.parse_args(argv)
    result = audit(args.directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
