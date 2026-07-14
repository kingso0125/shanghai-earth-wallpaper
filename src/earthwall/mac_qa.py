from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .config import mac_presets_for_location
from .geometry import camera_grid
from .lighting import daylight


def _metrics(path: Path, preset, lighting: datetime) -> dict:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if (rgb.shape[1], rgb.shape[0]) != preset.size:
        raise ValueError(f"{path} has incorrect dimensions")
    luminance = np.sum(
        rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1
    )
    yy, xx = np.mgrid[: rgb.shape[0], : rgb.shape[1]]
    cx, cy = preset.center_px
    earth_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= preset.globe_radius_px**2
    space_mask = ~earth_mask
    earth = luminance[earth_mask]
    space = luminance[space_mask]
    _, _, visible, _, vectors = camera_grid(preset)
    day_fraction = float(daylight(vectors, lighting)[visible].mean())
    return {
        "earth_mean": float(earth.mean()),
        "earth_p95": float(np.percentile(earth, 95)),
        "earth_clipped_fraction": float((earth > 0.985).mean()),
        "space_mean": float(space.mean()),
        "space_p99": float(np.percentile(space, 99)),
        "detail_gradient": float(
            np.abs(np.diff(luminance, axis=1)).mean()
            + np.abs(np.diff(luminance, axis=0)).mean()
        ),
        "day_fraction": day_fraction,
        "minimum_expected_brightness": 0.08 + day_fraction * 0.12,
        "globe_fully_visible": bool(
            cx - preset.globe_radius_px >= 0
            and cy - preset.globe_radius_px >= 0
            and cx + preset.globe_radius_px < preset.size[0]
            and cy + preset.globe_radius_px < preset.size[1]
        ),
    }


def audit(directory: Path) -> dict:
    manifest = json.loads((directory / "mac-manifest.json").read_text(encoding="utf-8"))
    observation = datetime.fromisoformat(manifest["observation_utc"].replace("Z", "+00:00"))
    lighting = datetime.fromisoformat(
        manifest.get("lighting_utc", manifest["observation_utc"]).replace("Z", "+00:00")
    )
    rendered = datetime.fromisoformat(manifest["rendered_utc"].replace("Z", "+00:00"))
    age_hours = (rendered.astimezone(UTC) - observation.astimezone(UTC)).total_seconds() / 3600
    target = manifest["target"]
    presets = mac_presets_for_location(float(target["latitude"]), float(target["longitude"]))
    result = {"observation_age_hours": age_hours}
    failures = []
    for preset in presets:
        metrics = _metrics(directory / f"mac-{preset.name}.jpg", preset, lighting)
        result[preset.name] = metrics
        if preset.name == "lock" and not metrics["globe_fully_visible"]:
            failures.append(f"mac {preset.name} globe is cropped")
        if preset.name == "home" and metrics["globe_fully_visible"]:
            failures.append("mac home should use the close-up hemisphere composition")
        if not metrics["minimum_expected_brightness"] <= metrics["earth_mean"] <= 0.70:
            failures.append(f"mac {preset.name} Earth brightness is outside target")
        if metrics["earth_clipped_fraction"] > 0.10:
            failures.append(f"mac {preset.name} has excessive highlight clipping")
        if metrics["space_mean"] > 0.055:
            failures.append(f"mac {preset.name} space is too bright")
        if metrics["detail_gradient"] < 0.006:
            failures.append(f"mac {preset.name} lacks visible detail")
    if manifest["source_status"] == "fresh" and not 0 <= age_hours <= 3.0:
        failures.append(f"fresh observation age is {age_hours:.2f} hours")
    result["passed"] = not failures
    result["failures"] = failures
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit Mac Earth wallpapers")
    parser.add_argument("directory", nargs="?", type=Path, default=Path("output/mac"))
    args = parser.parse_args(argv)
    result = audit(args.directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
