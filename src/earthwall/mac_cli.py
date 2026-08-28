from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import SHANGHAI, resolve_target
from .preview_sources import upgrade_v2_observation
from .preview_v2 import render_production_mac_pair
from .sources import Observation, _newest_cached_pair, acquire_for_target


def acquire_mac(cache: Path, longitude: float = SHANGHAI[1]) -> Observation:
    """Prefer a newer server-validated raw pair without changing phone acquisition."""
    observation = acquire_for_target(cache, longitude)
    if observation.source.startswith("EUMETSAT"):
        return observation
    manifest_path = cache / "server-manifest.json"
    cached = _newest_cached_pair(cache)
    if not manifest_path.exists() or cached is None:
        return observation
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        timestamp, visible, infrared = cached
        validated = datetime.fromisoformat(
            manifest["observation_utc"].replace("Z", "+00:00")
        )
        if (
            manifest.get("source_status") == "fresh"
            and timestamp == validated
            and timestamp > observation.timestamp
        ):
            return Observation(
                timestamp=timestamp,
                visible=visible,
                infrared=infrared,
                geocolor=None,
                base=observation.base,
                lights=observation.lights,
                status="fresh",
                source=f"{manifest['source']} (synchronized raw observation)",
                satellite_longitude=140.7,
                terrain=observation.terrain,
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return observation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render native Mac Earth wallpapers")
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--output", type=Path, default=Path("output/mac"))
    parser.add_argument("--latitude", type=float, default=SHANGHAI[0])
    parser.add_argument("--longitude", type=float, default=SHANGHAI[1])
    parser.add_argument("--location-name", default="Shanghai")
    args = parser.parse_args(argv)
    latitude, longitude, location_name = resolve_target(
        args.latitude, args.longitude, args.location_name
    )
    observation = acquire_mac(args.cache, longitude)
    observation = upgrade_v2_observation(args.cache, observation)
    manifest = render_production_mac_pair(
        observation,
        args.output,
        target_latitude=latitude,
        target_longitude=longitude,
        target_name=location_name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
