from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .config import SHANGHAI, resolve_target
from .preview_sources import upgrade_v2_observation
from .preview_v2 import render_production_mac_pair
from .sources import Observation, _acquire_static, _newest_cached_pair, acquire_for_target
from .location import Location
from .target import follow_server


def acquire_mac(cache: Path, longitude: float = SHANGHAI[1]) -> Observation:
    """Use the synchronized, validated observation before another network fetch."""
    manifest_path = cache / "server-manifest.json"
    cached = _newest_cached_pair(cache)
    if not manifest_path.exists() or cached is None:
        return acquire_for_target(cache, longitude)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        timestamp, visible, infrared = cached
        validated = datetime.fromisoformat(
            manifest["observation_utc"].replace("Z", "+00:00")
        )
        if (
            manifest.get("source_status") == "fresh"
            and manifest.get("source", "").startswith("NASA GIBS / JMA")
            and timestamp == validated
            and 0 <= (datetime.now(UTC) - timestamp).total_seconds() <= 3 * 3600
            and abs((longitude - 140.7 + 180) % 360 - 180) < 75
        ):
            base, lights, terrain = _acquire_static(cache)
            return Observation(
                timestamp=timestamp,
                visible=visible,
                infrared=infrared,
                geocolor=None,
                base=base,
                lights=lights,
                status="fresh",
                source=f"{manifest['source']} (synchronized raw observation)",
                satellite_longitude=140.7,
                terrain=terrain,
            )
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return acquire_for_target(cache, longitude)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render native Mac Earth wallpapers")
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--output", type=Path, default=Path("output/mac"))
    parser.add_argument("--latitude", type=float, default=SHANGHAI[0])
    parser.add_argument("--longitude", type=float, default=SHANGHAI[1])
    parser.add_argument("--location-name", default="Shanghai")
    parser.add_argument("--follow-server", action="store_true")
    args = parser.parse_args(argv)
    latitude, longitude, location_name = resolve_target(
        args.latitude, args.longitude, args.location_name
    )
    if args.follow_server:
        target = follow_server(args.cache, Location(latitude, longitude, location_name))
        latitude, longitude, location_name = target.latitude, target.longitude, target.name
    observation = acquire_mac(args.cache, longitude)
    observation = upgrade_v2_observation(args.cache, observation, require_calibrated=True)
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
