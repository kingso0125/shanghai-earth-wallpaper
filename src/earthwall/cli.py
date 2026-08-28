from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SHANGHAI, resolve_target
from .preview_v2 import render_production_pair
from .sources import acquire_for_target


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Render location-aware Earth wallpapers")
    result.add_argument("--cache", type=Path, default=Path("cache"))
    result.add_argument("--output", type=Path, default=Path("output/current"))
    result.add_argument("--latitude", type=float, default=SHANGHAI[0])
    result.add_argument("--longitude", type=float, default=SHANGHAI[1])
    result.add_argument("--location-name", default="Shanghai")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    latitude, longitude, location_name = resolve_target(
        args.latitude, args.longitude, args.location_name
    )
    observation = acquire_for_target(args.cache, longitude)
    manifest = render_production_pair(
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
