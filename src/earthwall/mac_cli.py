from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SHANGHAI
from .render import render_mac_pair
from .sources import acquire


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render native Mac Earth wallpapers")
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--output", type=Path, default=Path("output/mac"))
    parser.add_argument("--latitude", type=float, default=SHANGHAI[0])
    parser.add_argument("--longitude", type=float, default=SHANGHAI[1])
    parser.add_argument("--location-name", default="Shanghai")
    args = parser.parse_args(argv)
    observation = acquire(args.cache)
    manifest = render_mac_pair(
        observation,
        args.output,
        target_latitude=args.latitude,
        target_longitude=args.longitude,
        target_name=args.location_name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
