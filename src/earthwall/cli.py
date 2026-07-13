from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SHANGHAI
from .render import render_pair
from .sources import acquire


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
    observation = acquire(args.cache)
    manifest = render_pair(
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
