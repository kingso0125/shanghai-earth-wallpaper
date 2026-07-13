from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render import render_pair
from .sources import acquire


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Render Shanghai-centered Earth wallpapers")
    result.add_argument("--cache", type=Path, default=Path("cache"))
    result.add_argument("--output", type=Path, default=Path("output/current"))
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    observation = acquire(args.cache)
    manifest = render_pair(observation, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

