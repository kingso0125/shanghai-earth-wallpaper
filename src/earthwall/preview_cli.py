from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SHANGHAI
from .preview_v2 import render_preview_pair
from .preview_sources import upgrade_preview_observation
from .sources import acquire


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Render isolated cinematic Earth V2 previews")
    result.add_argument("--cache", type=Path, default=Path("cache"))
    result.add_argument("--output", type=Path, default=Path("output/preview-v2"))
    result.add_argument("--latitude", type=float, default=SHANGHAI[0])
    result.add_argument("--longitude", type=float, default=SHANGHAI[1])
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    observation = acquire(args.cache)
    observation = upgrade_preview_observation(args.cache, observation)
    manifest = render_preview_pair(
        observation,
        args.output,
        latitude=args.latitude,
        longitude=args.longitude,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
