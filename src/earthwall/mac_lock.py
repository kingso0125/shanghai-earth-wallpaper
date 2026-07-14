from __future__ import annotations

import argparse
import plistlib
from datetime import datetime
from pathlib import Path


def _choice(image: Path) -> dict:
    configuration = plistlib.dumps(
        {
            "type": "imageFile",
            "url": {"relative": image.resolve().as_uri()},
        },
        fmt=plistlib.FMT_BINARY,
    )
    return {
        "Configuration": configuration,
        "Files": [],
        "Provider": "com.apple.wallpaper.choice.image",
    }


def configure(index: Path, image: Path) -> int:
    data = plistlib.loads(index.read_bytes())
    content = {
        "Choices": [_choice(image)],
        "EncodedOptionValues": "$null",
        "Shuffle": "$null",
    }
    changed = 0

    def visit(node) -> None:
        nonlocal changed
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "Idle" and isinstance(value, dict):
                    value["Content"] = content.copy()
                    value["LastSet"] = datetime.now()
                    changed += 1
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(data)
    if changed == 0:
        raise ValueError("no macOS Idle wallpaper entries found")
    backup = index.with_suffix(".plist.earthwall-backup")
    if not backup.exists():
        backup.write_bytes(index.read_bytes())
    temporary = index.with_suffix(".plist.earthwall-new")
    temporary.write_bytes(plistlib.dumps(data, fmt=plistlib.FMT_BINARY, sort_keys=False))
    temporary.replace(index)
    return changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Set the macOS Idle image provider")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path.home()
        / "Library/Application Support/com.apple.wallpaper/Store/Index.plist",
    )
    args = parser.parse_args(argv)
    print(configure(args.index, args.image))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
