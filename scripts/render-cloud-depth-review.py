"""Cloud-only comparison: reuse the accepted study's observation and sunlight."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from earthwall.sources import Observation, sha256
from earthwall.preview_v2 import render_production_pair, render_production_mac_pair
from earthwall.qa import audit as audit_phone
from earthwall.mac_qa import audit as audit_mac


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phone-only", action="store_true", help="Check the server's phone-only workload")
    args = parser.parse_args()
    reference = json.loads((args.reference / "manifest.json").read_text())
    if reference["source"] != "NASA GIBS / JMA Himawari-9":
        raise ValueError("This frozen comparison expects the approved Himawari study")
    timestamp = datetime.fromisoformat(reference["observation_utc"].replace("Z", "+00:00"))
    lighting = datetime.fromisoformat(reference["lighting_utc"].replace("Z", "+00:00"))
    stamp = timestamp.strftime("%Y%m%dT%H%MZ")
    cache = args.cache / "cinematic-v2"
    observation = Observation(
        timestamp=timestamp, visible=cache / f"himawari-{stamp}-visible-8k.png",
        infrared=cache / f"himawari-{stamp}-infrared-8k.png", geocolor=None,
        base=cache / "blue-marble-8k.png", lights=cache / "city-lights-8k.png",
        terrain=cache / "terrain-relief-8k.png", water_mask=cache / "water-mask-8k.png",
        status="frozen-comparison", source=reference["source"],
    )
    if sha256(observation.visible) != reference["observation_asset_sha256"]:
        raise ValueError("Satellite source differs from the approved comparison")
    target = reference["target"]
    options = dict(target_latitude=target["latitude"], target_longitude=target["longitude"],
                   target_name=target["name"], lighting_time=lighting)
    render_production_pair(observation, args.output, **options)
    if not args.phone_only:
        render_production_mac_pair(observation, args.output, **options)
    manifests = ("manifest.json",) if args.phone_only else ("manifest.json", "mac-manifest.json")
    for name in manifests:
        path = args.output / name
        manifest = json.loads(path.read_text())
        manifest.update(preview_only=True, delivery_state="candidate-not-published",
                        cloud_material="observed-volume-r2", comparison_reference=str(args.reference))
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    qa = {"phone": audit_phone(args.output)}
    if not args.phone_only:
        qa["mac"] = audit_mac(args.output)
    (args.output / "review-qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps({"passed": all(v["passed"] for v in qa.values()), "output": str(args.output)}))


if __name__ == "__main__":
    main()
