"""Generate phone/Mac candidates from one observation and one lighting instant."""
import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from earthwall.sources import acquire_for_target
from earthwall.preview_sources import upgrade_v2_observation
from earthwall.preview_v2 import render_production_pair, render_production_mac_pair
from earthwall.qa import audit as audit_phone
from earthwall.mac_qa import audit as audit_mac

p = argparse.ArgumentParser()
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
observation = upgrade_v2_observation(args.cache, acquire_for_target(args.cache, 121.4737))
lighting = datetime.now(UTC)
render_production_pair(observation, args.output, lighting_time=lighting)
render_production_mac_pair(observation, args.output, lighting_time=lighting)
for name in ("manifest.json", "mac-manifest.json"):
    path = args.output / name
    manifest = json.loads(path.read_text())
    manifest["preview_only"] = True
    manifest["delivery_state"] = "candidate-not-published"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
qa = {"phone": audit_phone(args.output), "mac": audit_mac(args.output)}
(args.output / "review-qa.json").write_text(json.dumps(qa, indent=2) + "\n")
print(json.dumps({"observation_utc": observation.timestamp.isoformat(), "lighting_utc": lighting.isoformat(),
                  "passed": all(v["passed"] for v in qa.values()), "output": str(args.output)}, indent=2))
