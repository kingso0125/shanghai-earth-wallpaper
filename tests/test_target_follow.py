import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from earthwall.location import Location
from earthwall.target import follow_server
from earthwall.preview_v2 import presets_for_location, presets_for_mac_location
from earthwall.sources import acquire_for_target, Observation


class TargetFollowTests(unittest.TestCase):
    def test_network_failure_keeps_last_accepted_city(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            shanghai = Location(31.2304, 121.4737, "Shanghai")
            data = json.dumps({"target": {"latitude": 48.8566, "longitude": 2.3522, "name": "Paris"}}).encode()
            with patch("earthwall.target._request", return_value=data):
                self.assertEqual(follow_server(cache, shanghai).name, "Paris")
            with patch("earthwall.target._request", side_effect=OSError("offline")):
                self.assertEqual(follow_server(cache, shanghai).name, "Paris")

    def test_invalid_remote_coordinate_does_not_replace_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Location(31.2304, 121.4737, "Shanghai")
            with patch("earthwall.target._request", return_value=b'{"target":{"latitude":999,"longitude":1,"name":"bad"}}'):
                self.assertEqual(follow_server(Path(directory), original), original)

    def test_home_camera_follows_latitude_without_changing_composition(self):
        for factory in (presets_for_location, presets_for_mac_location):
            _lock, shanghai = factory(31.2304, 121.4737)
            _lock, paris = factory(48.8566, 2.3522)
            self.assertAlmostEqual(paris.target_lat - shanghai.target_lat, 48.8566 - 31.2304)
            self.assertEqual(paris.center_px, shanghai.center_px)
            self.assertEqual(paris.globe_radius_px, shanghai.globe_radius_px)

    def test_paris_does_not_require_himawari_to_be_online(self):
        from datetime import datetime, UTC
        with patch("earthwall.sources.acquire", side_effect=AssertionError("wrong satellite")), patch(
            "earthwall.sources._acquire_static", return_value=(Path("base"), Path("lights"), None)
        ), patch("earthwall.sources._acquire_eumetsat_layers", return_value=(datetime.now(UTC), Path("vis"), Path("ir"))):
            self.assertIn("EUMETSAT", acquire_for_target(Path("cache"), 2.35).source)
