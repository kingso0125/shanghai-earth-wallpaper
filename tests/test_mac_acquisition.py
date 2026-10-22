import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from earthwall.mac_cli import acquire_mac


class MacAcquisitionTests(unittest.TestCase):
    def test_valid_synced_pair_does_not_wait_for_external_network(self):
        stamp = datetime.now(UTC) - timedelta(minutes=30)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "server-manifest.json").write_text(json.dumps({
                "observation_utc": stamp.isoformat(), "source_status": "fresh", "source": "NASA GIBS / JMA Himawari-9",
            }))
            with patch("earthwall.mac_cli._newest_cached_pair", return_value=(stamp, Path("vis"), Path("ir"))), patch(
                "earthwall.mac_cli._acquire_static", return_value=(Path("base"), Path("lights"), None)
            ), patch("earthwall.mac_cli.acquire_for_target") as network:
                result = acquire_mac(cache)
                self.assertEqual(result.timestamp, stamp)
                self.assertIn("synchronized", result.source)
                network.assert_not_called()

    def test_stale_or_out_of_coverage_pair_uses_normal_source_selection(self):
        for age, longitude in ((4, 121.47), (.5, 2.35)):
            stamp = datetime.now(UTC) - timedelta(hours=age)
            with self.subTest(age=age, longitude=longitude), tempfile.TemporaryDirectory() as directory:
                cache = Path(directory)
                (cache / "server-manifest.json").write_text(json.dumps({
                    "observation_utc": stamp.isoformat(), "source_status": "fresh", "source": "NASA GIBS / JMA Himawari-9",
                }))
                with patch("earthwall.mac_cli._newest_cached_pair", return_value=(stamp, Path("vis"), Path("ir"))), patch(
                    "earthwall.mac_cli.acquire_for_target", return_value="fallback"
                ) as network:
                    self.assertEqual(acquire_mac(cache, longitude), "fallback")
                    network.assert_called_once_with(cache, longitude)
