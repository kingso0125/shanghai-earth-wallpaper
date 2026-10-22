import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from earthwall.delivery import copy_server_release


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 22, 15, 30, tzinfo=UTC)
        self.manifest = {
            "rendered_utc": "2026-10-22T15:20:00.123456Z",
            "target": {"latitude": 31.2304, "longitude": 121.4737, "name": "Shanghai"},
            "artifacts": {
                name: {"file": name + ".jpg", "sha256": hashlib.sha256(name.encode()).hexdigest()}
                for name in ("lock", "home")
            },
        }

    def test_one_immutable_release_copied_after_checks(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "earthwall.delivery._request", side_effect=[json.dumps(self.manifest).encode(), b"lock", b"home"]
        ) as request, patch("earthwall.delivery.audit", return_value={"passed": True}) as audit:
            output = Path(directory) / "public"
            copy_server_release(output, now=self.now)
            self.assertEqual((output / "home.jpg").read_bytes(), b"home")
            self.assertIn("/releases/2026-10-22T152000-123456Z/lock.jpg", request.call_args_list[1].args[0])
            audit.assert_called_once()

    def test_bad_download_keeps_previous_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "public"
            output.mkdir()
            (output / "lock.jpg").write_bytes(b"previous")
            with patch("earthwall.delivery._request", side_effect=[json.dumps(self.manifest).encode(), b"corrupt"]):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    copy_server_release(output, now=self.now)
            self.assertEqual((output / "lock.jpg").read_bytes(), b"previous")

    def test_stale_future_or_naive_timestamp_rejected(self):
        for stamp in ("2026-10-22T13:00:00Z", "2026-10-22T15:31:00Z", "2026-10-22T15:20:00"):
            with self.subTest(stamp=stamp), tempfile.TemporaryDirectory() as directory:
                manifest = {**self.manifest, "rendered_utc": stamp}
                with patch("earthwall.delivery._request", return_value=json.dumps(manifest).encode()) as request:
                    with self.assertRaises(ValueError):
                        copy_server_release(Path(directory) / "public", now=self.now)
                    self.assertEqual(request.call_count, 1)

    def test_failed_visual_audit_keeps_previous_pair(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "earthwall.delivery._request", side_effect=[json.dumps(self.manifest).encode(), b"lock", b"home"]
        ), patch("earthwall.delivery.audit", return_value={"passed": False, "failures": ["blank globe"]}):
            output = Path(directory) / "public"
            output.mkdir()
            (output / "home.jpg").write_bytes(b"previous")
            with self.assertRaisesRegex(ValueError, "blank globe"):
                copy_server_release(output, now=self.now)
            self.assertEqual((output / "home.jpg").read_bytes(), b"previous")
