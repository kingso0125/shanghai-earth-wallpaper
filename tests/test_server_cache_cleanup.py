from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.cleanup_server_cache import cleanup


class ServerCacheCleanupTests(unittest.TestCase):
    def test_deletes_previous_day_but_preserves_latest_fallback(self) -> None:
        timezone = datetime.now().astimezone().tzinfo
        now = datetime(2026, 8, 8, 1, 35, tzinfo=timezone)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old_visible = root / "himawari-20260806T0000Z-visible.png"
            latest_visible = root / "himawari-20260807T2300Z-visible.png"
            static = root / "blue-marble.jpg"
            for path in (old_visible, latest_visible, static):
                path.write_bytes(b"data")
            old_time = (now - timedelta(days=2)).timestamp()
            latest_time = (now - timedelta(hours=2)).timestamp()
            os.utime(old_visible, (old_time, old_time))
            os.utime(latest_visible, (latest_time, latest_time))
            os.utime(static, (old_time, old_time))

            result = cleanup(root, now=now)

            self.assertFalse(old_visible.exists())
            self.assertTrue(latest_visible.exists())
            self.assertTrue(static.exists())
            self.assertEqual(result["deleted_count"], 1)

    def test_dry_run_does_not_delete(self) -> None:
        timezone = datetime.now().astimezone().tzinfo
        now = datetime(2026, 8, 8, 1, 35, tzinfo=timezone)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "cira-gk2a-20260806T0000Z-geocolor.png"
            latest = root / "cira-gk2a-20260807T2300Z-geocolor.png"
            older.write_bytes(b"old")
            latest.write_bytes(b"new")
            os.utime(older, ((now - timedelta(days=2)).timestamp(),) * 2)
            os.utime(latest, ((now - timedelta(hours=2)).timestamp(),) * 2)

            result = cleanup(root, now=now, dry_run=True)

            self.assertTrue(older.exists())
            self.assertEqual(result["deleted_count"], 1)


if __name__ == "__main__":
    unittest.main()
