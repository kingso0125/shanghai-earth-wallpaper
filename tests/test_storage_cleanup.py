import unittest
from datetime import date

from scripts.cleanup_github_storage import select_artifact_ids, select_cache_ids


class StorageCleanupTests(unittest.TestCase):
    def test_deletes_old_earth_caches_but_keeps_latest(self):
        caches = [
            {"id": 1, "key": "earthwall-cache-Linux-1", "created_at": "2026-07-12T14:00:00Z"},
            {"id": 2, "key": "earthwall-cache-Linux-2", "created_at": "2026-07-12T16:30:00Z"},
            {"id": 3, "key": "earthwall-cache-Linux-3", "created_at": "2026-07-13T17:00:00Z"},
            {"id": 4, "key": "setup-python-Linux", "created_at": "2026-07-01T00:00:00Z"},
        ]
        self.assertEqual(select_cache_ids(caches, date(2026, 7, 14)), [1, 2])

    def test_keeps_latest_cache_even_when_every_cache_is_old(self):
        caches = [
            {"id": 1, "key": "earthwall-cache-Linux-1", "created_at": "2026-07-10T00:00:00Z"},
            {"id": 2, "key": "earthwall-cache-Linux-2", "created_at": "2026-07-11T00:00:00Z"},
        ]
        self.assertEqual(select_cache_ids(caches, date(2026, 7, 14)), [1])

    def test_deletes_artifacts_from_prior_shanghai_dates(self):
        artifacts = [
            {"id": 10, "created_at": "2026-07-13T15:59:59Z"},
            {"id": 11, "created_at": "2026-07-13T16:00:00Z"},
        ]
        self.assertEqual(select_artifact_ids(artifacts, date(2026, 7, 14)), [10])


if __name__ == "__main__":
    unittest.main()
