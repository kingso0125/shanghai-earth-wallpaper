import tempfile
import unittest
from pathlib import Path

from earthwall.location import LocationStore
from earthwall.location_api import LocationApplication


class FakePublisher:
    def __init__(self):
        self.locations = []

    def publish(self, location):
        self.locations.append(location)
        return {"artifacts": {"home": {"sha256": "a" * 64}}}


class LocationApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.publisher = FakePublisher()
        self.application = LocationApplication(
            LocationStore(Path(self.temporary.name) / "location.json", 80.0),
            self.publisher,
            "test-token-with-at-least-24-characters",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_rejects_missing_token(self):
        status, response = self.application.update(
            "", {"latitude": 23.1291, "longitude": 113.2644}
        )
        self.assertEqual(status, 401)
        self.assertEqual(response["error"], "unauthorized")

    def test_ignores_movement_below_80_km(self):
        status, response = self.application.update(
            "Bearer test-token-with-at-least-24-characters",
            {"latitude": 31.2100, "longitude": 121.1000, "name": "Nearby"},
        )
        self.assertEqual(status, 200)
        self.assertFalse(response["changed"])
        self.assertEqual(self.publisher.locations, [])

    def test_guangzhou_change_publishes_immediately(self):
        status, response = self.application.update(
            "Bearer test-token-with-at-least-24-characters",
            {
                "latitude": 23.1291,
                "longitude": 113.2644,
                "accuracy": 25,
                "name": "Guangzhou",
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(response["changed"])
        self.assertEqual(response["target"]["name"], "Guangzhou")
        self.assertEqual(len(self.publisher.locations), 1)
        self.assertEqual(response["version"], "a" * 16)


if __name__ == "__main__":
    unittest.main()
