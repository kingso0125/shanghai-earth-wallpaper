import unittest
from pathlib import Path

from earthwall.config import (
    HOME,
    LOCK,
    MAC_HOME,
    MAC_LOCK,
    SHANGHAI,
    mac_presets_for_location,
)


class MacContractTests(unittest.TestCase):
    def test_phone_contract_is_unchanged(self):
        self.assertEqual(LOCK.size, (1320, 2868))
        self.assertEqual(LOCK.center_px, (660.0, 1470.0))
        self.assertEqual(HOME.size, (1320, 2868))
        self.assertEqual(HOME.center_px, (660.0, 2440.0))
        workflow = Path(".github/workflows/hourly-wallpaper.yml").read_text()
        self.assertNotIn("earthwall-mac", workflow)
        self.assertNotIn("mac-home.jpg", workflow)
        self.assertNotIn("mac-lock.jpg", workflow)

    def test_mac_presets_are_native_and_location_centered(self):
        lock, home = mac_presets_for_location(*SHANGHAI)
        self.assertEqual(lock.size, (2560, 1664))
        self.assertEqual(home.size, (2560, 1664))
        self.assertEqual(home.target_lat, 0.0)
        self.assertEqual(home.target_lon, SHANGHAI[1])
        self.assertGreater(home.globe_radius_px, lock.globe_radius_px)
        for preset in (lock, MAC_LOCK):
            cx, cy = preset.center_px
            radius = preset.globe_radius_px
            self.assertGreaterEqual(cx - radius, 0)
            self.assertGreaterEqual(cy - radius, 0)
            self.assertLess(cx + radius, preset.size[0])
            self.assertLess(cy + radius, preset.size[1])
        self.assertGreater(MAC_HOME.center_px[1] - MAC_HOME.globe_radius_px, 180)
        self.assertGreater(MAC_HOME.center_px[0] - MAC_HOME.globe_radius_px, 180)
        self.assertGreater(MAC_HOME.center_px[1] + MAC_HOME.globe_radius_px, MAC_HOME.size[1])
        shanghai_y = MAC_HOME.center_px[1] - MAC_HOME.globe_radius_px * 0.518
        self.assertGreater(shanghai_y, 680)
        self.assertLess(shanghai_y, 760)


if __name__ == "__main__":
    unittest.main()
