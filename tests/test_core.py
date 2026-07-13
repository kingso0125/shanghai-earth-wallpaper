import unittest
from datetime import UTC, datetime

import numpy as np

from earthwall.config import HOME, LOCK, SHANGHAI
from earthwall.geometry import camera_grid, sample_himawari_plate
from earthwall.lighting import daylight, sun_vector
from earthwall.render import _blend_city_lights, _grade_geocolor, _night_cloud_alpha
from earthwall.sources import latest_common_time


class CoreTests(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(SHANGHAI, (31.2304, 121.4737))
        self.assertEqual(LOCK.size, (1320, 2868))
        self.assertEqual(HOME.size, (1320, 2868))
        self.assertGreater(HOME.globe_radius_px, LOCK.globe_radius_px)

    def test_camera_center_is_shanghai_meridian(self):
        lat, lon, visible, _, _ = camera_grid(LOCK)
        x, y = map(int, LOCK.center_px)
        self.assertAlmostEqual(float(np.rad2deg(lat[y, x])), 0.0, places=2)
        self.assertAlmostEqual(float(np.rad2deg(lon[y, x])), SHANGHAI[1], places=2)
        self.assertFalse(bool(visible[0, 0]))

    def test_daylight_is_bounded(self):
        _, _, _, _, vectors = camera_grid(LOCK)
        day = daylight(vectors, datetime(2026, 7, 13, 1, 20, tzinfo=UTC))
        self.assertGreaterEqual(float(day.min()), 0.0)
        self.assertLessEqual(float(day.max()), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(sun_vector(datetime.now(UTC)))), 1.0, places=5)

    def test_native_himawari_plate_keeps_center_pixel(self):
        image = np.zeros((101, 101, 4), dtype=np.float32)
        image[50, 50] = (0.2, 0.4, 0.8, 1.0)
        sampled, valid = sample_himawari_plate(image, LOCK)
        x, y = map(int, LOCK.center_px)
        np.testing.assert_allclose(sampled[y, x], image[50, 50], atol=1e-6)
        self.assertTrue(bool(valid[y, x]))

    def test_latest_common_time(self):
        xml = '''
        <Layer><Name>Himawari_AHI_Band3_Red_Visible_1km</Name><Dimension name="time" default="2026-07-13T01:30:00Z" /></Layer>
        <Layer><Name>Himawari_AHI_Band13_Clean_Infrared</Name><Dimension name="time" default="2026-07-13T01:20:00Z" /></Layer>
        '''
        self.assertEqual(
            latest_common_time(xml), datetime(2026, 7, 13, 1, 20, tzinfo=UTC)
        )

    def test_geocolor_night_grade_removes_infrared_purple(self):
        source = np.array([[[0.60, 0.10, 0.80]]], dtype=np.float32)
        raw_grade = np.power(np.clip(source * 1.08 + 0.012, 0.0, 1.0), 0.90)
        night = _grade_geocolor(source, np.zeros((1, 1), dtype=np.float32))
        day = _grade_geocolor(source, np.ones((1, 1), dtype=np.float32))

        self.assertLess(float(np.ptp(night[0, 0])), float(np.ptp(raw_grade[0, 0])) * 0.5)
        self.assertLess(float(night[0, 0, 0]), float(night[0, 0, 2]))
        self.assertLess(float(np.ptp(day[0, 0])), float(np.ptp(raw_grade[0, 0])) * 0.5)

        natural_land = np.array([[[0.20, 0.50, 0.15]]], dtype=np.float32)
        natural_grade = np.power(np.clip(natural_land * 1.08 + 0.012, 0.0, 1.0), 0.90)
        natural_grade = np.clip(
            natural_grade * np.array([1.035, 1.01, 0.975], dtype=np.float32),
            0.0,
            1.0,
        )
        np.testing.assert_allclose(
            _grade_geocolor(natural_land, np.ones((1, 1), dtype=np.float32)),
            natural_grade,
            atol=1e-6,
        )

    def test_city_lights_are_night_only_and_cloud_occluded(self):
        earth = np.zeros((9, 9, 3), dtype=np.float32)
        lights = np.zeros((9, 9, 4), dtype=np.float32)
        lights[4, 4] = (1.0, 1.0, 0.65, 1.0)
        clear = np.zeros((9, 9), dtype=np.float32)

        night = _blend_city_lights(earth, lights, np.zeros((9, 9), dtype=np.float32), clear)
        day = _blend_city_lights(earth, lights, np.ones((9, 9), dtype=np.float32), clear)
        cloudy = _blend_city_lights(
            earth,
            lights,
            np.zeros((9, 9), dtype=np.float32),
            np.ones((9, 9), dtype=np.float32),
        )

        self.assertGreater(float(night[4, 4].mean()), 0.25)
        np.testing.assert_allclose(day, earth, atol=1e-6)
        self.assertLess(float(cloudy[4, 4].mean()), float(night[4, 4].mean()) * 0.3)

    def test_warm_lights_are_not_classified_as_night_clouds(self):
        satellite = np.array([[[0.95, 0.62, 0.12]]], dtype=np.float32)
        cloud = _night_cloud_alpha(satellite, np.zeros((1, 1), dtype=np.float32))
        self.assertLess(float(cloud[0, 0]), 0.15)


if __name__ == "__main__":
    unittest.main()
