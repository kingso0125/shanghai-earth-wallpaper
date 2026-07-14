import unittest
from datetime import UTC, datetime

import numpy as np

from earthwall.config import (
    HOME,
    LOCK,
    LOCK_LATITUDE_OFFSET,
    SHANGHAI,
    presets_for_location,
)
from earthwall.geometry import camera_grid, sample_himawari_plate
from earthwall.lighting import daylight, sun_vector
from earthwall.location import Location, LocationStore, haversine_km
from earthwall.render import (
    _apple_natural_grade,
    _blend_city_lights,
    _fallback_cloud_appearance,
    _feather_coverage,
    _grade_geocolor,
    _night_cloud_alpha,
    _sharpen_cloud_texture,
)
from earthwall.sources import CIRA_SOURCES, latest_common_time


class CoreTests(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(SHANGHAI, (31.2304, 121.4737))
        self.assertEqual(LOCK.size, (1320, 2868))
        self.assertEqual(HOME.size, (1320, 2868))
        self.assertGreater(HOME.globe_radius_px, LOCK.globe_radius_px)

    def test_lock_places_shanghai_slightly_above_globe_center(self):
        preset = presets_for_location(*SHANGHAI)[0]
        lat, lon, visible, _, _ = camera_grid(preset)
        x, y = map(int, preset.center_px)
        self.assertAlmostEqual(
            float(np.rad2deg(lat[y, x])),
            SHANGHAI[0] + LOCK_LATITUDE_OFFSET,
            places=2,
        )
        self.assertAlmostEqual(float(np.rad2deg(lon[y, x])), SHANGHAI[1], places=2)
        shanghai_y = preset.center_px[1] - preset.globe_radius_px * np.sin(
            np.deg2rad(-LOCK_LATITUDE_OFFSET)
        )
        self.assertGreater(shanghai_y, 1390)
        self.assertLess(shanghai_y, 1420)
        self.assertFalse(bool(visible[0, 0]))

    def test_home_keeps_shanghai_above_the_bottom_controls(self):
        home = presets_for_location(*SHANGHAI)[1]
        self.assertEqual(home.target_lat, 0.0)
        shanghai_y = home.center_px[1] - home.globe_radius_px * np.sin(
            np.deg2rad(SHANGHAI[0])
        )
        self.assertGreater(shanghai_y, 1500)
        self.assertLess(shanghai_y, 1800)

    def test_camera_can_follow_guangzhou_meridian(self):
        guangzhou = presets_for_location(23.1291, 113.2644)[0]
        lat, lon, _, _, _ = camera_grid(guangzhou)
        x, y = map(int, guangzhou.center_px)
        self.assertAlmostEqual(
            float(np.rad2deg(lat[y, x])), 23.1291 + LOCK_LATITUDE_OFFSET, places=2
        )
        self.assertAlmostEqual(float(np.rad2deg(lon[y, x])), 113.2644, places=2)

    def test_location_store_requires_more_than_80_km(self):
        import tempfile
        from pathlib import Path

        shanghai = Location(31.2304, 121.4737, "Shanghai")
        nearby = Location(31.2100, 121.1000, "Nearby")
        guangzhou = Location(23.1291, 113.2644, "Guangzhou")
        self.assertLess(haversine_km(shanghai, nearby), 80.0)
        self.assertGreater(haversine_km(shanghai, guangzhou), 80.0)

        with tempfile.TemporaryDirectory() as directory:
            store = LocationStore(Path(directory) / "location.json", threshold_km=80.0)
            current, _, changed = store.update(nearby)
            self.assertFalse(changed)
            self.assertEqual(current.name, "Shanghai")
            current, _, changed = store.update(guangzhou)
            self.assertTrue(changed)
            self.assertEqual(current.name, "Guangzhou")

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

    def test_himawari_is_preferred_over_gk2a(self):
        self.assertEqual(CIRA_SOURCES[0][0], "himawari")
        self.assertEqual(CIRA_SOURCES[1][0], "gk2a")

    def test_geocolor_night_grade_removes_infrared_purple(self):
        source = np.array([[[0.60, 0.10, 0.80]]], dtype=np.float32)
        raw_grade = np.power(np.clip(source * 1.035 + 0.010, 0.0, 1.0), 0.95)
        night = _grade_geocolor(source, np.zeros((1, 1), dtype=np.float32))
        day = _grade_geocolor(source, np.ones((1, 1), dtype=np.float32))

        self.assertLess(float(np.ptp(night[0, 0])), float(np.ptp(raw_grade[0, 0])) * 0.5)
        self.assertLess(float(night[0, 0, 0]), float(night[0, 0, 2]))
        self.assertLess(float(np.ptp(day[0, 0])), float(np.ptp(raw_grade[0, 0])) * 0.5)

        natural_land = np.array([[[0.20, 0.50, 0.15]]], dtype=np.float32)
        natural_grade = np.power(np.clip(natural_land * 1.035 + 0.010, 0.0, 1.0), 0.95)
        luminance = np.sum(
            natural_grade
            * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
            axis=-1,
            keepdims=True,
        )
        natural_grade = luminance + (natural_grade - luminance) * 0.94
        natural_grade = np.clip(
            natural_grade * np.array([1.035, 1.015, 0.975], dtype=np.float32),
            0.0,
            1.0,
        )
        np.testing.assert_allclose(
            _grade_geocolor(natural_land, np.ones((1, 1), dtype=np.float32)),
            natural_grade,
            atol=1e-6,
        )

    def test_geocolor_day_grade_rolls_off_only_neutral_cloud_highlights(self):
        cloud = np.array([[[0.92, 0.92, 0.92]]], dtype=np.float32)
        land = np.array([[[0.20, 0.50, 0.15]]], dtype=np.float32)
        raw_cloud = np.power(np.clip(cloud * 1.035 + 0.010, 0.0, 1.0), 0.95)
        raw_luminance = np.sum(
            raw_cloud * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
            axis=-1,
            keepdims=True,
        )
        raw_cloud = raw_luminance + (raw_cloud - raw_luminance) * 0.94
        raw_cloud = np.clip(
            raw_cloud * np.array([1.035, 1.015, 0.975], dtype=np.float32),
            0.0,
            1.0,
        )

        cloud_grade = _grade_geocolor(cloud, np.ones((1, 1), dtype=np.float32))
        land_grade = _grade_geocolor(land, np.ones((1, 1), dtype=np.float32))

        self.assertLess(float(cloud_grade.mean()), float(raw_cloud.mean()) * 0.90)
        natural_grade = np.power(np.clip(land * 1.035 + 0.010, 0.0, 1.0), 0.95)
        luminance = np.sum(
            natural_grade
            * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
            axis=-1,
            keepdims=True,
        )
        natural_grade = luminance + (natural_grade - luminance) * 0.94
        natural_grade = np.clip(
            natural_grade * np.array([1.035, 1.015, 0.975], dtype=np.float32),
            0.0,
            1.0,
        )
        np.testing.assert_allclose(land_grade, natural_grade, atol=1e-6)

    def test_display_grade_warms_without_destroying_observed_structure(self):
        source = np.array([[[0.42, 0.30, 0.18]]], dtype=np.float32)
        graded = _apple_natural_grade(
            source,
            np.ones((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
        )
        self.assertGreater(float(graded.mean()), float(source.mean()))
        self.assertGreater(float(graded[0, 0, 0]), float(graded[0, 0, 2]) * 1.5)
        self.assertGreater(float(np.ptp(graded[0, 0])), 0.20)

    def test_display_grade_keeps_ocean_blue(self):
        ocean = np.array([[[0.05, 0.12, 0.36]]], dtype=np.float32)
        graded = _apple_natural_grade(
            ocean,
            np.ones((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
        )
        self.assertGreater(float(graded[0, 0, 2]), float(graded[0, 0, 1]) * 1.6)
        self.assertGreater(float(graded[0, 0, 1]), float(graded[0, 0, 0]))
        self.assertGreater(float(graded.mean()), float(ocean.mean()) * 1.25)

    def test_display_grade_maps_vegetation_to_teal_and_desert_to_gold(self):
        source = np.array(
            [[[0.12, 0.42, 0.17], [0.58, 0.43, 0.22]]], dtype=np.float32
        )
        graded = _apple_natural_grade(
            source,
            np.ones((1, 2), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
        )
        vegetation, desert = graded[0]
        self.assertGreater(float(vegetation[1]), float(vegetation[0]) * 1.25)
        self.assertGreater(float(vegetation[2]), float(source[0, 0, 2]))
        self.assertGreater(float(desert[0]), float(desert[2]) * 1.8)
        self.assertGreater(float(desert[1]), float(desert[2]) * 1.4)

    def test_cloud_sharpening_increases_local_detail_only_inside_cloud_mask(self):
        earth = np.full((9, 9, 3), 0.35, dtype=np.float32)
        earth[4, 4] = 0.75
        mask = np.zeros((9, 9), dtype=np.float32)
        mask[2:7, 2:7] = 1.0
        sharpened = _sharpen_cloud_texture(earth, mask)
        self.assertGreater(float(sharpened[4, 4].mean()), float(earth[4, 4].mean()))
        np.testing.assert_allclose(sharpened[0, 0], earth[0, 0], atol=1e-6)

    def test_display_grade_separates_day_and_night(self):
        source = np.full((1, 1, 3), 0.35, dtype=np.float32)
        day = _apple_natural_grade(
            source, np.ones((1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)
        )
        night = _apple_natural_grade(
            source, np.zeros((1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)
        )
        self.assertGreater(float(day.mean()), float(night.mean()) * 1.35)

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

    def test_fallback_cloud_coverage_edge_is_feathered(self):
        alpha = np.zeros((101, 101), dtype=np.float32)
        alpha[:, 50:] = 1.0
        feathered = _feather_coverage(alpha, radius=8.0)
        self.assertEqual(float(feathered[50, 45]), 0.0)
        self.assertEqual(float(feathered[50, 50]), 0.0)
        self.assertGreater(float(feathered[50, 55]), 0.0)
        self.assertLess(float(feathered[50, 55]), 1.0)

    def test_fallback_cloud_style_preserves_optical_depth(self):
        visible = np.array(
            [[[[0.16, 0.16, 0.16], [0.68, 0.68, 0.68]]]], dtype=np.float32
        ).reshape(1, 2, 3)
        alpha = np.array([[0.42, 0.92]], dtype=np.float32)
        color, mix = _fallback_cloud_appearance(
            visible, alpha, np.ones((1, 2), dtype=np.float32)
        )

        self.assertGreater(float(color[0, 1].mean()), float(color[0, 0].mean()) + 0.2)
        self.assertGreater(float(mix[0, 1]), float(mix[0, 0]) + 0.4)
        self.assertGreater(float(color[0, 1, 0]), float(color[0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
