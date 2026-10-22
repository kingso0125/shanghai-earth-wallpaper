import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from earthwall.elevation import ELEVATION, terrain_normals
from earthwall.preview_v2 import _blur_scalar, _cloud_layer, V2_LOCK
from earthwall.geometry import sample_equirectangular


class ReliefLayerTests(unittest.TestCase):
    def test_non_power_of_two_height_map_wraps_antimeridian_rounding(self):
        height = np.zeros((4, 5400, 1), dtype=np.uint16)
        height[:, 0] = 100
        latitude = np.zeros((1, 1), dtype=np.float32)
        longitude = np.array([[-3.141593]], dtype=np.float32)
        sampled = sample_equirectangular(height, latitude, longitude)
        self.assertAlmostEqual(float(sampled[0, 0, 0]), 100, places=2)

    def test_cloud_filter_preserves_sub_8bit_gradients(self):
        ramp = np.tile(np.linspace(.4501, .4510, 48, dtype=np.float32), (32, 1))
        for radius in (.65, 4):
            with self.subTest(radius=radius):
                actual = _blur_scalar(ramp, radius)
                self.assertGreater(len(np.unique(actual[16])), 35)
                self.assertTrue(np.all(np.diff(actual[16]) > 0))

    def test_float_filter_keeps_constant_fields_constant(self):
        for radius in (.55, 2.5, 11):
            np.testing.assert_allclose(_blur_scalar(np.full((24, 32), .4507), radius), .4507, atol=1e-6)

    def test_elevation_asset_is_meter_data_with_correct_orientation(self):
        with Image.open(ELEVATION) as image:
            data = np.asarray(image)
        self.assertEqual(data.shape, (2700, 5400))
        row = round((90 - 28) / 180 * 2699)
        column = int((86.9 + 180) / 360 * 5400)
        self.assertGreater(int(data[row, column]), 3000)
        self.assertEqual(int(data[1350, 600]), 0)

    def test_meter_slope_tilts_land_normal_but_keeps_water_smooth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'height.png'
            heights = np.tile(np.arange(128, dtype=np.uint16) * 60, (64, 1))
            Image.fromarray(heights).save(path)
            lat = np.zeros((12, 12), dtype=np.float32)
            lon = np.zeros_like(lat)
            vectors = np.zeros((12, 12, 3), dtype=np.float32)
            vectors[..., 0] = 1
            normal, _ = terrain_normals(path, lat, lon, vectors, np.ones_like(lat))
            self.assertLess(float(normal[..., 1].mean()), -.0001)
            np.testing.assert_allclose(np.linalg.norm(normal, axis=-1), 1, atol=1e-6)
            water, _ = terrain_normals(path, lat, lon, vectors, np.zeros_like(lat))
            np.testing.assert_array_equal(water, vectors)

    def test_city_emission_is_below_clouds_and_does_not_light_cloud_tops(self):
        shape = (32, 32)
        surface = np.full((*shape, 3), .01, dtype=np.float32)
        emission = np.zeros_like(surface)
        emission[16, 16] = [1, .7, .3]
        alpha = np.zeros(shape, dtype=np.float32)
        alpha[8:24, 8:24] = .92
        args = (alpha, alpha, -np.ones(shape), np.ones(shape), np.array([1, 0, 0]), V2_LOCK)
        dark = _cloud_layer(surface, *args, apple_night=True)
        lit = _cloud_layer(surface, *args, apple_night=True, city_emission=emission)
        self.assertLess(float((lit-dark)[16, 16].max()), .01)
        self.assertEqual(float((lit-dark)[15, 16].max()), 0)
        clear = _cloud_layer(surface, np.zeros(shape), *args[1:], apple_night=True, city_emission=emission)
        np.testing.assert_allclose(clear[16, 16], surface[16, 16] + emission[16, 16])
