import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from earthwall.thermal import coldness, sample_coldness
from earthwall.geometry import sample_equirectangular
from earthwall.preview_v2 import cloud_material, _apple_night_ir_cloud, _cloud_layer, _cloud_volume_shading, V2_LOCK
from earthwall.render import _city_light_signal


class ReferenceMaterialTests(unittest.TestCase):
    def test_sparse_thermal_decode_matches_full_plate_including_seam_and_poles(self):
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, (24, 48, 3), dtype=np.uint8)
        lat = np.linspace(-np.pi / 2, np.pi / 2, 30, dtype=np.float32)[:, None]
        lon = np.linspace(-np.pi, np.pi, 40, dtype=np.float32)[None, :]
        lat, lon = np.broadcast_arrays(lat, lon)
        expected = sample_equirectangular(coldness(rgb)[..., None], lat, lon)[..., 0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ir.png"
            Image.fromarray(rgb).save(path)
            np.testing.assert_array_equal(sample_coldness(path, lat, lon), expected)

    def test_ir_palette_temperature_order_is_not_chroma_order(self):
        # Official palette: warm grey, -19 C cyan, -40 C green, -60 C red.
        rgb = np.array([[[80, 80, 80], [0, 255, 255], [0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
        field = coldness(rgb)
        self.assertTrue(np.all(np.diff(field[0]) > 0))

    def test_continuous_thermal_field_retains_internal_cloud_structure(self):
        field = np.tile(np.linspace(.3, .75, 48, dtype=np.float32), (48, 1))
        alpha, texture = _apple_night_ir_cloud(np.zeros((48, 48, 4)), thermal=field)
        self.assertTrue(np.all(np.diff(alpha[24, 4:-4]) > 0))
        self.assertGreater(float(texture.std()), .15)

    def test_cloud_opacity_does_not_disappear_at_sunset(self):
        alpha = np.full((16, 16), .85, dtype=np.float32)
        structure = np.full_like(alpha, .7)
        day = cloud_material(alpha, structure, np.ones_like(alpha), apple_night=True)
        night = cloud_material(alpha, structure, -np.ones_like(alpha), apple_night=True)
        np.testing.assert_allclose(day.opacity, night.opacity)
        self.assertGreater(float(night.opacity.mean()), .75)

    def test_city_white_core_remains_brighter_than_yellow_halo(self):
        lights = np.ones((12, 12, 4), dtype=np.float32)
        lights[:, :6, :3] = [0.5, 0.4, 0.15]
        strength = _city_light_signal(lights)
        self.assertGreater(float(strength[:, 9].mean()), float(strength[:, 2].mean()) * 2)

    def test_dense_cloud_is_opaque_but_thin_wisps_stay_translucent(self):
        alpha = np.full((24, 24), .92, dtype=np.float32)
        dense = cloud_material(alpha, alpha, -np.ones_like(alpha), apple_night=True)
        thin = cloud_material(alpha * .1, alpha * .1, -np.ones_like(alpha), apple_night=True)
        self.assertGreater(float(dense.opacity.mean()), .96)
        self.assertLess(float(thin.opacity.mean()), .10)

    def test_volume_does_not_expand_cloud_footprint_or_change_clear_surface(self):
        alpha = np.zeros((32, 32), dtype=np.float32)
        alpha[8:24, 8:24] = .9
        surface = np.full((32, 32, 3), .1, dtype=np.float32)
        for solar in (-.5, .5):
            actual = _cloud_layer(
                surface, alpha, alpha, np.full_like(alpha, solar), np.ones_like(alpha),
                np.array([1, 0, 0]), V2_LOCK, apple_night=True,
            )
            np.testing.assert_array_equal(actual[alpha == 0], surface[alpha == 0])

    def test_night_volume_has_relief_without_fictitious_sun_direction(self):
        y, x = np.mgrid[-1:1:64j, -1:1:64j]
        observed = np.exp(-(x*x + y*y) * 8).astype(np.float32)
        night = np.zeros_like(observed)
        view = np.ones_like(observed)
        left = _cloud_volume_shading(observed, night, view, 1, 0, 1)
        right = _cloud_volume_shading(observed, night, view, -1, 0, 1)
        np.testing.assert_array_equal(left, right)
        self.assertGreater(float(left[32, 32]), 1.10)
        self.assertLess(float(left.min()), .99)

    def test_cloud_top_does_not_copy_city_emission(self):
        shape = (16, 16)
        alpha = np.full(shape, .9, dtype=np.float32)
        texture = np.full(shape, .8, dtype=np.float32)
        night, view = -np.ones(shape, dtype=np.float32), np.ones(shape, dtype=np.float32)
        dark = np.zeros((*shape, 3), dtype=np.float32)
        light = dark.copy()
        light[8, 8] = 1
        args = (alpha, texture, night, view, np.array([1, 0, 0]), V2_LOCK)
        delta = _cloud_layer(light, *args, apple_night=True) - _cloud_layer(dark, *args, apple_night=True)
        self.assertLess(float(delta[8, 8].max()), .2)
        self.assertEqual(float(delta[7, 8].max()), 0)
