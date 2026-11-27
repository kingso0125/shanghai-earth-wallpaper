import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from earthwall.thermal import coldness, sample_coldness
from earthwall.geometry import sample_equirectangular
from earthwall.preview_v2 import cloud_material, _apple_night_ir_cloud, _cloud_layer, _cloud_volume_shading, _surface_radiance, V2_LOCK, V2_HOME
from earthwall.render import _city_light_signal


class ReferenceMaterialTests(unittest.TestCase):
    def test_noon_ocean_glint_retains_highlight_headroom(self):
        normal = np.zeros((12, 12, 3), dtype=np.float32)
        normal[..., 2] = 1
        albedo = np.full_like(normal, .16)
        ocean = np.ones((12, 12), dtype=np.float32)
        surface, _ = _surface_radiance(
            albedo, ocean, normal, normal, normal,
            np.array([0, 0, 1], dtype=np.float32), apple_night=True,
        )
        self.assertTrue(np.isfinite(surface).all())
        self.assertLess(float(surface.max()), .4)

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

    def test_plateau_reduces_ambiguous_ir_but_preserves_cold_storm_tops(self):
        field = np.full((32, 32), .38, dtype=np.float32)
        infrared = np.zeros((*field.shape, 4), dtype=np.float32)
        plateau = np.full_like(field, 5000)
        for coldness in (.38, .80):
            field.fill(coldness)
            low, _ = _apple_night_ir_cloud(infrared, thermal=field)
            high, _ = _apple_night_ir_cloud(infrared, thermal=field, height_meters=plateau)
            if coldness < .4:
                self.assertLess(high.mean(), low.mean() * .5)
            else:
                np.testing.assert_array_equal(high, low)

    def test_mountain_slope_cannot_receive_direct_sun_in_deep_night(self):
        shape = (12, 12, 3)
        ground = np.zeros(shape, dtype=np.float32)
        ground[..., 2] = 1
        slope = np.zeros_like(ground)
        slope[..., 0] = .8
        slope[..., 2] = .6
        albedo = np.full_like(ground, .4)
        land = np.zeros(shape[:2], dtype=np.float32)
        sun = np.array([.866, 0, -.5], dtype=np.float32)
        flat, _ = _surface_radiance(albedo, land, ground, ground, ground, sun, apple_night=True)
        tilted, _ = _surface_radiance(albedo, land, slope, ground, ground, sun, apple_night=True)
        np.testing.assert_allclose(tilted, flat)

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

    def test_home_transmits_surface_through_medium_clouds_and_retains_dense_cores(self):
        # A common browse alpha can represent either a veil or a bright core.
        def material(coverage, texture, home):
            alpha = np.full((24, 24), coverage, dtype=np.float32)
            return cloud_material(alpha, np.full_like(alpha, texture), np.ones_like(alpha),
                                  apple_night=True, oblique_home=home)

        medium = material(.55, .50, True)
        previous = material(.55, .50, False)
        self.assertGreater(1.0 - medium.opacity.mean(), .60)
        self.assertLess(medium.radiance.mean(), previous.radiance.mean() * .80)
        dim_bank = material(.92, .50, True)
        bright_core = material(.92, .88, True)
        self.assertGreater(bright_core.opacity.mean(), .95)
        self.assertGreater(bright_core.opacity.mean() - dim_bank.opacity.mean(), .1)
        self.assertEqual(material(0., .75, True).opacity.max(), 0.)

    def test_home_limb_softens_clouds_and_keeps_dense_banks_over_lights(self):
        shape = (48, 80)
        surface = np.full((*shape, 3), .012, dtype=np.float32)
        alpha = np.zeros(shape, dtype=np.float32)
        alpha[:, 16:64] = .95
        texture = np.ones_like(alpha) * .85
        sun = np.array([1, 0, 0], dtype=np.float32)
        for solar_cos in (-.8, .8):
            def render(view_cos, emission=None):
                return _cloud_layer(surface, alpha, texture, np.full_like(alpha, solar_cos),
                                    np.full_like(alpha, view_cos), sun, V2_HOME, apple_night=True,
                                    city_emission=emission)

            far, near = render(.23), render(.85)
            np.testing.assert_array_equal(far[:, :16], surface[:, :16])
            self.assertLess(far[24, 40].mean(), near[24, 40].mean() * .75)
            transmitted = render(.23, np.ones_like(surface)) - far
            self.assertLess(transmitted[24, 40].max(), .08)
            self.assertGreater(transmitted[24, 0].min(), .99)

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
