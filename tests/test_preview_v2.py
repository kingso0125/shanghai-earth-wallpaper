from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from earthwall.preview_v2 import (
    V2_LOCK,
    V2_MAC_HOME,
    V2_MAC_LOCK,
    _apple_night_ir_cloud,
    _apple_night_midtone_grade,
    _edge_alpha,
    aces_tonemap,
    atmosphere_scattering,
    cloud_material,
    cloud_optical_properties,
    perspective_camera_grid,
    presets_for_mac_location,
    preview_output,
)


class PreviewV2Tests(unittest.TestCase):
    def test_preview_output_is_isolated_from_production(self) -> None:
        self.assertEqual(preview_output("lock"), Path("output/preview-v2/lock.jpg"))
        self.assertEqual(preview_output("home"), Path("output/preview-v2/home.jpg"))

    def test_mac_v2_contract_preserves_full_lock_and_breathing_home(self) -> None:
        lock, home = presets_for_mac_location(31.2304, 121.4737)
        self.assertEqual(lock.size, (2560, 1664))
        self.assertEqual(home.size, (2560, 1664))
        self.assertEqual(lock, V2_MAC_LOCK)
        self.assertEqual(home, V2_MAC_HOME)
        self.assertGreaterEqual(lock.center_px[1] - lock.globe_radius_px, 0)
        self.assertLess(lock.center_px[1] + lock.globe_radius_px, lock.size[1])
        self.assertGreater(home.center_px[1] - home.globe_radius_px, 450)
        self.assertGreater(home.center_px[1] + home.globe_radius_px, home.size[1])

    def test_perspective_center_points_at_requested_location(self) -> None:
        lat, lon, visible, view_cos, _vectors, _view = perspective_camera_grid(V2_LOCK)
        y, x = round(V2_LOCK.center_px[1]), round(V2_LOCK.center_px[0])
        self.assertTrue(visible[y, x])
        self.assertAlmostEqual(float(view_cos[y, x]), 1.0, places=4)
        self.assertAlmostEqual(float(np.rad2deg(lat[y, x])), V2_LOCK.target_lat, places=2)
        self.assertAlmostEqual(float(np.rad2deg(lon[y, x])), V2_LOCK.target_lon, places=2)

    def test_filmic_curve_preserves_highlight_order(self) -> None:
        values = np.array([0.25, 1.0, 4.0], dtype=np.float32)
        mapped = aces_tonemap(values)
        self.assertTrue(0.0 < mapped[0] < mapped[1] < mapped[2] <= 1.0)

    def test_dense_cloud_is_brighter_and_more_opaque_than_thin_cloud(self) -> None:
        alpha = np.array([[0.2, 0.9]], dtype=np.float32)
        props = cloud_optical_properties(alpha, np.ones_like(alpha))
        self.assertGreater(float(props.opacity[0, 1]), float(props.opacity[0, 0]))
        self.assertGreater(float(props.radiance[0, 1].mean()), float(props.radiance[0, 0].mean()))

    def test_day_cloud_material_is_bright_neutral_and_textured(self) -> None:
        alpha = np.full((32, 32), 0.88, dtype=np.float32)
        observed = np.tile(np.linspace(0.2, 0.85, 32, dtype=np.float32), (32, 1))
        material = cloud_material(alpha, observed, np.full_like(alpha, 0.75))
        left = material.radiance[:, 2].mean(axis=0)
        right = material.radiance[:, -3].mean(axis=0)
        self.assertGreater(float(right.mean()), float(left.mean()))
        self.assertGreater(float(right.mean()), 0.35)
        self.assertLess(float(right.max() - right.min()), 0.12)
        self.assertGreater(float(material.opacity[:, -3].mean()), 0.55)

    def test_night_cloud_material_stays_dark_but_occludes_lights(self) -> None:
        alpha = np.full((24, 24), 0.92, dtype=np.float32)
        observed = np.full_like(alpha, 0.72)
        material = cloud_material(alpha, observed, np.full_like(alpha, -0.5))
        self.assertLess(float(material.radiance.mean()), 0.03)
        self.assertGreater(float(material.opacity.mean()), 0.55)

    def test_apple_night_cloud_material_is_visible_but_not_day_white(self) -> None:
        alpha = np.full((24, 24), 0.92, dtype=np.float32)
        observed = np.tile(np.linspace(0.16, 0.82, 24, dtype=np.float32), (24, 1))
        material = cloud_material(
            alpha,
            observed,
            np.full_like(alpha, -0.5),
            apple_night=True,
        )
        self.assertGreater(float(material.radiance.mean()), 0.008)
        self.assertLess(float(material.radiance.mean()), 0.08)
        self.assertGreater(
            float(material.radiance[:, -2].mean()),
            float(material.radiance[:, 1].mean()),
        )
        self.assertGreater(float(material.opacity.mean()), 0.06)

    def test_apple_night_ir_decoder_preserves_continuous_cloud_depth(self) -> None:
        infrared = np.zeros((32, 32, 4), dtype=np.float32)
        infrared[..., :3] = 0.45
        infrared[:, 8:16, :3] = np.array([0.20, 0.48, 0.72], dtype=np.float32)
        infrared[:, 16:24, :3] = np.array([0.05, 0.85, 0.20], dtype=np.float32)
        alpha, texture = _apple_night_ir_cloud(infrared)
        self.assertEqual(alpha.shape, infrared.shape[:2])
        self.assertGreater(float(alpha[:, 12].mean()), float(alpha[:, 2].mean()))
        self.assertGreater(float(alpha[:, 20].mean()), float(alpha[:, 2].mean()))
        self.assertGreater(float(texture.std()), 0.05)

    def test_apple_night_grade_opens_midtones_without_lifting_black(self) -> None:
        earth = np.array([[[0.0, 0.0, 0.0], [0.008, 0.010, 0.012]]], dtype=np.float32)
        graded = _apple_night_midtone_grade(earth, np.ones((1, 2), dtype=np.float32))
        self.assertTrue(np.array_equal(graded[0, 0], earth[0, 0]))
        self.assertGreater(float(graded[0, 1].mean()), float(earth[0, 1].mean()))

    def test_atmosphere_is_thin_and_brighter_on_sunward_limb(self) -> None:
        yy, xx = np.mgrid[-1:1:65j, -1:1:65j]
        rho2 = xx * xx + yy * yy
        visible = rho2 <= 1.0
        z = np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))
        vectors = np.stack((xx, -yy, z), axis=-1).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        vectors /= np.maximum(norms, 1e-6)
        sun = np.array([1.0, 0.0, 0.2], dtype=np.float32)
        sun /= np.linalg.norm(sun)
        rim, halo = atmosphere_scattering(visible, z, vectors, sun, (65, 65))
        right = visible & (xx > 0.75)
        left = visible & (xx < -0.75)
        self.assertGreater(float(rim[right].mean()), float(rim[left].mean()))
        self.assertGreater(float(halo[~visible].max()), 0.0)

    def test_globe_edge_has_a_visible_soft_transition(self) -> None:
        alpha = _edge_alpha(V2_LOCK)
        row = alpha[round(V2_LOCK.center_px[1])]
        transition = (row > 0.05) & (row < 0.95)
        self.assertGreaterEqual(int(transition.sum()), 10)
        self.assertEqual(float(row[round(V2_LOCK.center_px[0])]), 1.0)


if __name__ == "__main__":
    unittest.main()
