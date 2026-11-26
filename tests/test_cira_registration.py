from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from earthwall.geometry import sample_cira_geocolor
from earthwall.preview_v2 import V2_HOME, V2_LOCK, _cloud_layer, _render_region, perspective_camera_grid
from earthwall.sources import Observation


class CiraRegistrationTests(unittest.TestCase):
    def test_native_disk_coordinates_and_hidden_hemisphere(self):
        yy, xx = np.indices((688, 688), dtype=np.float32)
        grid = np.stack((xx, yy), axis=-1)
        # SLIDER zoom-0 controls for Tokyo, Sydney, and an invisible point.
        lat = np.deg2rad(np.array([35.68, -33.87, 0]))
        lon = np.deg2rad(np.array([139.76, 151.21, -40]))
        for sub_lon, expected in [(140.69, [[338.41, 120.91], [402.08, 556.31]]),
                                  (128.0, [[407.25, 121.64], [468.69, 554.12]])]:
            with self.subTest(sub_lon=sub_lon):
                pixels, coverage = sample_cira_geocolor(grid, lat, lon, sub_lon)
                np.testing.assert_allclose(pixels[:2], expected, atol=.02)
                np.testing.assert_array_equal(coverage, [1, 1, 0])

    def test_cloud_over_tokyo_stays_over_tokyo_in_both_camera_compositions(self):
        instant = datetime(2026, 11, 26, 2, 30, tzinfo=UTC)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            earth, disk = root / 'earth.png', root / 'disk.png'
            Image.new('RGB', (360, 180)).save(earth)
            plate = Image.new('RGB', (688, 688))
            plate.paste('white', (334, 117, 343, 126))
            plate.save(disk)
            obs = Observation(instant, disk, disk, disk, earth, earth, 'fresh',
                              source='CIRA SLIDER / JMA Himawari-9', satellite_longitude=140.69)
            for original in (V2_LOCK, V2_HOME):
                with self.subTest(view=original.name):
                    preset = replace(original, size=(200, 200), center_px=(100, 100), globe_radius_px=97)
                    with patch('earthwall.preview_v2._cloud_layer', wraps=_cloud_layer) as layer:
                        _render_region(obs, preset, root/'photo.jpg', lighting_time=instant,
                                       background_asset=None, apple_night=True)
                    alpha = layer.call_args.args[1]
                    lat, lon, *_ = perspective_camera_grid(preset)
                    bright = alpha > .5
                    self.assertTrue(bright.any())
                    self.assertAlmostEqual(float(np.rad2deg(lat[bright]).mean()), 35.68, delta=1)
                    self.assertAlmostEqual(float(np.rad2deg(lon[bright]).mean()), 139.76, delta=1)
