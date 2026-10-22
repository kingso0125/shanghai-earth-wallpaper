import bz2
import struct
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from earthwall.himawari import decode_segment, sample_temperature, scan_coordinates


PROJECTION = np.array([140.7, 20466275, 20466275, 2750.5, 2750.5,
                       42164, 6378.137, 6356.7523])
STAMP = datetime(2026, 10, 22, 14, 20, tzinfo=UTC)


def synthetic_segment():
    lengths = [282, 50, 127, 139, 147, 259, 47, 81, 85, 47, 259]
    blocks = []
    for number, length in enumerate(lengths, 1):
        block = bytearray(length)
        struct.pack_into('<BI' if number == 10 else '<BH', block, 0, number, length)
        blocks.append(block)
    basic, dimensions, projection, _, calibration, _, segment = blocks[:7]
    basic[6:22] = b'Himawari-9'.ljust(16, b' ')
    struct.pack_into('<H', basic, 44, 1420)
    mjd = (STAMP - datetime(1858, 11, 17, tzinfo=UTC)).total_seconds() / 86400
    struct.pack_into('<d', basic, 46, mjd)
    struct.pack_into('<II', basic, 70, sum(lengths), 5500 * 550 * 2)
    struct.pack_into('<HHHB', dimensions, 3, 16, 5500, 550, 0)
    struct.pack_into('<dIIffddd', projection, 3, PROJECTION[0], int(PROJECTION[1]),
                     int(PROJECTION[2]), *PROJECTION[3:])
    struct.pack_into('<HdHHH', calibration, 3, 13, 10.4074, 12, 65535, 65534)
    struct.pack_into('<11d', calibration, 19, -.00375, 15.2, 0, 1, 0, 0, 1, 0,
                     299792458, 6.62606957e-34, 1.3806488e-23)
    struct.pack_into('<BBH', segment, 3, 10, 5, 2201)
    pixels = np.full((550, 5500), 1800, dtype='<u2')
    pixels[0, :2] = [65534, 65535]
    return bz2.compress(b''.join(blocks) + pixels.tobytes())


class HimawariTests(unittest.TestCase):
    def test_calibration_matches_planck_and_rejects_fill_counts(self):
        pixels, lookup, projection = decode_segment(synthetic_segment(), STAMP, 5)
        wave, light, planck, boltzmann = 10.4074e-6, 299792458, 6.62606957e-34, 1.3806488e-23
        radiance = 15.2 - .00375 * 1800
        expected = planck * light / (boltzmann * wave) / np.log1p(
            2 * planck * light**2 / (wave**5 * radiance * 1e6))
        self.assertAlmostEqual(float(lookup[1800]), expected, places=4)
        self.assertTrue(np.isnan(lookup[pixels[0, :2]]).all())
        np.testing.assert_allclose(projection, PROJECTION)

    def test_wrong_timestamp_or_segment_rejected(self):
        payload = synthetic_segment()
        for timestamp, sequence in ((STAMP.replace(hour=13), 5), (STAMP, 4)):
            with self.subTest(sequence=sequence, timestamp=timestamp):
                with self.assertRaises(ValueError):
                    decode_segment(payload, timestamp, sequence)
        with self.assertRaises(ValueError):
            decode_segment(bz2.compress(b'truncated'), STAMP, 5)

    def test_projection_center_orientation_and_hidden_hemisphere(self):
        lat = np.deg2rad(np.array([0, 10, -10, 0, 0, 0], dtype=np.float32))
        lon = np.deg2rad(np.array([140.7, 140.7, 140.7, 150.7, 130.7, -39.3], dtype=np.float32))
        x, y, visible = scan_coordinates(lat, lon, PROJECTION)
        self.assertAlmostEqual(float(x[0]), 2749.5, places=2)
        self.assertAlmostEqual(float(y[0]), 2749.5, places=2)
        self.assertLess(y[1], y[0])
        self.assertGreater(y[2], y[0])
        self.assertGreater(x[3], x[0])
        self.assertLess(x[4], x[0])
        self.assertEqual(visible.tolist(), [True, True, True, True, True, False])

    def test_sampling_preserves_extreme_cold_and_masks_missing_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.npz'
            # Synthetic tiny scan; SSP lies at its half-pixel center.
            projection = PROJECTION.copy()
            projection[3:5] = 2.5
            lookup = np.array([190, 200, np.nan], dtype=np.float32)
            counts = np.zeros((4, 4), dtype=np.uint16)
            counts[:, 2:] = 1
            np.savez_compressed(path, counts=counts, temperature=lookup, projection=projection)
            lat = np.zeros((1, 2), dtype=np.float32)
            lon = np.deg2rad(np.array([[140.7, -39.3]], dtype=np.float32))
            cold, coverage = sample_temperature(path, lat, lon)
            self.assertAlmostEqual(float(cold[0, 0]), (40 - (195 - 273.15)) / 140, places=4)
            self.assertEqual(coverage.tolist(), [[1, 0]])
            counts[:] = 2
            np.savez_compressed(path, counts=counts, temperature=lookup, projection=projection)
            cold, coverage = sample_temperature(path, lat, lon)
            self.assertFalse(coverage.any())
            self.assertFalse(cold.any())
