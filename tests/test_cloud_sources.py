import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from earthwall.preview_sources import upgrade_v2_observation
from earthwall.sources import (
    Observation, _acquire_gibs_layers, _newest_cached_pair, _valid_cloud_image,
)


class CloudSourceTests(unittest.TestCase):
    def test_transparent_placeholder_rejected_but_black_night_accepted(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.png"
            Image.new("RGBA", (64, 32), (0, 0, 0, 0)).save(path)
            self.assertFalse(_valid_cloud_image(path, (64, 32)))
            Image.new("RGBA", (64, 32), (0, 0, 0, 255)).save(path)
            self.assertTrue(_valid_cloud_image(path, (64, 32)))

    def test_cached_pair_with_empty_ir_is_not_a_fallback(self):
        with TemporaryDirectory() as directory:
            cache = Path(directory)
            for channel, alpha in (("visible", 255), ("infrared", 0)):
                Image.new("RGBA", (4096, 2048), (0, 0, 0, alpha)).save(
                    cache / f"himawari-20260907T1150Z-{channel}.png"
                )
            self.assertIsNone(_newest_cached_pair(cache))

    def test_empty_latest_frame_uses_validated_wms_common_frame(self):
        latest = datetime(2026, 9, 7, 11, 50, tzinfo=UTC)
        available = datetime(2026, 9, 7, 11, 30, tzinfo=UTC)
        pair = (Path("visible.png"), Path("infrared.png"))
        with patch("earthwall.sources.latest_gibs_time", return_value=latest), patch(
            "earthwall.sources._newest_cached_pair", return_value=None
        ), patch("earthwall.sources._gibs_pair", side_effect=[ValueError("empty"), pair]) as fetch, patch(
            "earthwall.sources._request", return_value=b"capabilities"
        ), patch("earthwall.sources.latest_common_time", return_value=available):
            result = _acquire_gibs_layers(Path("cache"), Path("base"), Path("lights"), None)
        self.assertEqual(result.timestamp, available)
        self.assertEqual([call.args[1] for call in fetch.call_args_list], [latest, available])

    def test_empty_8k_keeps_valid_4k_at_same_observed_time(self):
        observation = Observation(
            datetime(2026, 9, 7, 11, 30, tzinfo=UTC), Path("visible.png"),
            Path("infrared.png"), None, Path("base.png"), Path("lights.png"), "fresh",
        )
        with patch("earthwall.preview_sources._download", side_effect=[
            Path("base8"), Path("lights8"), Path("terrain8"), Path("water8"),
            Path("visible8"), ValueError("empty 8K IR"),
        ]), patch("earthwall.preview_sources._valid_cloud_image", return_value=True):
            result = upgrade_v2_observation(Path("cache"), observation)
        self.assertEqual(result.visible, observation.visible)
        self.assertEqual(result.infrared, observation.infrared)
        self.assertEqual(result.timestamp, observation.timestamp)

    def test_empty_8k_and_invalid_4k_fail_closed(self):
        observation = Observation(
            datetime(2026, 9, 7, 11, 30, tzinfo=UTC), Path("visible.png"),
            Path("infrared.png"), None, Path("base.png"), Path("lights.png"), "fresh",
        )
        with patch("earthwall.preview_sources._download", side_effect=[
            Path("base8"), Path("lights8"), Path("terrain8"), Path("water8"),
            ValueError("empty 8K"),
        ]), patch("earthwall.preview_sources._valid_cloud_image", return_value=False):
            with self.assertRaises(ValueError):
                upgrade_v2_observation(Path("cache"), observation)
