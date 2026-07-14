import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from earthwall.config import SHANGHAI
from earthwall.location import Location
from earthwall.publisher import Publisher


class PublisherTests(unittest.TestCase):
    def test_published_release_is_traversable_by_nginx(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = Publisher(root / "output", root / "cache", root / "render.lock")

            def fake_render(_observation, staging, **_kwargs):
                (staging / "lock.jpg").write_bytes(b"lock")
                (staging / "home.jpg").write_bytes(b"home")
                return {"rendered_utc": "2026-07-14T14:17:00Z"}

            with patch("earthwall.publisher.acquire", return_value=object()), patch(
                "earthwall.publisher.render_pair", side_effect=fake_render
            ), patch(
                "earthwall.publisher.audit", return_value={"passed": True, "failures": []}
            ):
                publisher.publish(Location(*SHANGHAI, "Shanghai"))

            mode = stat.S_IMODE((root / "output" / "current").resolve().stat().st_mode)
            self.assertEqual(mode, 0o755)


if __name__ == "__main__":
    unittest.main()
