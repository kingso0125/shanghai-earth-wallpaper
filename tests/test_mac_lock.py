import plistlib
import tempfile
import unittest
from pathlib import Path

from earthwall.mac_lock import configure


class MacLockTests(unittest.TestCase):
    def test_configures_every_idle_entry_with_fixed_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "Index.plist"
            image = root / "mac-lock.jpg"
            image.touch()
            original = {
                "SystemDefault": {"Idle": {"Content": {}}},
                "Displays": {"display": {"Idle": {"Content": {}}}},
            }
            index.write_bytes(plistlib.dumps(original, fmt=plistlib.FMT_BINARY))

            self.assertEqual(configure(index, image), 2)
            result = plistlib.loads(index.read_bytes())
            for idle in (
                result["SystemDefault"]["Idle"],
                result["Displays"]["display"]["Idle"],
            ):
                choice = idle["Content"]["Choices"][0]
                self.assertEqual(choice["Provider"], "com.apple.wallpaper.choice.image")
                configuration = plistlib.loads(choice["Configuration"])
                self.assertEqual(configuration["url"]["relative"], image.resolve().as_uri())


if __name__ == "__main__":
    unittest.main()
