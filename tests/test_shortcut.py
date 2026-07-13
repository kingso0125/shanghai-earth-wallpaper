import plistlib
import unittest
from pathlib import Path


class ShortcutTests(unittest.TestCase):
    def test_lock_and_home_targets_are_explicit(self):
        source = Path(__file__).parents[1] / "shortcuts" / "更新上海实时地球.plist"
        workflow = plistlib.loads(source.read_bytes())
        actions = workflow["WFWorkflowActions"]

        self.assertEqual(
            [action["WFWorkflowActionIdentifier"] for action in actions],
            [
                "is.workflow.actions.url",
                "is.workflow.actions.downloadurl",
                "is.workflow.actions.wallpaper.set",
                "is.workflow.actions.url",
                "is.workflow.actions.downloadurl",
                "is.workflow.actions.wallpaper.set",
            ],
        )
        self.assertTrue(actions[0]["WFWorkflowActionParameters"]["WFURLActionURL"].endswith("/lock.jpg"))
        self.assertTrue(actions[3]["WFWorkflowActionParameters"]["WFURLActionURL"].endswith("/home.jpg"))

        lock = actions[2]["WFWorkflowActionParameters"]
        home = actions[5]["WFWorkflowActionParameters"]
        self.assertEqual(lock["WFWallpaperLocation"], ["Lock Screen"])
        self.assertEqual(home["WFWallpaperLocation"], ["Home Screen"])
        for parameters in (lock, home):
            self.assertFalse(parameters["WFWallpaperShowPreview"])
            self.assertFalse(parameters["WFWallpaperSmartCrop"])
            self.assertFalse(parameters["WFWallpaperLegibilityBlur"])

    def test_install_page_explains_hourly_phone_automation(self):
        page = (Path(__file__).parents[1] / "web" / "index.html").read_text()
        self.assertIn("更新上海实时地球.shortcut?v=20260713-home-fix", page)
        self.assertIn("从 00:45 到 23:45", page)
        self.assertIn("云端每小时 :33", page)


if __name__ == "__main__":
    unittest.main()
