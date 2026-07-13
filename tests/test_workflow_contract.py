import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WorkflowContractTests(unittest.TestCase):
    def test_watchdog_can_recover_before_phone_automation(self):
        workflow = (ROOT / ".github/workflows/hourly-watchdog.yml").read_text()
        self.assertIn("- scheduler", workflow)
        self.assertIn('cron: "35 * * * *"', workflow)
        self.assertIn("age <= 50 * 60", workflow)

        iphone_guide = (ROOT / "docs/iphone-shortcut.md").read_text()
        self.assertIn("从 00:45 到 23:45", iphone_guide)


if __name__ == "__main__":
    unittest.main()
