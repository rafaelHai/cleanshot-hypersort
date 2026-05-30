from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from cleanshot.core import brain, classifier
from cleanshot.core.config import normalize_config


class SmartPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_app_data_dir = brain.app_data_dir
        brain.app_data_dir = lambda: self.root
        self.old_app = classifier.get_frontmost_app_name
        self.old_title = classifier.get_frontmost_window_title
        classifier.get_frontmost_app_name = lambda: "Unknown App"
        classifier.get_frontmost_window_title = lambda: ""
        brain.reset_learning_memory()

    def tearDown(self) -> None:
        brain.app_data_dir = self.old_app_data_dir
        classifier.get_frontmost_app_name = self.old_app
        classifier.get_frontmost_window_title = self.old_title
        self.tmp.cleanup()

    def test_manual_teach_beats_old_code_memory(self) -> None:
        image = self.root / "cleanshot-dashboard.png"
        _draw_dashboard(image)
        for index in range(3):
            brain.remember_visual_decision(
                image,
                "Code",
                "JavaScript",
                source="learn_from_folder",
                destination=f"old-code-{index}.png",
                confidence=0.86,
            )
        brain.teach_example(image, "Apps", "CleanShot", normalize_config({}))

        result = classifier.classify_smart_screenshot(image, normalize_config({"enable_ocr": False}))

        self.assertEqual(result.category, "Apps")
        self.assertEqual(result.subcategory, "CleanShot")
        self.assertEqual(result.source, "manual_teach")
        self.assertEqual(result.matched_rule, "manual_teach_override")

    def test_javascript_filename_and_layout_goes_to_code(self) -> None:
        image = self.root / "javascript-function-const-return.png"
        _draw_code(image)

        result = classifier.classify_smart_screenshot(image, normalize_config({"enable_ocr": False}))

        self.assertEqual(result.category, "Code")
        self.assertIn(result.subcategory, {"JavaScript", "Projects"})

    def test_cleanshot_app_without_code_goes_to_apps(self) -> None:
        classifier.get_frontmost_app_name = lambda: "CleanShot HyperSort"
        classifier.get_frontmost_window_title = lambda: "Dashboard"
        image = self.root / "dashboard.png"
        _draw_dashboard(image)

        result = classifier.classify_smart_screenshot(image, normalize_config({"enable_ocr": False}))

        self.assertEqual(result.category, "Apps")
        self.assertEqual(result.subcategory, "CleanShot")
        self.assertEqual(result.source, "app_detection")

    def test_vs_code_with_visible_code_goes_to_code(self) -> None:
        classifier.get_frontmost_app_name = lambda: "Visual Studio Code"
        classifier.get_frontmost_window_title = lambda: "app.tsx"
        image = self.root / "app.tsx.png"
        _draw_code(image)

        result = classifier.classify_smart_screenshot(image, normalize_config({"enable_ocr": False}))

        self.assertEqual(result.category, "Code")
        self.assertIn(result.subcategory, {"JavaScript", "TypeScript", "Projects"})


def _draw_dashboard(path: Path) -> None:
    image = Image.new("RGB", (900, 560), "#f7f8fb")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 180, 560), fill="#172033")
    draw.rectangle((210, 80, 420, 190), fill="#ffffff", outline="#d9dee8")
    draw.rectangle((450, 80, 660, 190), fill="#ffffff", outline="#d9dee8")
    draw.rectangle((210, 220, 820, 500), fill="#ffffff", outline="#d9dee8")
    draw.text((230, 105), "CleanShot HyperSort", fill="#111827")
    draw.text((230, 245), "Activity Dashboard", fill="#111827")
    image.save(path)


def _draw_code(path: Path) -> None:
    image = Image.new("RGB", (900, 560), "#10141f")
    draw = ImageDraw.Draw(image)
    for index in range(18):
        y = 28 + index * 27
        draw.text((32, y), str(index + 1).rjust(2), fill="#586174")
        draw.text((78, y), "const value = items.map((item) => item.id);", fill="#9cdcfe")
        if index % 3 == 0:
            draw.text((360, y), "return console.log(value);", fill="#dcdcaa")
    image.save(path)


if __name__ == "__main__":
    unittest.main()
