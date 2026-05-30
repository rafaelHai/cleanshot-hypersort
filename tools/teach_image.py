from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleanshot.core.brain import brain_path, mark_override, teach_image
from cleanshot.core.classifier import classify_smart_screenshot
from cleanshot.core.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Teach CleanShot one screenshot example.")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("category", help="Correct category, e.g. Code, Photos, Browser")
    parser.add_argument("subcategory", nargs="?", default="", help="Optional subcategory, e.g. JavaScript")
    args = parser.parse_args()

    config = load_config()
    image = Path(args.image).expanduser().resolve()
    previous = classify_smart_screenshot(image, config) if image.exists() else None
    ok = teach_image(image, args.category, args.subcategory, config)
    if not ok:
        print("Could not teach from that image.")
        return 1
    if previous and (previous.category != args.category or previous.subcategory != args.subcategory):
        mark_override(previous.category, previous.subcategory, args.category, args.subcategory)
    label = args.category + (("/" + args.subcategory) if args.subcategory else "")
    print(f"CleanShot learned this as {label}.")
    print(f"Brain: {brain_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
