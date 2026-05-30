from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleanshot.core.brain import brain_path, teach_image
from cleanshot.core.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Teach CleanShot one screenshot example.")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("category", help="Correct category, e.g. Code, Photos, Browser")
    parser.add_argument("subcategory", nargs="?", default="", help="Optional subcategory, e.g. JavaScript")
    args = parser.parse_args()

    ok = teach_image(Path(args.image), args.category, args.subcategory, load_config())
    if not ok:
        print("Could not teach from that image.")
        return 1
    label = args.category + (("/" + args.subcategory) if args.subcategory else "")
    print(f"Learned {args.image} as {label}")
    print(f"Brain: {brain_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
