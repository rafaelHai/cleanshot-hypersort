#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleanshot.core.brain import brain_path, load_brain, memory_stats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect CleanShot learning memory.")
    parser.add_argument("--limit", type=int, default=20, help="Recent examples to show")
    args = parser.parse_args()

    data = load_brain()
    stats = memory_stats()
    print("CleanShot brain")
    print(f"Path:   {brain_path()}")
    print(f"Version:{data.get('version')}")
    print(
        "Memory: "
        f"{stats['total']} total, {stats['manual_teach']} manual, "
        f"{stats['learn_from_folder']} folder, {stats['auto_observed']} auto"
    )

    labels = Counter(
        f"{item.get('category', 'Other')}/{item.get('subcategory', '')}".rstrip("/")
        for item in data.get("visual_memory", [])
    )
    if labels:
        print("\nTop labels")
        for label, count in labels.most_common(12):
            print(f"- {label}: {count}")

    print("\nRecent examples")
    for item in data.get("visual_memory", [])[: max(0, args.limit)]:
        label = f"{item.get('category', 'Other')}/{item.get('subcategory', '')}".rstrip("/")
        print(
            f"- {label} · {item.get('source', 'auto_observed')} · "
            f"confirmed={item.get('times_confirmed', 1)} overridden={item.get('times_overridden', 0)} · "
            f"{item.get('source_path', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
