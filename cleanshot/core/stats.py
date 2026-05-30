from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from .activity import read_activity
from .config import expand_path


@dataclass
class FolderSummary:
    path: Path
    count: int
    size: int


def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def collect_stats(config: Dict[str, Any]) -> Dict[str, Any]:
    output_folder = expand_path(config.get("output_folder", "~/Pictures/CleanShot"))
    file_types = {str(item).lower() for item in config.get("file_types", [])}

    files: List[Path] = []
    if output_folder.exists():
        files = [path for path in output_folder.rglob("*") if path.is_file() and path.suffix.lower() in file_types]

    total_size = 0
    type_counter: Counter[str] = Counter()
    for path in files:
        try:
            total_size += path.stat().st_size
        except OSError:
            pass
        type_counter[path.suffix.lower().replace(".", "").upper() or "OTHER"] += 1

    activity = read_activity(limit=50)
    today = date.today().isoformat()
    organized_today = len([
        row for row in activity
        if str(row.get("time", "")).startswith(today)
        and row.get("action") in {"organized", "duplicate"}
    ])

    folders = collect_folders(output_folder, file_types)

    return {
        "output_folder": str(output_folder),
        "total_screenshots": len(files),
        "total_size": total_size,
        "total_size_label": human_size(total_size),
        "organized_today": organized_today,
        "types": dict(type_counter.most_common()),
        "folders": folders,
        "activity": activity[:12],
    }


def collect_folders(output_folder: Path, file_types: set[str]) -> List[FolderSummary]:
    if not output_folder.exists():
        return []

    summaries: List[FolderSummary] = []
    for child in sorted(output_folder.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue

        count = 0
        size = 0
        for file_path in child.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in file_types:
                continue
            count += 1
            try:
                size += file_path.stat().st_size
            except OSError:
                pass

        if count:
            summaries.append(FolderSummary(path=child, count=count, size=size))

    summaries.sort(key=lambda row: (row.count, row.size), reverse=True)
    return summaries[:8]
