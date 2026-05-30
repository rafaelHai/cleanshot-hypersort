from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

APP_NAME = "CleanShot"
APP_DIR_NAME = ".cleanshot"

ORGANIZATION_MODES = ["day", "app", "smart"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "watch_folder": "~/Desktop",
    "output_folder": "~/Pictures/CleanShot",
    "file_types": [".png", ".jpg", ".jpeg", ".webp"],
    "screenshot_keywords": ["screenshot", "screen shot", "clean shot"],
    "organization_mode": "smart",
    "use_date_folders": True,
    "auto_rename": True,
    "detect_duplicates": True,
    "recursive_watch": False,
    "copy_instead_of_move": False,
    "enable_ocr": False,
    "smart_confidence_threshold": 0.72,
    "smart_subfolders": True,
    "review_low_confidence": True,
    "enable_visual_memory": True,
    "enable_session_memory": True,
    "enable_neuro_learning": True,
    "neuro_similarity_threshold": 0.86,
    "visual_similarity_distance": 8,
    "session_context_minutes": 4,
    "unknown_app_folder": "Unknown App",
    "other_folder": "Other",
    "review_folder": "_Review",
    "folder_template": "{mode_folder}",
    "filename_template": "screenshot-{timestamp}{extension}",
    "theme": "light",
}


def app_data_dir() -> Path:
    path = Path.home() / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return app_data_dir() / "config.json"


def activity_path() -> Path:
    return app_data_dir() / "activity.jsonl"


def hashes_path() -> Path:
    return app_data_dir() / "hashes.json"


def expand_path(value: str | Path) -> Path:
    text = str(value).strip()
    if not text:
        return Path.home()
    return Path(text).expanduser().resolve()


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    incoming = config or {}
    had_mode_setting = "organization_mode" in incoming
    merged = dict(DEFAULT_CONFIG)
    merged.update(incoming)

    # v0.2 used {date} as the default folder template. In v0.3 the template
    # should follow the selected mode unless the user intentionally changes it.
    if not had_mode_setting and str(merged.get("folder_template", "")).strip() == "{date}":
        merged["folder_template"] = DEFAULT_CONFIG["folder_template"]

    file_types = []
    for extension in merged.get("file_types", []):
        ext = str(extension).strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in file_types:
            file_types.append(ext)
    merged["file_types"] = file_types or DEFAULT_CONFIG["file_types"]

    keywords = []
    for keyword in merged.get("screenshot_keywords", []):
        item = str(keyword).strip().lower()
        if item and item not in keywords:
            keywords.append(item)
    merged["screenshot_keywords"] = keywords or DEFAULT_CONFIG["screenshot_keywords"]

    mode = str(merged.get("organization_mode", "smart")).strip().lower()
    # v0.9 removed Specific/content mode from the UI. Existing installs are migrated to Smart mode.
    if mode == "content":
        mode = "smart"
    if mode not in ORGANIZATION_MODES:
        mode = "smart"
    merged["organization_mode"] = mode

    for key in [
        "use_date_folders",
        "auto_rename",
        "detect_duplicates",
        "recursive_watch",
        "copy_instead_of_move",
        "enable_ocr",
        "smart_subfolders",
        "review_low_confidence",
        "enable_visual_memory",
        "enable_session_memory",
        "enable_neuro_learning",
    ]:
        merged[key] = bool(merged.get(key, DEFAULT_CONFIG.get(key, False)))

    for threshold_key, default_value in [
        ("smart_confidence_threshold", 0.72),
        ("neuro_similarity_threshold", 0.86),
    ]:
        try:
            threshold = float(merged.get(threshold_key, default_value))
        except (TypeError, ValueError):
            threshold = default_value
        merged[threshold_key] = min(max(threshold, 0.0), 1.0)

    try:
        distance = int(merged.get("visual_similarity_distance", 8))
    except (TypeError, ValueError):
        distance = 8
    merged["visual_similarity_distance"] = min(max(distance, 0), 32)

    try:
        minutes = int(merged.get("session_context_minutes", 4))
    except (TypeError, ValueError):
        minutes = 4
    merged["session_context_minutes"] = min(max(minutes, 1), 60)

    for key in ["unknown_app_folder", "other_folder", "review_folder", "folder_template", "filename_template"]:
        value = str(merged.get(key, DEFAULT_CONFIG[key])).strip()
        merged[key] = value or DEFAULT_CONFIG[key]

    return merged


def load_config() -> Dict[str, Any]:
    path = config_path()
    if not path.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        data = {}

    normalized = normalize_config(data)
    save_config(normalized)
    return normalized


def save_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_config(config)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)
    return normalized
