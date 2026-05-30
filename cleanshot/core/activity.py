from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import activity_path

MAX_ACTIVITY_ROWS = 500


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_activity(
    action: str,
    message: str,
    source: str = "",
    destination: str = "",
    *,
    category: str = "",
    subcategory: str = "",
    confidence: Optional[float] = None,
    mode: str = "",
    app_name: str = "",
    reason: str = "",
    classifier_source: str = "",
    matched_rule: str = "",
    debug_signals: Optional[Dict[str, Any]] = None,
    ocr_enabled: Optional[bool] = None,
    learning_memory_used: Optional[bool] = None,
    **extra: Any,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "time": now_iso(),
        "action": action,
        "message": message,
        "source": source,
        "destination": destination,
        "category": category,
        "subcategory": subcategory,
        "confidence": confidence,
        "mode": mode,
        "app_name": app_name,
        "reason": reason,
        "classifier_source": classifier_source,
        "matched_rule": matched_rule,
        "debug_signals": debug_signals or {},
        "ocr_enabled": ocr_enabled,
        "learning_memory_used": learning_memory_used,
    }
    record.update(extra)

    path = activity_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    records = read_activity(limit=MAX_ACTIVITY_ROWS - 1)
    records.insert(0, record)

    with path.open("w", encoding="utf-8") as handle:
        for item in records[:MAX_ACTIVITY_ROWS]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return record


def read_activity(limit: int = 25) -> List[Dict[str, Any]]:
    path = activity_path()
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    records.append(json.loads(text))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    records.sort(key=lambda row: row.get("time", ""), reverse=True)
    return records[:limit]


def clear_activity() -> None:
    path = activity_path()
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
