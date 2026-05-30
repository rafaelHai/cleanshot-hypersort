from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .activity import append_activity
from .classifier import ClassificationResult, classify_app_folder, classify_screenshot, classify_smart_screenshot
from .config import expand_path, hashes_path
from .brain import remember_session_decision, remember_visual_decision


@dataclass
class OrganizeResult:
    status: str
    source: Path
    destination: Optional[Path] = None
    message: str = ""
    mode: str = "day"
    category: str = ""
    confidence: float = 0.0
    app_name: str = "Unknown App"


class ScreenshotOrganizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.file_types = set(config.get("file_types", []))
        self.keywords = [str(item).lower() for item in config.get("screenshot_keywords", [])]
        self.output_folder = expand_path(config.get("output_folder", "~/Pictures/CleanShot"))
        self.watch_folder = expand_path(config.get("watch_folder", "~/Desktop"))
        self._hashes = self._load_hashes()

    def update_config(self, config: Dict[str, Any]) -> None:
        self.__init__(config)

    def is_candidate(self, path: Path) -> bool:
        if not path.exists() or not path.is_file():
            return False

        if path.suffix.lower() not in self.file_types:
            return False

        if not self.keywords:
            return True

        name = path.name.lower()
        return any(keyword in name for keyword in self.keywords)

    def wait_until_ready(self, path: Path, timeout: float = 8.0) -> bool:
        start = time.time()
        last_size = -1

        while time.time() - start < timeout:
            if not path.exists():
                return False
            try:
                current_size = path.stat().st_size
                with path.open("rb"):
                    pass
            except OSError:
                time.sleep(0.25)
                continue

            if current_size > 0 and current_size == last_size:
                return True

            last_size = current_size
            time.sleep(0.35)

        return path.exists()

    def organize_file(self, path: Path) -> OrganizeResult:
        path = Path(path)
        if not self.is_candidate(path):
            return OrganizeResult(status="ignored", source=path, message="Not a screenshot candidate.")

        if not self.wait_until_ready(path):
            return OrganizeResult(status="failed", source=path, message="File was not ready.")

        try:
            file_hash = self._file_hash(path)
        except OSError as exc:
            return OrganizeResult(status="failed", source=path, message=str(exc))

        if self.config.get("detect_duplicates", True) and file_hash in self._hashes:
            duplicate_folder = self.output_folder / "Duplicates"
            duplicate_folder.mkdir(parents=True, exist_ok=True)
            duplicate_destination = self._unique_path(duplicate_folder / path.name)
            try:
                self._move_or_copy(path, duplicate_destination)
            except OSError as exc:
                return OrganizeResult(status="failed", source=path, message=str(exc))

            message = f"Duplicate saved to {duplicate_destination.name}"
            append_activity("duplicate", message, str(path), str(duplicate_destination))
            return OrganizeResult(
                status="duplicate",
                source=path,
                destination=duplicate_destination,
                message=message,
                mode=str(self.config.get("organization_mode", "day")),
                category="Duplicates",
            )

        destination, classification = self._destination_for(path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination = self._unique_path(destination)
            self._move_or_copy(path, destination)
            self._hashes[file_hash] = str(destination)
            self._save_hashes()
            if mode := str(self.config.get("organization_mode", "smart")).lower():
                if mode == "smart" and classification.category not in {"Other", "_Review", "Review", "Duplicates"}:
                    remember_visual_decision(
                        destination,
                        category=classification.category,
                        subcategory=getattr(classification, "subcategory", ""),
                        app_name=classification.app_name,
                        destination=str(destination),
                        source="auto_observed",
                        confidence=classification.confidence,
                    )
                    remember_session_decision(
                        category=classification.category,
                        subcategory=getattr(classification, "subcategory", ""),
                        app_name=classification.app_name,
                        confidence=classification.confidence,
                    )
        except OSError as exc:
            append_activity("error", f"Failed to organize {path.name}: {exc}", str(path), str(destination))
            return OrganizeResult(status="failed", source=path, destination=destination, message=str(exc))

        action_word = "Copied" if self.config.get("copy_instead_of_move", False) else "Moved"
        mode = str(self.config.get("organization_mode", "day"))
        target_label = classification.category if classification else destination.parent.name
        if classification and getattr(classification, "subcategory", ""):
            target_label = f"{classification.category}/{classification.subcategory}"
        confidence_text = ""
        if classification and mode == "smart" and classification.confidence > 0:
            source_label = self._source_label(classification)
            confidence_text = f" ({source_label} {round(classification.confidence * 100)}%)" if source_label else f" ({round(classification.confidence * 100)}%)"
        message = f"{action_word} {path.name} → {target_label}{confidence_text}"
        append_activity(
            "organized",
            message,
            str(path),
            str(destination),
            category=classification.category if classification else target_label,
            subcategory=getattr(classification, "subcategory", "") if classification else "",
            confidence=classification.confidence if classification else None,
            mode=mode,
            app_name=classification.app_name if classification else "Unknown App",
            reason=classification.reason if classification else "",
            classifier_source=classification.source if classification else "",
            matched_rule=getattr(classification, "matched_rule", "") if classification else "",
            debug_signals=getattr(classification, "debug_signals", {}) if classification else {},
            ocr_enabled=bool(self.config.get("enable_ocr", False)),
            learning_memory_used=bool(classification and classification.source in {"manual_teach", "folder_learning", "auto_observed"}),
        )
        return OrganizeResult(
            status="organized",
            source=path,
            destination=destination,
            message=message,
            mode=mode,
            category=target_label,
            confidence=classification.confidence if classification else 0.0,
            app_name=classification.app_name if classification else "Unknown App",
        )

    def organize_existing(self) -> List[OrganizeResult]:
        watch_folder = self.watch_folder
        recursive = bool(self.config.get("recursive_watch", False))
        iterator: Iterable[Path]

        if not watch_folder.exists():
            return [OrganizeResult(status="failed", source=watch_folder, message="Watch folder does not exist.")]

        iterator = watch_folder.rglob("*") if recursive else watch_folder.iterdir()
        results: List[OrganizeResult] = []

        for path in list(iterator):
            if path.is_file() and self.is_candidate(path):
                results.append(self.organize_file(path))

        organized_count = len([item for item in results if item.status in {"organized", "duplicate"}])
        append_activity("scan", f"Organized {organized_count} existing screenshot(s)")
        return results

    def _destination_for(self, source: Path) -> Tuple[Path, ClassificationResult]:
        now = datetime.now()
        extension = source.suffix.lower()
        mode = str(self.config.get("organization_mode", "day")).lower()
        classification = self._classify_for_mode(source, mode, now)

        category_name = self._safe_name(classification.category)
        subcategory_name = self._safe_name(getattr(classification, "subcategory", "")) if getattr(classification, "subcategory", "") else ""
        smart_folder = category_name
        if subcategory_name:
            smart_folder = f"{category_name}/{subcategory_name}"

        tokens = {
            "date": now.strftime("%Y-%m-%d"),
            "year": now.strftime("%Y"),
            "month": now.strftime("%m"),
            "day": now.strftime("%d"),
            "mode": mode,
            "mode_folder": smart_folder if mode == "smart" else category_name,
            "smart_folder": smart_folder,
            "category": category_name,
            "subcategory": subcategory_name,
            "app": self._safe_name(classification.app_name),
            "confidence": str(round(classification.confidence * 100)),
            "extension": extension.replace(".", "").upper(),
            "original_stem": self._safe_name(source.stem),
            "timestamp": now.strftime("%Y%m%d-%H%M%S"),
        }

        folder_template = self.config.get("folder_template", "{mode_folder}") or "{mode_folder}"
        folder_name = self._format_template(folder_template, tokens)
        target_folder = self.output_folder / folder_name

        if self.config.get("auto_rename", True):
            filename_template = self.config.get("filename_template", "screenshot-{timestamp}{extension}")
            filename = self._format_template(filename_template, {**tokens, "extension": extension})
            if not Path(filename).suffix:
                filename = f"{filename}{extension}"
        else:
            filename = source.name

        return target_folder / filename, classification

    def _classify_for_mode(self, source: Path, mode: str, now: datetime) -> ClassificationResult:
        if mode == "day":
            return ClassificationResult(
                category=now.strftime("%Y-%m-%d"),
                confidence=1.0,
                reason="Day mode",
                app_name="Date",
                source="day",
            )

        if mode == "app":
            result = classify_app_folder(self.config)
            unknown_folder = str(self.config.get("unknown_app_folder", "Unknown App")).strip() or "Unknown App"
            if result.category == "Unknown App":
                return ClassificationResult(
                    category=unknown_folder,
                    confidence=result.confidence,
                    reason=result.reason,
                    app_name=result.app_name,
                    source=result.source,
                )
            return result

        if mode == "content":
            # Legacy configs from older versions are treated as Smart mode.
            mode = "smart"

        if mode == "smart":
            result = classify_smart_screenshot(source, self.config)
            threshold = float(self.config.get("smart_confidence_threshold", 0.72))
            if result.confidence < threshold and result.category not in {str(self.config.get("review_folder", "_Review")), str(self.config.get("other_folder", "Other"))}:
                folder = str(self.config.get("review_folder", "_Review" if self.config.get("review_low_confidence", True) else "Other")).strip() or "_Review"
                return ClassificationResult(
                    category=folder,
                    confidence=result.confidence,
                    reason=f"Low smart confidence: {result.reason}",
                    app_name=result.app_name,
                    source=result.source,
                    signals=result.signals,
                )
            return result

        return ClassificationResult(category="Other", confidence=0.0, reason="Unknown mode", app_name="Unknown App")

    def _format_template(self, template: str, tokens: Dict[str, str]) -> str:
        output = str(template)
        for key, value in tokens.items():
            output = output.replace("{" + key + "}", value)
        return self._safe_relative_name(output)

    def _move_or_copy(self, source: Path, destination: Path) -> None:
        if self.config.get("copy_instead_of_move", False):
            shutil.copy2(str(source), str(destination))
        else:
            shutil.move(str(source), str(destination))

    def _unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 2
        while True:
            candidate = parent / f"{stem}-{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _file_hash(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_hashes(self) -> Dict[str, str]:
        path = hashes_path()
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return {str(key): str(value) for key, value in data.items()}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_hashes(self) -> None:
        path = hashes_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self._hashes, handle, indent=2)

    def _safe_name(self, value: str) -> str:
        cleaned = "".join(char if char.isalnum() or char in "-_. " else "-" for char in str(value))
        cleaned = " ".join(cleaned.strip().split())
        return cleaned or "screenshot"

    def _safe_relative_name(self, value: str) -> str:
        text = str(value).strip().replace("\\", "/")
        pieces = [self._safe_name(piece) for piece in text.split("/") if piece.strip()]
        return "/".join(pieces) if pieces else "Screenshots"

    def _source_label(self, classification: ClassificationResult) -> str:
        labels = {
            "manual_teach": "manual teach match",
            "app_detection": "app detection",
            "content_detection": "content match",
            "folder_learning": "folder learned match",
            "ocr": "OCR match",
            "visioncore": "visual match",
            "fallback_review": "review",
            "auto_observed": "learned match",
        }
        return labels.get(str(classification.source), "")
