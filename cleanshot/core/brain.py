from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import app_data_dir, expand_path


BRAIN_VERSION = 3
MAX_VISUAL_MEMORY = 5000
MAX_SESSION_MEMORY = 400
MEMORY_SOURCES = {"manual_teach", "learn_from_folder", "auto_observed"}
SOURCE_WEIGHTS = {
    "manual_teach": 1.35,
    "learn_from_folder": 0.78,
    "auto_observed": 0.42,
}


@dataclass(frozen=True)
class VisualMatch:
    category: str
    subcategory: str
    confidence: float
    distance: int = 0
    source_path: str = ""
    destination: str = ""
    reason: str = ""
    source: str = ""
    matched_rule: str = ""
    debug_signals: Dict[str, Any] | None = None


@dataclass(frozen=True)
class LearnedItem:
    category: str
    subcategory: str
    hash_hex: str
    app_name: str = "Unknown App"
    source_path: str = ""
    destination: str = ""
    created_at: str = ""


def brain_path() -> Path:
    return app_data_dir() / "brain.json"


def load_brain() -> Dict[str, Any]:
    path = brain_path()
    if not path.exists():
        return _empty_brain()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return _empty_brain()

    if not isinstance(data, dict):
        return _empty_brain()
    return _migrate_brain(data)


def save_brain(data: Dict[str, Any]) -> None:
    path = brain_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    visual = list(data.get("visual_memory", []))[:MAX_VISUAL_MEMORY]
    sessions = list(data.get("sessions", []))[:MAX_SESSION_MEMORY]
    safe_data = {
        "version": BRAIN_VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "visual_memory": [_normalize_memory_item(item) for item in visual if isinstance(item, dict)],
        "sessions": sessions,
        "manual_rules": data.get("manual_rules", {}),
        "teach_history": [_normalize_memory_item(item) for item in list(data.get("teach_history", []))[:800] if isinstance(item, dict)],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe_data, handle, indent=2, ensure_ascii=False)


def _empty_brain() -> Dict[str, Any]:
    return {"version": BRAIN_VERSION, "visual_memory": [], "sessions": [], "manual_rules": {}, "teach_history": []}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _migrate_brain(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize old and new brain.json shapes without dropping user learning."""
    visual_memory = data.get("visual_memory", [])
    if not isinstance(visual_memory, list):
        visual_memory = []
    teach_history = data.get("teach_history", [])
    if not isinstance(teach_history, list):
        teach_history = []
    sessions = data.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []

    normalized = {
        "version": BRAIN_VERSION,
        "updated_at": data.get("updated_at", ""),
        "visual_memory": [_normalize_memory_item(item) for item in visual_memory if isinstance(item, dict)],
        "sessions": sessions[:MAX_SESSION_MEMORY],
        "manual_rules": data.get("manual_rules", {}) if isinstance(data.get("manual_rules", {}), dict) else {},
        "teach_history": [_normalize_memory_item(item, default_source="manual_teach") for item in teach_history if isinstance(item, dict)],
    }
    return normalized


def _normalize_memory_item(item: Dict[str, Any], default_source: str = "auto_observed") -> Dict[str, Any]:
    source = str(item.get("source") or "").strip()
    if source not in MEMORY_SOURCES:
        # Old brain.json only had taught=True. Treat it as manual so existing
        # user corrections remain authoritative after migration.
        source = "manual_teach" if item.get("taught") else default_source
    if source not in MEMORY_SOURCES:
        source = "auto_observed"

    created_at = str(item.get("timestamp") or item.get("created_at") or item.get("updated_at") or _now_iso())
    hash_hex = str(item.get("perceptual_hash") or item.get("hash") or "")
    vector = item.get("feature_vector", item.get("vector", []))
    if not isinstance(vector, list):
        vector = []

    try:
        confidence = float(item.get("confidence", 1.0 if source == "manual_teach" else 0.74))
    except (TypeError, ValueError):
        confidence = 1.0 if source == "manual_teach" else 0.74

    try:
        times_confirmed = int(item.get("times_confirmed", 1))
    except (TypeError, ValueError):
        times_confirmed = 1
    try:
        times_overridden = int(item.get("times_overridden", 0))
    except (TypeError, ValueError):
        times_overridden = 0

    normalized = dict(item)
    normalized.update(
        {
            "category": str(item.get("category", "Other") or "Other"),
            "subcategory": str(item.get("subcategory", "") or ""),
            "source": source,
            "timestamp": created_at,
            "created_at": created_at,
            "feature_vector": [float(value) for value in vector if isinstance(value, (int, float))],
            "vector": [float(value) for value in vector if isinstance(value, (int, float))],
            "perceptual_hash": hash_hex,
            "hash": hash_hex,
            "ocr_fingerprint": str(item.get("ocr_fingerprint", "") or ""),
            "app_name": str(item.get("app_name", "Unknown App") or "Unknown App"),
            "window_title": str(item.get("window_title", "") or ""),
            "confidence": max(0.0, min(1.0, confidence)),
            "times_confirmed": max(0, times_confirmed),
            "times_overridden": max(0, times_overridden),
            "taught": source == "manual_teach",
        }
    )
    return normalized


# ---------------------------------------------------------------------------
# ScreenshotDNA hash memory
# ---------------------------------------------------------------------------

def perceptual_hash(path: Path, hash_size: int = 8) -> str:
    """Small dependency-free dHash for near-duplicate visual matching."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for visual fingerprinting") from exc

    with Image.open(path) as image:
        image = image.convert("L").resize((hash_size + 1, hash_size))
        pixels = list(image.getdata())

    bits: List[str] = []
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[offset + col]
            right = pixels[offset + col + 1]
            bits.append("1" if left > right else "0")

    value = int("".join(bits), 2)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    try:
        value = int(hash_a, 16) ^ int(hash_b, 16)
    except ValueError:
        return 999
    return value.bit_count()


def find_visual_match(path: Path, max_distance: int = 8) -> Optional[VisualMatch]:
    try:
        hash_hex = perceptual_hash(path)
    except Exception:
        return None

    best: Optional[Dict[str, Any]] = None
    best_distance = 999
    for item in load_brain().get("visual_memory", []):
        item_hash = str(item.get("hash", ""))
        if not item_hash:
            continue
        distance = hamming_distance(hash_hex, item_hash)
        if distance < best_distance:
            best = item
            best_distance = distance

    if best is None or best_distance > max_distance:
        return None

    confidence = max(0.72, 0.97 - (best_distance / max(max_distance, 1)) * 0.20)
    return VisualMatch(
        category=str(best.get("category", "Other")) or "Other",
        subcategory=str(best.get("subcategory", "")) or "",
        confidence=confidence,
        distance=best_distance,
        source_path=str(best.get("source_path", "")),
        destination=str(best.get("destination", "")),
        reason=f"ScreenshotDNA hash distance {best_distance}",
    )


# ---------------------------------------------------------------------------
# CleanShot-created learning engine: NeuroVector
# ---------------------------------------------------------------------------

def extract_neuro_vector(path: Path) -> List[float]:
    """Create a local visual embedding from the screenshot.

    This is CleanShot's own lightweight learning vector. It is not a pretrained
    cloud model. It combines thumbnail histograms, layout projections, edge maps,
    aspect ratio, brightness, dark-mode clues, and color structure. It lets the
    app learn from folders without uploading anything.
    """
    try:
        from PIL import Image, ImageFilter, ImageStat
    except ImportError:
        return []

    try:
        with Image.open(path) as source:
            width, height = source.size
            rgb = source.convert("RGB")
            small = rgb.resize((96, 96))
            gray = small.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
    except Exception:
        return []

    pixels = list(small.getdata())
    gray_pixels = list(gray.getdata())
    edge_pixels = list(edges.getdata())
    if not pixels or not gray_pixels:
        return []

    vector: List[float] = []

    # Global shape/color statistics.
    stat_rgb = ImageStat.Stat(small)
    stat_gray = ImageStat.Stat(gray)
    means = [value / 255.0 for value in stat_rgb.mean]
    stds = [value / 128.0 for value in stat_rgb.stddev]
    mean_luma = stat_gray.mean[0] / 255.0
    std_luma = stat_gray.stddev[0] / 128.0
    aspect = min(3.0, width / max(height, 1)) / 3.0
    edge_density = sum(1 for value in edge_pixels if value > 35) / len(edge_pixels)
    dark_ratio = sum(1 for value in gray_pixels if value < 55) / len(gray_pixels)
    light_ratio = sum(1 for value in gray_pixels if value > 205) / len(gray_pixels)
    sat_ratio = 0
    for r, g, b in pixels:
        mx = max(r, g, b)
        mn = min(r, g, b)
        if mx > 0 and (mx - mn) / mx > 0.28:
            sat_ratio += 1
    sat_ratio = sat_ratio / len(pixels)
    vector.extend([aspect, mean_luma, std_luma, edge_density, dark_ratio, light_ratio, sat_ratio])
    vector.extend(means)
    vector.extend(stds)

    # Luma histogram, 16 bins.
    hist = [0] * 16
    for value in gray_pixels:
        hist[min(15, value // 16)] += 1
    total = max(len(gray_pixels), 1)
    vector.extend([count / total for count in hist])

    # Saturation-ish histogram, 8 bins.
    sat_hist = [0] * 8
    for r, g, b in pixels:
        mx = max(r, g, b)
        mn = min(r, g, b)
        sat = 0 if mx == 0 else (mx - mn) / mx
        sat_hist[min(7, int(sat * 8))] += 1
    vector.extend([count / total for count in sat_hist])

    # Layout projections: rows and columns with edge activity.
    w, h = gray.size
    for bands, axis in [(12, "row"), (12, "col")]:
        for band in range(bands):
            if axis == "row":
                y1 = int(h * band / bands)
                y2 = max(y1 + 1, int(h * (band + 1) / bands))
                values = [edge_pixels[y * w + x] for y in range(y1, y2) for x in range(w)]
            else:
                x1 = int(w * band / bands)
                x2 = max(x1 + 1, int(w * (band + 1) / bands))
                values = [edge_pixels[y * w + x] for y in range(h) for x in range(x1, x2)]
            vector.append(sum(1 for value in values if value > 35) / max(len(values), 1))

    return _normalize_vector(vector)


def _normalize_vector(vector: List[float]) -> List[float]:
    if not vector:
        return []
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [round(value / norm, 6) for value in vector]


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(vector_a, vector_b))))


def _source_weight(source: str, timestamp: str = "", times_confirmed: int = 1, times_overridden: int = 0) -> float:
    weight = SOURCE_WEIGHTS.get(source, SOURCE_WEIGHTS["auto_observed"])

    # Manual corrections are strongest when recent. Folder and auto examples age
    # more quickly so old bad patterns do not permanently poison the classifier.
    try:
        age_days = max(0.0, (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds() / 86400)
    except (TypeError, ValueError):
        age_days = 365.0

    if source == "manual_teach":
        if age_days <= 7:
            weight *= 1.18
        elif age_days <= 30:
            weight *= 1.08
    elif source == "learn_from_folder":
        weight *= max(0.72, 1.0 - min(age_days, 365.0) / 900.0)
    else:
        weight *= max(0.55, 1.0 - min(age_days, 365.0) / 520.0)

    weight *= 1.0 + min(max(times_confirmed - 1, 0), 8) * 0.035
    weight *= max(0.12, 1.0 - min(max(times_overridden, 0), 8) * 0.16)
    return weight


def _hash_similarity(path_hash: str, item_hash: str, max_distance: int = 18) -> Tuple[float, int]:
    if not path_hash or not item_hash:
        return 0.0, 999
    distance = hamming_distance(path_hash, item_hash)
    if distance > max_distance:
        return 0.0, distance
    return max(0.0, 1.0 - (distance / max(max_distance, 1)) * 0.28), distance


def find_memory_match(
    path: Path,
    *,
    threshold: float = 0.84,
    min_examples: int = 1,
    sources: Optional[Iterable[str]] = None,
    matched_rule: str = "learned_memory",
) -> Optional[VisualMatch]:
    vector = extract_neuro_vector(path)
    try:
        hash_hex = perceptual_hash(path)
    except Exception:
        hash_hex = ""
    if not vector and not hash_hex:
        return None

    allowed_sources = set(sources or MEMORY_SOURCES)

    candidates: List[Tuple[float, float, int, Dict[str, Any]]] = []
    for item in load_brain().get("visual_memory", []):
        source = str(item.get("source") or "auto_observed")
        if source not in allowed_sources:
            continue
        item_vector = item.get("feature_vector", item.get("vector", []))
        vector_similarity = 0.0
        if vector and isinstance(item_vector, list):
            vector_similarity = cosine_similarity(vector, [float(value) for value in item_vector])
        hash_similarity, hash_distance = _hash_similarity(hash_hex, str(item.get("perceptual_hash") or item.get("hash") or ""))
        similarity = max(vector_similarity, hash_similarity)
        if similarity > 0:
            candidates.append((similarity, vector_similarity, hash_distance, item))

    if not candidates:
        return None

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    top = candidates[:9]

    # Weighted vote by category/subcategory. This is stronger than 1-nearest-neighbor.
    votes: Dict[Tuple[str, str], float] = {}
    counts: Dict[Tuple[str, str], int] = {}
    best_by_label: Dict[Tuple[str, str], Tuple[float, int, Dict[str, Any]]] = {}
    best_item = top[0][3]
    best_similarity = top[0][0]
    best_distance = top[0][2]

    for similarity, _vector_similarity, hash_distance, item in top:
        key = (str(item.get("category", "Other")) or "Other", str(item.get("subcategory", "")) or "")
        source = str(item.get("source") or "auto_observed")
        try:
            times_confirmed = int(item.get("times_confirmed", 1))
        except (TypeError, ValueError):
            times_confirmed = 1
        try:
            times_overridden = int(item.get("times_overridden", 0))
        except (TypeError, ValueError):
            times_overridden = 0
        weighted_similarity = similarity * _source_weight(source, str(item.get("timestamp") or item.get("created_at") or ""), times_confirmed, times_overridden)
        votes[key] = votes.get(key, 0.0) + weighted_similarity
        counts[key] = counts.get(key, 0) + 1
        previous = best_by_label.get(key)
        if previous is None or similarity > previous[0]:
            best_by_label[key] = (similarity, hash_distance, item)

    ranked = sorted(votes.items(), key=lambda pair: pair[1], reverse=True)
    (category, subcategory), vote_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    example_count = counts.get((category, subcategory), 0)
    best_similarity, best_distance, best_item = best_by_label.get((category, subcategory), (best_similarity, best_distance, best_item))

    agreement_bonus = min(0.05, max(0, example_count - 1) * 0.018)
    margin_bonus = min(0.05, max(0.0, vote_score - second_score) * 0.025)
    source = str(best_item.get("source") or "auto_observed")
    source_bonus = 0.03 if source == "manual_teach" else 0.0
    override_penalty = min(0.28, float(best_item.get("times_overridden", 0) or 0) * 0.08)
    confidence = min(0.99, best_similarity + agreement_bonus + margin_bonus + source_bonus - override_penalty)

    if confidence < threshold or example_count < min_examples:
        return None

    source_labels = {
        "manual_teach": "manual teaching",
        "learn_from_folder": "folder-learned",
        "auto_observed": "auto-observed",
    }
    label = f"{category}/{subcategory}" if subcategory else category
    reason = f"Matched {source_labels.get(source, source)} example {label} at {round(confidence * 100)}% similarity"
    if source != "manual_teach":
        reason = f"NeuroVector {source_labels.get(source, source)} match {round(confidence * 100)}% from {example_count} example(s)"

    return VisualMatch(
        category=category,
        subcategory=subcategory,
        confidence=confidence,
        distance=best_distance if best_distance != 999 else round((1.0 - best_similarity) * 100),
        source_path=str(best_item.get("source_path", "")),
        destination=str(best_item.get("destination", "")),
        reason=reason,
        source=source,
        matched_rule=matched_rule,
        debug_signals={
            "similarity": round(best_similarity, 4),
            "vote_score": round(vote_score, 4),
            "second_score": round(second_score, 4),
            "examples": example_count,
            "source": source,
        },
    )


def find_manual_teach_match(path: Path, threshold: float = 0.82) -> Optional[VisualMatch]:
    return find_memory_match(
        path,
        threshold=threshold,
        min_examples=1,
        sources={"manual_teach"},
        matched_rule="manual_teach_override",
    )


def find_folder_learned_match(path: Path, threshold: float = 0.86) -> Optional[VisualMatch]:
    return find_memory_match(
        path,
        threshold=threshold,
        min_examples=1,
        sources={"learn_from_folder"},
        matched_rule="folder_learning",
    )


def find_neural_match(path: Path, threshold: float = 0.84, min_examples: int = 1) -> Optional[VisualMatch]:
    return find_memory_match(
        path,
        threshold=threshold,
        min_examples=min_examples,
        sources=MEMORY_SOURCES,
        matched_rule="weighted_neurovector",
    )


def remember_visual_decision(
    path: Path,
    category: str,
    subcategory: str = "",
    app_name: str = "Unknown App",
    destination: str = "",
    taught: bool = False,
    source: str = "auto_observed",
    window_title: str = "",
    ocr_fingerprint: str = "",
    confidence: float = 1.0,
) -> None:
    if not category or category in {"Other", "_Review", "Review", "Duplicates"}:
        return
    if taught:
        source = "manual_teach"
    if source not in MEMORY_SOURCES:
        source = "auto_observed"
    try:
        hash_hex = perceptual_hash(path)
    except Exception:
        hash_hex = ""
    vector = extract_neuro_vector(path)

    data = load_brain()
    timestamp = _now_iso()
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 1.0 if source == "manual_teach" else 0.74
    item = {
        "hash": hash_hex,
        "perceptual_hash": hash_hex,
        "vector": vector,
        "feature_vector": vector,
        "category": str(category),
        "subcategory": str(subcategory or ""),
        "source": source,
        "app_name": str(app_name or "Unknown App"),
        "window_title": str(window_title or ""),
        "ocr_fingerprint": str(ocr_fingerprint or ""),
        "source_path": str(path),
        "destination": str(destination or ""),
        "taught": source == "manual_teach",
        "timestamp": timestamp,
        "created_at": timestamp,
        "confidence": max(0.0, min(1.0, confidence_value)),
        "times_confirmed": 1,
        "times_overridden": 0,
    }

    visual: List[Dict[str, Any]] = []
    merged = False
    for row in data.get("visual_memory", []):
        same_example = (
            str(row.get("hash", "") or row.get("perceptual_hash", "")) == hash_hex
            and str(row.get("category", "")) == category
            and str(row.get("subcategory", "")) == str(subcategory or "")
            and str(row.get("source") or "") == source
        )
        if same_example and not merged:
            previous = _normalize_memory_item(row)
            item["times_confirmed"] = int(previous.get("times_confirmed", 1)) + 1
            item["times_overridden"] = int(previous.get("times_overridden", 0))
            merged = True
            continue
        visual.append(row)
    visual.insert(0, item)
    data["visual_memory"] = visual[:MAX_VISUAL_MEMORY]
    if source == "manual_teach":
        history = list(data.get("teach_history", []))
        history.insert(0, item)
        data["teach_history"] = history[:800]
    save_brain(data)


def teach_image(path: Path, category: str, subcategory: str = "", config: Optional[Dict[str, Any]] = None) -> bool:
    path = Path(path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return False
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    if config:
        allowed.update(str(item).lower() for item in config.get("file_types", []))
    if path.suffix.lower() not in allowed:
        return False
    category = str(category).strip() or "Other"
    subcategory = str(subcategory).strip()
    remember_visual_decision(
        path,
        category=category,
        subcategory=subcategory,
        destination=str(path),
        taught=True,
        source="manual_teach",
        confidence=1.0,
    )
    return True


def teach_example(path: Path, category: str, subcategory: str = "", config: Optional[Dict[str, Any]] = None) -> bool:
    return teach_image(path, category, subcategory, config)


def remember_session_decision(category: str, subcategory: str, app_name: str, confidence: float) -> None:
    if not category or category in {"Other", "_Review", "Review", "Duplicates"}:
        return
    data = load_brain()
    sessions = list(data.get("sessions", []))
    sessions.insert(0, {
        "category": str(category),
        "subcategory": str(subcategory or ""),
        "app_name": str(app_name or "Unknown App"),
        "confidence": float(confidence),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    data["sessions"] = sessions[:MAX_SESSION_MEMORY]
    save_brain(data)


def find_recent_session(app_name: str, within_minutes: int = 4) -> Optional[VisualMatch]:
    app = str(app_name or "").strip().lower()
    if not app or app == "unknown app":
        return None

    cutoff = datetime.now() - timedelta(minutes=max(1, within_minutes))
    for item in load_brain().get("sessions", []):
        if str(item.get("app_name", "")).strip().lower() != app:
            continue
        try:
            created_at = datetime.fromisoformat(str(item.get("created_at", "")))
        except ValueError:
            continue
        if created_at < cutoff:
            continue
        base_conf = float(item.get("confidence", 0.74))
        return VisualMatch(
            category=str(item.get("category", "Other")) or "Other",
            subcategory=str(item.get("subcategory", "")) or "",
            confidence=min(0.82, max(0.70, base_conf - 0.10)),
            reason="Recent session memory",
        )
    return None


def learn_from_folders(root_folder: str | Path, config: Optional[Dict[str, Any]] = None) -> int:
    """Scan a folder tree and learn from category/subcategory folder names.

    Folder names become labels. Example:
    Pictures/CleanShot/Photos/Trips/image.png -> category Photos, subcategory Trips.
    Pictures/CleanShot/Code/JavaScript/image.png -> category Code, subcategory JavaScript.
    """
    folder = expand_path(root_folder)
    if not folder.exists():
        return 0

    config = config or {}
    file_types = {str(item).lower() for item in config.get("file_types", [])} or {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    ignored = {"duplicates", "other", "review", "_review", "unknown app", "unknown"}

    learned = 0
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in file_types:
            continue
        try:
            rel = path.relative_to(folder)
        except ValueError:
            continue
        parts = list(rel.parts[:-1])
        if not parts:
            continue
        category = parts[0]
        subcategory = parts[1] if len(parts) > 1 else ""
        if category.strip().lower() in ignored:
            continue
        remember_visual_decision(
            path,
            category=category,
            subcategory=subcategory,
            destination=str(path),
            source="learn_from_folder",
            confidence=0.82,
        )
        learned += 1
    return learned


def learn_from_output_folders(config: Dict[str, Any]) -> int:
    return learn_from_folders(expand_path(config.get("output_folder", "~/Pictures/CleanShot")), config)


def reset_learning_memory() -> None:
    save_brain(_empty_brain())


def forget_examples(
    category: str | None = None,
    subcategory: str | None = None,
    source: str | None = None,
) -> int:
    data = load_brain()
    category_norm = str(category or "").strip().lower()
    subcategory_norm = str(subcategory or "").strip().lower()
    source_norm = str(source or "").strip()
    if not category_norm and not subcategory_norm and not source_norm:
        return 0

    kept: List[Dict[str, Any]] = []
    removed = 0
    for item in data.get("visual_memory", []):
        matches = True
        if category_norm:
            matches = matches and str(item.get("category", "")).strip().lower() == category_norm
        if subcategory_norm:
            matches = matches and str(item.get("subcategory", "")).strip().lower() == subcategory_norm
        if source_norm:
            matches = matches and str(item.get("source", "")).strip() == source_norm
        if matches:
            removed += 1
        else:
            kept.append(item)

    data["visual_memory"] = kept
    if removed:
        data["teach_history"] = [
            item
            for item in data.get("teach_history", [])
            if not (
                (not category_norm or str(item.get("category", "")).strip().lower() == category_norm)
                and (not subcategory_norm or str(item.get("subcategory", "")).strip().lower() == subcategory_norm)
                and (not source_norm or str(item.get("source", "")).strip() == source_norm)
            )
        ]
        save_brain(data)
    return removed


def mark_override(
    old_category: str,
    old_subcategory: str,
    new_category: str,
    new_subcategory: str,
) -> int:
    old_key = (str(old_category or "").strip().lower(), str(old_subcategory or "").strip().lower())
    new_key = (str(new_category or "").strip().lower(), str(new_subcategory or "").strip().lower())
    if not old_key[0] or old_key == new_key:
        return 0

    data = load_brain()
    changed = 0
    for item in data.get("visual_memory", []):
        key = (str(item.get("category", "")).strip().lower(), str(item.get("subcategory", "")).strip().lower())
        if key == old_key:
            item["times_overridden"] = int(item.get("times_overridden", 0) or 0) + 1
            changed += 1
    if changed:
        save_brain(data)
    return changed


def memory_stats() -> Dict[str, int]:
    stats = {"total": 0, "manual_teach": 0, "learn_from_folder": 0, "auto_observed": 0}
    for item in load_brain().get("visual_memory", []):
        stats["total"] += 1
        source = str(item.get("source") or "auto_observed")
        if source in stats:
            stats[source] += 1
    return stats


def memory_count() -> int:
    return len(load_brain().get("visual_memory", []))


def taught_count() -> int:
    return len([item for item in load_brain().get("visual_memory", []) if item.get("source") == "manual_teach" or item.get("taught")])
