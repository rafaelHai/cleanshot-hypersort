from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import app_data_dir, expand_path


MAX_VISUAL_MEMORY = 5000
MAX_SESSION_MEMORY = 400


@dataclass(frozen=True)
class VisualMatch:
    category: str
    subcategory: str
    confidence: float
    distance: int = 0
    source_path: str = ""
    destination: str = ""
    reason: str = ""


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
    data.setdefault("visual_memory", [])
    data.setdefault("sessions", [])
    data.setdefault("manual_rules", {})
    data.setdefault("teach_history", [])
    return data


def save_brain(data: Dict[str, Any]) -> None:
    path = brain_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    visual = list(data.get("visual_memory", []))[:MAX_VISUAL_MEMORY]
    sessions = list(data.get("sessions", []))[:MAX_SESSION_MEMORY]
    safe_data = {
        "version": 2,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "visual_memory": visual,
        "sessions": sessions,
        "manual_rules": data.get("manual_rules", {}),
        "teach_history": list(data.get("teach_history", []))[:800],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe_data, handle, indent=2, ensure_ascii=False)


def _empty_brain() -> Dict[str, Any]:
    return {"version": 2, "visual_memory": [], "sessions": [], "manual_rules": {}, "teach_history": []}


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


def find_neural_match(path: Path, threshold: float = 0.84, min_examples: int = 1) -> Optional[VisualMatch]:
    vector = extract_neuro_vector(path)
    if not vector:
        return None

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for item in load_brain().get("visual_memory", []):
        item_vector = item.get("vector")
        if not isinstance(item_vector, list):
            continue
        similarity = cosine_similarity(vector, [float(value) for value in item_vector])
        if similarity > 0:
            candidates.append((similarity, item))

    if not candidates:
        return None

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    top = candidates[:7]

    # Weighted vote by category/subcategory. This is stronger than 1-nearest-neighbor.
    votes: Dict[Tuple[str, str], float] = {}
    counts: Dict[Tuple[str, str], int] = {}
    best_item = top[0][1]
    best_similarity = top[0][0]
    for similarity, item in top:
        key = (str(item.get("category", "Other")) or "Other", str(item.get("subcategory", "")) or "")
        votes[key] = votes.get(key, 0.0) + similarity
        counts[key] = counts.get(key, 0) + 1

    ranked = sorted(votes.items(), key=lambda pair: pair[1], reverse=True)
    (category, subcategory), vote_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    example_count = counts.get((category, subcategory), 0)

    agreement_bonus = min(0.05, max(0, example_count - 1) * 0.018)
    margin_bonus = min(0.04, max(0.0, vote_score - second_score) * 0.02)
    confidence = min(0.98, best_similarity + agreement_bonus + margin_bonus)

    if confidence < threshold or example_count < min_examples:
        return None

    return VisualMatch(
        category=category,
        subcategory=subcategory,
        confidence=confidence,
        distance=round((1.0 - best_similarity) * 100),
        source_path=str(best_item.get("source_path", "")),
        destination=str(best_item.get("destination", "")),
        reason=f"NeuroVector learned match {round(confidence * 100)}% from {example_count} example(s)",
    )


def remember_visual_decision(
    path: Path,
    category: str,
    subcategory: str = "",
    app_name: str = "Unknown App",
    destination: str = "",
    taught: bool = False,
) -> None:
    if not category or category in {"Other", "_Review", "Review", "Duplicates"}:
        return
    try:
        hash_hex = perceptual_hash(path)
    except Exception:
        hash_hex = ""
    vector = extract_neuro_vector(path)

    data = load_brain()
    item = {
        "hash": hash_hex,
        "vector": vector,
        "category": str(category),
        "subcategory": str(subcategory or ""),
        "app_name": str(app_name or "Unknown App"),
        "source_path": str(path),
        "destination": str(destination or ""),
        "taught": bool(taught),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    visual = [row for row in data.get("visual_memory", []) if not (
        str(row.get("hash", "")) == hash_hex and str(row.get("category", "")) == category and str(row.get("subcategory", "")) == str(subcategory or "")
    )]
    visual.insert(0, item)
    data["visual_memory"] = visual[:MAX_VISUAL_MEMORY]
    if taught:
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
    remember_visual_decision(path, category=category, subcategory=subcategory, destination=str(path), taught=True)
    return True


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


def learn_from_output_folders(config: Dict[str, Any]) -> int:
    """Scan organized folders and teach CleanShot from the user's folder choices.

    Folder names become labels. Example:
    Pictures/CleanShot/Photos/Trips/image.png -> category Photos, subcategory Trips.
    Pictures/CleanShot/Code/JavaScript/image.png -> category Code, subcategory JavaScript.
    """
    output_folder = expand_path(config.get("output_folder", "~/Pictures/CleanShot"))
    if not output_folder.exists():
        return 0

    file_types = {str(item).lower() for item in config.get("file_types", [])}
    ignored = {"duplicates", "other", "review", "_review", "unknown app", "unknown"}

    learned = 0
    for path in output_folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in file_types:
            continue
        try:
            rel = path.relative_to(output_folder)
        except ValueError:
            continue
        parts = list(rel.parts[:-1])
        if not parts:
            continue
        category = parts[0]
        subcategory = parts[1] if len(parts) > 1 else ""
        if category.strip().lower() in ignored:
            continue
        remember_visual_decision(path, category=category, subcategory=subcategory, destination=str(path), taught=True)
        learned += 1
    return learned


def memory_count() -> int:
    return len(load_brain().get("visual_memory", []))


def taught_count() -> int:
    return len([item for item in load_brain().get("visual_memory", []) if item.get("taught")])
