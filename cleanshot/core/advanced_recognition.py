from __future__ import annotations

import hashlib
import importlib.util
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class AdvancedSignals:
    average_hash: str = ""
    difference_hash: str = ""
    ocr_fingerprint: str = ""
    color_histogram: List[float] = field(default_factory=list)
    edge_density: float = 0.0
    text_density: float = 0.0
    layout_segments: Dict[str, float] = field(default_factory=dict)
    optional_backends: Dict[str, bool] = field(default_factory=dict)

    def as_debug(self) -> Dict[str, Any]:
        return {
            "average_hash": self.average_hash,
            "difference_hash": self.difference_hash,
            "ocr_fingerprint": self.ocr_fingerprint[:16],
            "edge_density": round(self.edge_density, 4),
            "text_density": round(self.text_density, 4),
            "layout_segments": {key: round(value, 4) for key, value in self.layout_segments.items()},
            "optional_backends": self.optional_backends,
        }


def recognition_status() -> Dict[str, bool]:
    system = platform.system().lower()
    return {
        "pillow": importlib.util.find_spec("PIL") is not None,
        "pytesseract": importlib.util.find_spec("pytesseract") is not None,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
        "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
        "apple_vision_possible": system == "darwin",
        "windows_ocr_possible": system == "windows",
    }


def ocr_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not normalized:
        return ""
    tokens = re.findall(r"[a-z0-9_./:-]+", normalized)
    compact = " ".join(tokens[:160])
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()


def extract_advanced_signals(path: Path, ocr_text: str = "") -> AdvancedSignals:
    status = recognition_status()
    if not status["pillow"]:
        return AdvancedSignals(optional_backends=status, ocr_fingerprint=ocr_fingerprint(ocr_text))

    try:
        from PIL import Image, ImageFilter
    except ImportError:
        return AdvancedSignals(optional_backends=status, ocr_fingerprint=ocr_fingerprint(ocr_text))

    try:
        with Image.open(path) as source:
            rgb = source.convert("RGB").resize((64, 64))
            gray = rgb.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
    except Exception:
        return AdvancedSignals(optional_backends=status, ocr_fingerprint=ocr_fingerprint(ocr_text))

    gray_pixels = list(gray.getdata())
    edge_pixels = list(edges.getdata())
    rgb_pixels = list(rgb.getdata())
    total = max(len(gray_pixels), 1)

    average = sum(gray_pixels) / total
    average_bits = "".join("1" if value >= average else "0" for value in gray_pixels[:64])
    average_hash = f"{int(average_bits or '0', 2):016x}"

    dhash_bits: List[str] = []
    small = gray.resize((9, 8))
    pixels = list(small.getdata())
    for row in range(8):
        offset = row * 9
        for col in range(8):
            dhash_bits.append("1" if pixels[offset + col] > pixels[offset + col + 1] else "0")
    difference_hash = f"{int(''.join(dhash_bits), 2):016x}"

    color_hist = [0] * 12
    for r, g, b in rgb_pixels:
        channel = max(range(3), key=(r, g, b).__getitem__)
        brightness_bucket = min(3, int(((r + g + b) / 3) // 64))
        color_hist[channel * 4 + brightness_bucket] += 1
    color_histogram = [round(count / total, 5) for count in color_hist]

    edge_density = sum(1 for value in edge_pixels if value > 35) / total
    text_density = _estimate_text_density(gray_pixels, 64, 64)
    layout_segments = _layout_segments(edge_pixels, 64, 64)

    return AdvancedSignals(
        average_hash=average_hash,
        difference_hash=difference_hash,
        ocr_fingerprint=ocr_fingerprint(ocr_text),
        color_histogram=color_histogram,
        edge_density=edge_density,
        text_density=text_density,
        layout_segments=layout_segments,
        optional_backends=status,
    )


def _estimate_text_density(gray_pixels: List[int], width: int, height: int) -> float:
    rows_with_transitions = 0
    for y in range(height):
        row = gray_pixels[y * width : (y + 1) * width]
        transitions = sum(1 for left, right in zip(row, row[1:]) if abs(left - right) > 32)
        if transitions >= 8:
            rows_with_transitions += 1
    return rows_with_transitions / max(height, 1)


def _layout_segments(edge_pixels: List[int], width: int, height: int) -> Dict[str, float]:
    def density(x1: int, y1: int, x2: int, y2: int) -> float:
        values = [edge_pixels[y * width + x] for y in range(y1, y2) for x in range(x1, x2)]
        return sum(1 for value in values if value > 35) / max(len(values), 1)

    return {
        "top_bar": density(0, 0, width, max(1, height // 8)),
        "left_sidebar": density(0, 0, max(1, width // 5), height),
        "center": density(width // 5, height // 5, width - width // 5, height - height // 5),
        "bottom_bar": density(0, height - max(1, height // 8), width, height),
    }
