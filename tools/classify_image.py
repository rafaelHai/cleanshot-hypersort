#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleanshot.core.classifier import (  # noqa: E402
    _code_vision_profile,
    _image_profile,
    classify_screenshot,
    classify_smart_screenshot,
    detect_code_language,
)
from cleanshot.core.config import load_config, normalize_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Test CleanShot classification on a single image.")
    parser.add_argument("image", help="Path to screenshot/image")
    parser.add_argument("--ocr", action="store_true", help="Enable optional OCR if pytesseract/Tesseract are installed")
    parser.add_argument("--debug", action="store_true", help="Print VisionCore image features")
    args = parser.parse_args()

    image = Path(args.image).expanduser().resolve()
    if not image.exists():
        print(f"Image not found: {image}", file=sys.stderr)
        return 1

    config = normalize_config({**load_config(), "organization_mode": "smart", "enable_ocr": args.ocr})
    result = classify_smart_screenshot(image, config)

    print("CleanShot Smart Mode")
    print(f"Image:       {image}")
    print(f"Category:    {result.category}")
    print(f"Subcategory: {result.subcategory or '-'}")
    print(f"Confidence:  {round(result.confidence * 100)}%")
    print(f"Source:      {result.source}")
    print(f"App:         {result.app_name}")
    print(f"Reason:      {result.reason}")
    if result.signals:
        print(f"Signals:     {', '.join(result.signals)}")

    if args.debug:
        profile = _image_profile(image)
        code = _code_vision_profile(image, profile)
        print("\nVisionCore debug")
        print(f"ImageProfile:    photo={profile.photo_score:.2f}, text_ui={profile.text_ui_score:.2f}, layout={profile.layout_text_score:.2f}, dark={profile.dark_score:.2f}, color={profile.colorfulness:.2f}, edges={profile.edge_density:.2f}")
        print(f"CodeVision:      score={code.score:.2f}, rows={code.row_count}, line={code.line_score:.2f}, gutter={code.gutter_score:.2f}, contrast={code.contrast_score:.2f}, canvas={code.editor_canvas_score:.2f}, syntax={code.syntax_color_score:.2f}, columns={code.mono_column_score:.2f}")
        language, lang_confidence, lang_reason = detect_code_language(result.reason + " " + image.name, filename=image.name, app_name=result.app_name)
        print(f"LanguageMind:    {language} ({round(lang_confidence * 100)}%) - {lang_reason}")
        print(f"Code reasons:    {', '.join(code.reason)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
