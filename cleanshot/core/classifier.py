from __future__ import annotations

import math
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    confidence: float
    reason: str
    app_name: str = "Unknown App"
    source: str = "rules"
    subcategory: str = ""
    signals: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageProfile:
    width: int = 0
    height: int = 0
    ratio: float = 0.0
    stddev: float = 0.0
    colorfulness: float = 0.0
    edge_density: float = 0.0
    photo_score: float = 0.0
    text_ui_score: float = 0.0
    layout_text_score: float = 0.0
    dark_score: float = 0.0


@dataclass(frozen=True)
class CodeVisionProfile:
    """OCR-free visual evidence that an image is code/editor/terminal content.

    This is not a neural network. It is a local computer-vision ensemble that looks
    for things code screenshots normally have: repeated text rows, a line-number
    gutter, high contrast, editor-like dark/light canvases, and low photo texture.
    """

    score: float = 0.0
    line_score: float = 0.0
    gutter_score: float = 0.0
    contrast_score: float = 0.0
    editor_canvas_score: float = 0.0
    syntax_color_score: float = 0.0
    mono_column_score: float = 0.0
    row_count: int = 0
    active_row_ratio: float = 0.0
    reason: Tuple[str, ...] = ()


# App names should only be strong in App mode. In Specific mode they are hints.
CATEGORY_APPS: Dict[str, Iterable[str]] = {
    "Code": [
        "visual studio code", "vs code", "code", "cursor", "xcode", "pycharm", "webstorm",
        "intellij", "idea", "android studio", "sublime text", "atom", "zed", "nova", "bbedit", "terminal", "iterm2", "iterm", "windows terminal", "powershell", "vim", "neovim",
    ],
    "Browser": ["google chrome", "chrome", "safari", "firefox", "arc", "brave browser", "edge", "opera"],
    "Video": ["quicktime player", "vlc", "iina", "tv", "youtube", "netflix", "plex"],
    "Chat": ["discord", "slack", "messages", "whatsapp", "telegram", "signal", "messenger", "teams"],
    "Design": ["figma", "sketch", "adobe photoshop", "photoshop", "illustrator", "canva", "framer"],
    "Documents": ["pages", "microsoft word", "word", "preview", "adobe acrobat", "notes", "notion", "obsidian"],
    "Photos": ["photos"],
    "Games": ["steam", "epic games", "roblox", "minecraft", "unreal editor", "fivem", "gta"],
}

APP_FOLDER_ALIASES: Dict[str, str] = {
    "google chrome": "Chrome",
    "chrome": "Chrome",
    "safari": "Safari",
    "firefox": "Firefox",
    "arc": "Arc",
    "brave browser": "Brave",
    "visual studio code": "VS Code",
    "code": "VS Code",
    "cursor": "Cursor",
    "xcode": "Xcode",
    "discord": "Discord",
    "slack": "Slack",
    "messages": "Messages",
    "whatsapp": "WhatsApp",
    "telegram": "Telegram",
    "figma": "Figma",
    "photos": "Photos",
    "preview": "Preview",
    "terminal": "Terminal",
    "iterm2": "iTerm",
    "finder": "Finder",
    "vs code": "VS Code",
    "android studio": "Android Studio",
    "idea": "IntelliJ IDEA",
    "intellij idea": "IntelliJ IDEA",
    "webstorm": "WebStorm",
    "pycharm": "PyCharm",
    "zed": "Zed",
    "sublime text": "Sublime Text",
    "brave": "Brave",
    "microsoft edge": "Edge",
    "edge": "Edge",
    "opera": "Opera",
    "iterm": "iTerm",
    "windows terminal": "Windows Terminal",
    "powershell": "PowerShell",
    "notion": "Notion",
    "obsidian": "Obsidian",
    "microsoft word": "Word",
    "word": "Word",
    "excel": "Excel",
    "powerpoint": "PowerPoint",
    "teams": "Teams",
    "zoom": "Zoom",
    "canva": "Canva",
}

# Whole-word phrases. Do not use tiny substring matches like "api" inside random words.
CATEGORY_KEYWORDS: Dict[str, Iterable[str]] = {
    "Browser": [
        "browser", "search", "google", "safari", "chrome", "firefox", "arc", "brave", "edge", "tab", "new tab",
        "address bar", "url", "website", "web page", "webpage", "docs", "stackoverflow",
    ],
    "Video": [
        "youtube", "netflix", "video", "watch", "subscribe", "views", "play", "pause",
        "fullscreen", "timeline", "movie", "episode", "stream", "trailer",
    ],
    "Chat": [
        "discord", "slack", "whatsapp", "telegram", "message", "typing", "reply",
        "thread", "channel", "dm", "chat", "send message",
    ],
    "Design": [
        "figma", "design", "canvas", "frame", "prototype", "layers", "component",
        "photoshop", "illustrator", "gradient", "font", "spacing", "auto layout",
    ],
    "Documents": [
        "document", "pdf", "page", "chapter", "paragraph", "invoice", "receipt",
        "agreement", "resume", "notes", "notion", "report", "essay", "spreadsheet",
    ],
    "Photos": [
        "family", "photo", "image", "vacation", "birthday", "wedding", "album",
        "portrait", "trip", "travel", "memories", "memorial", "dec trip", "beach",
    ],
    "Games": [
        "steam", "game", "server", "fps", "unreal", "gta", "fivem", "minecraft",
        "roblox", "play", "match", "lobby", "health", "ammo",
    ],
}



# ---------------------------------------------------------------------------
# LanguageMind: broad code-language fingerprinting
# ---------------------------------------------------------------------------
# CleanShot is not a compiler, but this table lets it recognize the most common
# languages from OCR/window title/filename tokens. The scoring function below is
# deliberately conservative and works even when OCR drops punctuation.
CODE_LANGUAGE_FINGERPRINTS: Dict[str, Dict[str, Iterable[str]]] = {
    "JavaScript": {
        "extensions": ["js", "jsx", "mjs", "cjs"],
        "keywords": ["javascript", "js", "node", "npm", "yarn", "pnpm", "console log", "document queryselector", "add event listener", "function", "const", "let", "var", "require", "module exports", "promise", "math floor"],
        "patterns": [r"\b(console\.log|console log)\b", r"\b(function|const|let|var)\b", r"=>", r"\$\{[^}]+\}", r"\bMath\.[A-Za-z]+"],
    },
    "TypeScript": {
        "extensions": ["ts", "tsx", "d.ts"],
        "keywords": ["typescript", "tsx", "interface", "type ", "readonly", "enum", "implements", "as const", "generic"],
        "patterns": [r"\binterface\s+[A-Za-z_]", r"\btype\s+[A-Za-z_]\s*=", r":\s*(string|number|boolean|unknown|any)\b", r"<[A-Z][A-Za-z0-9_]*>"],
    },
    "Python": {
        "extensions": ["py", "ipynb"],
        "keywords": ["python", "pytest", "pip", "venv", "django", "flask", "fastapi", "pandas", "numpy", "self", "elif", "lambda"],
        "patterns": [r"\bdef\s+[A-Za-z_]", r"\bclass\s+[A-Za-z_]", r"\bimport\s+[A-Za-z_]", r"\bfrom\s+[A-Za-z_][\w.]*\s+import", r"\belif\b", r"if\s+__name__\s*=="],
    },
    "PHP": {
        "extensions": ["php", "phtml"],
        "keywords": ["php", "laravel", "composer", "namespace", "use ", "echo", "mysqli", "pdo", "artisan"],
        "patterns": [r"<\?php", r"\$[A-Za-z_][A-Za-z0-9_]*", r"->\s*[A-Za-z_]", r"\bnamespace\s+[A-Za-z_]", r"\bpublic\s+function\b"],
    },
    "HTML": {
        "extensions": ["html", "htm"],
        "keywords": ["html", "doctype", "div", "span", "body", "head", "section", "button", "input"],
        "patterns": [r"<!doctype\s+html", r"</?[a-z][a-z0-9-]*(\s|>|/>)", r"\bclass=", r"\bid="],
    },
    "CSS": {
        "extensions": ["css", "scss", "sass", "less"],
        "keywords": ["css", "tailwind", "stylesheet", "display flex", "grid", "margin", "padding", "border radius", "background", "font size"],
        "patterns": [r"[.#][A-Za-z_][\w-]*\s*\{", r"\b(display|position|margin|padding|background|color|font-size|border-radius)\s*:", r"@media\b", r"@keyframes\b"],
    },
    "SQL": {
        "extensions": ["sql"],
        "keywords": ["sql", "mysql", "postgres", "sqlite", "select", "insert", "update", "delete", "where", "join", "group by", "order by", "create table"],
        "patterns": [r"\bSELECT\b.+\bFROM\b", r"\bINSERT\s+INTO\b", r"\bUPDATE\b.+\bSET\b", r"\bCREATE\s+TABLE\b", r"\bWHERE\b.+="],
    },
    "JSON": {
        "extensions": ["json"],
        "keywords": ["json", "package json", "tsconfig", "composer json"],
        "patterns": [r"\{\s*\"[A-Za-z0-9_ -]+\"\s*:", r"\"scripts\"\s*:", r"\"dependencies\"\s*:"],
    },
    "Shell": {
        "extensions": ["sh", "bash", "zsh", "fish"],
        "keywords": ["terminal", "shell", "bash", "zsh", "chmod", "sudo", "brew", "pip install", "npm install", "cd ", "mkdir", "grep", "curl", "git clone"],
        "patterns": [r"^\s*(sudo|brew|npm|python3?|pip|git|cd|mkdir|rm|chmod)\b", r"\$\s+\w+", r"#!/bin/(bash|zsh|sh)", r"\bexport\s+[A-Z_]+="],
    },
    "C++": {
        "extensions": ["cpp", "cc", "cxx", "hpp", "hxx"],
        "keywords": ["c++", "cpp", "std", "cout", "cin", "iostream", "vector", "namespace", "template"],
        "patterns": [r"#include\s*<", r"std::", r"\bcout\s*<<", r"\btemplate\s*<", r"\bint\s+main\s*\("],
    },
    "C": {
        "extensions": ["c", "h"],
        "keywords": ["stdio", "printf", "scanf", "malloc", "free", "struct", "typedef"],
        "patterns": [r"#include\s*<stdio\.h>", r"\bprintf\s*\(", r"\bscanf\s*\(", r"\bint\s+main\s*\("],
    },
    "C#": {
        "extensions": ["cs"],
        "keywords": ["csharp", "c#", "using system", "namespace", "public class", "private", "static void main", "unityengine"],
        "patterns": [r"\busing\s+System", r"\bnamespace\s+[A-Za-z_]", r"\bpublic\s+class\b", r"\bConsole\.WriteLine\b"],
    },
    "Java": {
        "extensions": ["java"],
        "keywords": ["java", "spring", "public static void main", "system out println", "package", "extends", "implements"],
        "patterns": [r"\bpublic\s+class\b", r"\bSystem\.out\.println\b", r"\bpackage\s+[a-zA-Z_]", r"\bpublic\s+static\s+void\s+main\b"],
    },
    "Go": {
        "extensions": ["go"],
        "keywords": ["golang", "go", "package main", "func", "fmt println", "goroutine", "channel"],
        "patterns": [r"\bpackage\s+main\b", r"\bfunc\s+[A-Za-z_]", r"\bfmt\.Print", r"\bgo\s+[A-Za-z_]"],
    },
    "Rust": {
        "extensions": ["rs"],
        "keywords": ["rust", "cargo", "fn main", "let mut", "println", "use std", "impl", "trait"],
        "patterns": [r"\bfn\s+[A-Za-z_]", r"\blet\s+mut\b", r"\bprintln!", r"\bimpl\s+", r"\buse\s+std::"],
    },
    "Swift": {
        "extensions": ["swift"],
        "keywords": ["swift", "swiftui", "uikit", "import swiftui", "var body", "view", "struct"],
        "patterns": [r"\bimport\s+SwiftUI\b", r"\bstruct\s+[A-Za-z_].*:\s*View", r"\bvar\s+body\s*:\s*some\s+View", r"\blet\s+[A-Za-z_]"],
    },
    "Kotlin": {
        "extensions": ["kt", "kts"],
        "keywords": ["kotlin", "fun main", "val", "var", "println", "android"],
        "patterns": [r"\bfun\s+[A-Za-z_]", r"\bval\s+[A-Za-z_]", r"\bdata\s+class\b"],
    },
    "Ruby": {
        "extensions": ["rb"],
        "keywords": ["ruby", "rails", "gem", "bundle", "puts", "def ", "end", "class"],
        "patterns": [r"\bdef\s+[A-Za-z_]", r"\bputs\s+", r"\brequire\s+['\"]", r"\bend\b"],
    },
    "Lua": {
        "extensions": ["lua"],
        "keywords": ["lua", "local", "end", "elseif", "qbcore", "fivem", "citizen", "registernetevent"],
        "patterns": [r"\blocal\s+[A-Za-z_]", r"\bfunction\s+[A-Za-z_]", r"\bend\b", r"RegisterNetEvent"],
    },
    "Dart": {
        "extensions": ["dart"],
        "keywords": ["dart", "flutter", "widget", "buildcontext", "materialapp", "scaffold"],
        "patterns": [r"\bWidget\s+build\s*\(", r"\bimport\s+'package:flutter", r"\bclass\s+[A-Za-z_].*extends\s+StatelessWidget"],
    },
    "R": {
        "extensions": ["r", "rmd"],
        "keywords": ["rstats", "tidyverse", "ggplot", "data frame", "library", "dplyr"],
        "patterns": [r"\blibrary\s*\(", r"<-", r"\bggplot\s*\(", r"%>%"],
    },
    "Markdown": {
        "extensions": ["md", "markdown"],
        "keywords": ["readme", "markdown", "```", "## ", "### ", "- ", "github flavored"],
        "patterns": [r"^\s*#{1,6}\s+", r"```[a-zA-Z0-9_-]*", r"\[[^\]]+\]\([^)]+\)", r"^\s*-\s+"],
    },
    "YAML": {
        "extensions": ["yml", "yaml"],
        "keywords": ["yaml", "docker compose", "github actions", "workflow", "services", "version"],
        "patterns": [r"\bversion:\s*['\"]?\d", r"\bservices:\s*$", r"\bname:\s+", r"\bsteps:\s*$"],
    },
    "XML": {
        "extensions": ["xml"],
        "keywords": ["xml", "plist", "manifest"],
        "patterns": [r"<\?xml", r"</[A-Za-z][A-Za-z0-9_-]+>", r"<plist\b"],
    },
    "Docker": {
        "extensions": ["dockerfile"],
        "keywords": ["dockerfile", "docker", "container", "from alpine", "from node", "copy", "run", "cmd", "entrypoint"],
        "patterns": [r"\bFROM\s+[a-zA-Z0-9_./:-]+", r"\bRUN\s+", r"\bCOPY\s+", r"\bENTRYPOINT\b"],
    },
}

COMMON_IDE_APPS = [
    "vs code", "visual studio code", "cursor", "xcode", "pycharm", "webstorm", "intellij", "android studio",
    "sublime", "zed", "terminal", "iterm", "windows terminal", "powershell", "vim", "neovim",
]


def detect_code_language(text: str, filename: str = "", app_name: str = "") -> Tuple[str, float, str]:
    """Return (language, confidence, reason) from noisy OCR/title/filename text."""
    combined = _normalized_text(" ".join([filename, text, app_name]))
    if not combined:
        return "Projects", 0.0, "No code language text"

    scores: Dict[str, float] = {}
    reasons: Dict[str, List[str]] = {}

    def add(language: str, amount: float, reason: str) -> None:
        scores[language] = scores.get(language, 0.0) + amount
        reasons.setdefault(language, []).append(reason)

    filename_lower = filename.lower()
    for language, data in CODE_LANGUAGE_FINGERPRINTS.items():
        for ext in data.get("extensions", []):
            ext = str(ext).lower().lstrip(".")
            if filename_lower.endswith(f".{ext}") or f" {ext} " in f" {combined} ":
                add(language, 0.42, f"extension .{ext}")
                break
        keyword_hits = [kw for kw in data.get("keywords", []) if _contains_phrase(combined, kw)]
        if keyword_hits:
            add(language, min(0.46, 0.10 * len(keyword_hits)), f"keywords: {', '.join(keyword_hits[:4])}")
        pattern_hits = []
        for pattern in data.get("patterns", []):
            if re.search(pattern, text or combined, flags=re.IGNORECASE | re.MULTILINE):
                pattern_hits.append(pattern)
        if pattern_hits:
            add(language, min(0.50, 0.16 * len(pattern_hits)), f"syntax fingerprints: {len(pattern_hits)}")

    # Framework overrides / refinements.
    if _has_any_keyword(combined, ["react", "jsx", "tsx", "useeffect", "usestate"]):
        add("JavaScript", 0.22, "React/JS framework hint")
    if _has_any_keyword(combined, ["tailwind", "className", "classname"]):
        add("CSS", 0.12, "styling/framework hint")
    if _has_any_keyword(combined, ["fiveM", "qbcore", "citizen createthread", "registercommand"]):
        add("Lua", 0.26, "FiveM/Lua hint")

    if not scores:
        if _has_any_keyword(combined, COMMON_IDE_APPS):
            return "Projects", 0.42, "Code app but language unknown"
        return "Projects", 0.0, "No language match"

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second
    confidence = min(0.98, 0.46 + best_score + min(0.08, max(0.0, margin) * 0.10))

    # If TypeScript and JavaScript tie closely, prefer TypeScript when TS-only clues exist.
    if best == "JavaScript" and scores.get("TypeScript", 0) >= best_score - 0.10:
        if _has_any_keyword(combined, ["typescript", "interface", "type ", "tsx"]):
            best = "TypeScript"
            confidence = max(confidence, 0.82)

    return best, confidence, "; ".join(reasons.get(best, []))


CODE_STRONG_WORDS = [
    "function", "const", "let", "var", "class", "import", "export", "return", "console",
    "console.log", "def", "elif", "async", "await", "lambda", "public", "private", "static",
    "extends", "implements", "interface", "namespace", "package", "include", "require", "echo",
    "select", "insert", "update", "delete", "where", "join", "commit", "pull request",
]

CODE_TECH_WORDS = [
    "python", "javascript", "typescript", "php", "html", "css", "react", "vue", "node", "npm",
    "json", "api", "database", "sql", "github", "git", "docker", "localhost", "terminal", "vscode",
]

CODE_SYNTAX_PATTERNS = [
    r"[{};]{2,}",
    r"=>",
    r"</?[a-z][a-z0-9-]*(\s|>|/>)",
    r"\b(if|for|while|switch|catch)\s*\(",
    r"\b(def|class)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*[:(]",
    r"\b(function)\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*\(",
    r"\b(import|from)\s+[a-zA-Z0-9_./-]+",
    r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*['\"0-9\[{]",
    r"\b\d+\s*\|\s*\S+",  # line-number gutter copied by OCR
]

URL_PATTERNS = [
    r"https?://",
    r"www\.",
    r"\b[a-z0-9-]+\.(com|org|net|dev|app|io|co|ai|gg|edu)\b",
]

CALENDAR_WEATHER_WORDS = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "clear", "cloudy", "rain", "sunny", "weather", "fahrenheit", "celsius", "trip",
]


def classify_screenshot(path: Path, config: Dict[str, Any]) -> ClassificationResult:
    """Classify a screenshot with conservative local rules.

    The goal is not to be flashy; the goal is to avoid embarrassing wrong folders.
    This version uses confidence gates and whole-word matching, so OCR mistakes like
    reading "Clear" as "class" should not instantly become Code.
    """

    app_name = get_frontmost_app_name() or "Unknown App"
    filename_text = _normalized_text(path.stem)
    ocr_text = _read_text_with_optional_ocr(path, config) if config.get("enable_ocr", False) else ""
    text = " ".join(part for part in [filename_text, ocr_text] if part).strip()
    profile = _image_profile(path)

    # 1) Strong photo / personal screenshot check. This should beat weak OCR noise.
    photo_result = _classify_photo_like(text, profile, app_name)
    if photo_result:
        return photo_result

    # 2) OCR-free visual code detection. This catches IDE/terminal/Markdown code
    # screenshots even when Tesseract is not installed.
    visual_code_result = _classify_visual_code(path, profile, app_name)
    if visual_code_result:
        return visual_code_result

    # 3) Strict text code detection. Code needs real code signals, not one fuzzy OCR word.
    code_result = _classify_code(text, profile, app_name)
    if code_result:
        return code_result

    # 4) Other specific content groups.
    specific_result = _classify_specific_text(text, profile, app_name)
    if specific_result:
        return specific_result

    # 5) App hints are allowed, but only as medium confidence in Specific mode.
    # App mode still uses classify_app_folder() below.
    app_category = _category_from_app(app_name)
    if app_category:
        confidence = 0.62
        reason = f"App hint only: {app_name}"

        # If the app hint matches the image shape, increase confidence.
        if app_category == "Photos" and profile.photo_score >= 0.45:
            confidence = 0.78
            reason = f"Photos app with photo-like image ({profile.photo_score:.2f})"
        elif app_category == "Browser" and _has_url_signal(text):
            confidence = 0.82
            reason = "Browser app with URL/webpage text"
        elif app_category == "Video" and _has_any_keyword(text, CATEGORY_KEYWORDS["Video"]):
            confidence = 0.80
            reason = "Video app with video text"
        elif app_category == "Chat" and _has_any_keyword(text, CATEGORY_KEYWORDS["Chat"]):
            confidence = 0.80
            reason = "Chat app with chat text"

        return ClassificationResult(app_category, confidence, reason, app_name=app_name, source="app-hint")

    # 6) Visual fallback. Only Photos is safe enough visually; everything else should be Other.
    if profile.photo_score >= 0.68:
        return ClassificationResult(
            category="Photos",
            confidence=0.72,
            reason=f"Photo-like visual structure ({profile.photo_score:.2f})",
            app_name=app_name,
            source="visual",
        )

    return ClassificationResult(category="Other", confidence=0.0, reason="No confident content match", app_name=app_name, source="fallback")


def classify_app_folder(config: Dict[str, Any]) -> ClassificationResult:
    app_name = get_frontmost_app_name() or "Unknown App"
    folder = normalize_app_folder(app_name)
    if folder == "Unknown App":
        return ClassificationResult(category=folder, confidence=0.0, reason="Could not detect frontmost app", app_name=app_name, source="app")
    return ClassificationResult(category=folder, confidence=0.9, reason=f"Detected frontmost app: {app_name}", app_name=app_name, source="app")


def normalize_app_folder(app_name: str) -> str:
    clean = app_name.strip()
    if not clean or clean.lower() in {"unknown", "unknown app"}:
        return "Unknown App"
    lowered = clean.lower()
    return APP_FOLDER_ALIASES.get(lowered, clean)


def get_frontmost_app_name() -> Optional[str]:
    system = platform.system().lower()
    if system == "darwin":
        return _mac_frontmost_app()
    if system == "windows":
        return _windows_frontmost_app()
    return _linux_frontmost_app()


def get_frontmost_window_title() -> Optional[str]:
    system = platform.system().lower()
    if system == "darwin":
        return _mac_frontmost_window_title()
    if system == "windows":
        return _windows_frontmost_window_title()
    return None


def _mac_frontmost_app() -> Optional[str]:
    script = 'tell application "System Events" to get name of first application process whose frontmost is true'
    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    value = completed.stdout.strip()
    if completed.returncode == 0 and value:
        return value
    return None


def _mac_frontmost_window_title() -> Optional[str]:
    script = (
        'tell application "System Events" to tell (first application process whose frontmost is true) '
        'to if exists front window then get name of front window'
    )
    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    value = completed.stdout.strip()
    if completed.returncode == 0 and value:
        return value
    return None


def _windows_frontmost_app() -> Optional[str]:
    """Return the active Windows process name using only the standard library.

    This keeps CleanShot cross-platform without requiring pywin32.
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return None

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        PROCESS_VM_READ = 0x0010
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid.value)
        if not handle:
            return None

        try:
            buffer = ctypes.create_unicode_buffer(1024)
            if psapi.GetModuleBaseNameW(handle, None, buffer, len(buffer)):
                name = buffer.value.strip()
                if name.lower().endswith(".exe"):
                    name = name[:-4]
                return normalize_app_folder(name)
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return None
    return None


def _windows_frontmost_window_title() -> Optional[str]:
    try:
        import ctypes
    except Exception:
        return None
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return None
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value.strip() or None
    except Exception:
        return None

def _linux_frontmost_app() -> Optional[str]:
    if not shutil.which("xdotool"):
        return None
    try:
        window_id = subprocess.check_output(["xdotool", "getactivewindow"], text=True, timeout=1.0).strip()
        if not window_id:
            return None
        name = subprocess.check_output(["xdotool", "getwindowname", window_id], text=True, timeout=1.0).strip()
        return name or None
    except (OSError, subprocess.SubprocessError):
        return None


def _category_from_app(app_name: str) -> Optional[str]:
    lowered = app_name.lower().strip()
    if not lowered or lowered == "unknown app":
        return None

    for category, apps in CATEGORY_APPS.items():
        for app in apps:
            app_lower = str(app).lower()
            if lowered == app_lower or app_lower in lowered:
                return category
    return None


def _classify_photo_like(text: str, profile: ImageProfile, app_name: str) -> Optional[ClassificationResult]:
    app_category = _category_from_app(app_name)
    has_photo_words = _has_any_keyword(text, CATEGORY_KEYWORDS["Photos"])
    has_calendar_weather = _has_any_keyword(text, CALENDAR_WEATHER_WORDS)

    # Example: iOS widgets / calendar / weather over a big personal photo card.
    if profile.photo_score >= 0.62 and (has_photo_words or has_calendar_weather):
        return ClassificationResult(
            category="Photos",
            confidence=0.86,
            reason=f"Photo-like image with personal/travel/widget text ({profile.photo_score:.2f})",
            app_name=app_name,
            source="visual+text",
        )

    if profile.photo_score >= 0.78:
        return ClassificationResult(
            category="Photos",
            confidence=0.80,
            reason=f"Strong photo-like visual structure ({profile.photo_score:.2f})",
            app_name=app_name,
            source="visual",
        )

    if app_category == "Photos" and (profile.photo_score >= 0.42 or has_photo_words):
        return ClassificationResult(
            category="Photos",
            confidence=0.82,
            reason="Photos app plus photo/personal signals",
            app_name=app_name,
            source="app+visual",
        )

    return None


def _classify_code(text: str, profile: ImageProfile, app_name: str) -> Optional[ClassificationResult]:
    app_category = _category_from_app(app_name)
    strong_word_count = sum(1 for word in CODE_STRONG_WORDS if _contains_phrase(text, word))
    tech_word_count = sum(1 for word in CODE_TECH_WORDS if _contains_phrase(text, word))
    syntax_count = sum(1 for pattern in CODE_SYNTAX_PATTERNS if re.search(pattern, text, flags=re.IGNORECASE))

    code_score = 0.0
    reasons: List[str] = []

    if strong_word_count:
        code_score += min(0.35, strong_word_count * 0.10)
        reasons.append(f"{strong_word_count} code word(s)")
    if tech_word_count:
        code_score += min(0.22, tech_word_count * 0.07)
        reasons.append(f"{tech_word_count} tech word(s)")
    if syntax_count:
        code_score += min(0.36, syntax_count * 0.12)
        reasons.append(f"{syntax_count} syntax signal(s)")
    if app_category == "Code":
        code_score += 0.24
        reasons.append(f"code app: {app_name}")
    if profile.text_ui_score >= 0.62 and profile.photo_score < 0.50:
        code_score += 0.12
        reasons.append("text-heavy UI")

    # Guardrail: a colorful/photo-like screenshot should not become Code from weak OCR.
    if profile.photo_score >= 0.60 and syntax_count < 2 and strong_word_count < 2:
        return None

    # Guardrail: one OCR mistake like Clear -> class is not enough.
    strict_signal_count = syntax_count + strong_word_count
    if strict_signal_count < 2 and not (app_category == "Code" and (syntax_count >= 1 or strong_word_count >= 1 or tech_word_count >= 2)):
        return None

    if code_score >= 0.72:
        return ClassificationResult(
            category="Code",
            confidence=min(0.94, code_score),
            reason=", ".join(reasons) or "Code signals",
            app_name=app_name,
            source="strict-code",
        )

    return None



def _classify_visual_code(path: Path, profile: ImageProfile, app_name: str) -> Optional[ClassificationResult]:
    """Classify code/editor screenshots using local computer vision.

    VisionCore v2 catches both full IDE screenshots and small cropped code snippets.
    It does not depend on OCR for the Code decision, but it will use OCR when
    available to pick a smarter subfolder such as Code/JavaScript.
    """

    evidence = _code_vision_profile(path, profile)
    app_category = _category_from_app(app_name)

    score = evidence.score
    reasons = list(evidence.reason)

    if app_category == "Code":
        score += 0.12
        reasons.append(f"code app: {app_name}")

    # Small snippets often have only 3-6 active rows, but can still be obvious
    # code because of braces, syntax colors, indentation columns and editor canvas.
    if (
        3 <= evidence.row_count < 7
        and evidence.editor_canvas_score >= 0.55
        and evidence.syntax_color_score >= 0.45
        and evidence.mono_column_score >= 0.45
        and profile.photo_score < 0.30
    ):
        score += 0.14
        reasons.append("small code snippet pattern")

    # Guardrail: a true photo should almost never become Code from layout alone.
    if profile.photo_score >= 0.52 and evidence.score < 0.86:
        return None

    # Strong visual code evidence should work in Specific mode, not only Smart mode.
    if score >= 0.72 and _is_credible_visual_code(evidence, profile):
        detected_text = _read_text_with_optional_ocr(path, {"enable_ocr": True})
        confidence = min(0.96, max(0.78, score))
        return ClassificationResult(
            category="Code",
            subcategory=_visual_code_subcategory(path, app_name, detected_text),
            confidence=confidence,
            reason="VisionCore v2 code layout: " + ", ".join(reasons),
            app_name=app_name,
            source="vision-code-v2",
            signals=(f"codevision:{round(evidence.score * 100)}", f"rows:{evidence.row_count}"),
        )

    return None


def _code_vision_profile(path: Path, profile: ImageProfile) -> CodeVisionProfile:
    """Return visual code evidence from pixels only.

    The algorithm is intentionally dependency-light: Pillow only. It looks at:
    - repeated horizontal text bands
    - possible left line-number gutter
    - high contrast dark/light editor canvas
    - low photo/color texture
    - syntax-highlighting colored pixels
    - vertical projection peaks that look like monospaced columns
    """

    try:
        from PIL import Image, ImageFilter, ImageStat
    except ImportError:
        return CodeVisionProfile()

    try:
        with Image.open(path) as source_image:
            image = source_image.convert("RGB")
            width, height = image.size
            if width <= 0 or height <= 0:
                return CodeVisionProfile()

            max_width = 720
            if width > max_width:
                new_height = max(1, round(height * max_width / width))
                image = image.resize((max_width, new_height))

            width, height = image.size
            gray = image.convert("L")
            stat = ImageStat.Stat(gray)
            mean_luma = float(stat.mean[0])
            std_luma = float(stat.stddev[0])

            # Edge mask works for both light and dark themes.
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_values = list(edges.getdata())
            gray_values = list(gray.getdata())
            rgb_values = list(image.getdata())
    except Exception:
        return CodeVisionProfile()

    if not edge_values or not gray_values or not rgb_values:
        return CodeVisionProfile()

    edge_threshold = 34 if std_luma < 45 else 42
    edge_mask = [value > edge_threshold for value in edge_values]

    # Ink mask catches white-on-dark and black-on-light text.
    if mean_luma < 110:
        ink_threshold = min(245, mean_luma + max(34, std_luma * 0.48))
        ink_mask = [value > ink_threshold for value in gray_values]
        editor_canvas_score = _scale(130 - mean_luma, low=20, high=95)
    else:
        ink_threshold = max(0, mean_luma - max(35, std_luma * 0.55))
        ink_mask = [value < ink_threshold for value in gray_values]
        # A white webpage with text is not automatically a code editor.
        editor_canvas_score = _scale(std_luma, low=26, high=65) * 0.35

    # Combine masks: text is usually either edge-heavy or ink-heavy.
    combined_rows: List[bool] = []
    row_fractions: List[float] = []
    left_fractions: List[float] = []
    body_fractions: List[float] = []

    left_end = max(10, int(width * 0.18))
    body_start = max(left_end + 1, int(width * 0.20))

    for y in range(height):
        start = y * width
        end = start + width
        row_edge = edge_mask[start:end]
        row_ink = ink_mask[start:end]
        combined = [a or b for a, b in zip(row_edge, row_ink)]
        frac = sum(1 for item in combined if item) / width
        left = combined[:left_end]
        body = combined[body_start:]
        left_frac = sum(1 for item in left if item) / max(len(left), 1)
        body_frac = sum(1 for item in body if item) / max(len(body), 1)
        active = frac > 0.010 and body_frac > 0.004
        combined_rows.append(active)
        row_fractions.append(frac)
        left_fractions.append(left_frac)
        body_fractions.append(body_frac)

    bands: List[Tuple[int, int]] = []
    start_band: Optional[int] = None
    for index, active in enumerate(combined_rows):
        if active and start_band is None:
            start_band = index
        elif not active and start_band is not None:
            if index - start_band >= 2:
                bands.append((start_band, index - 1))
            start_band = None
    if start_band is not None and height - start_band >= 2:
        bands.append((start_band, height - 1))

    # Merge tiny gaps common in antialiased text rows.
    merged: List[Tuple[int, int]] = []
    for band in bands:
        if merged and band[0] - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)
    bands = merged

    row_count = len(bands)
    active_row_ratio = sum(1 for item in combined_rows if item) / max(height, 1)
    line_score = min(1.0, row_count / 11.0) * 0.72 + min(1.0, active_row_ratio * 3.2) * 0.28

    gutter_hits = 0
    body_hits = 0
    for y1, y2 in bands:
        mid = (y1 + y2) // 2
        if 0 <= mid < height:
            if left_fractions[mid] > 0.006:
                gutter_hits += 1
            if body_fractions[mid] > 0.010:
                body_hits += 1
    gutter_ratio = gutter_hits / max(row_count, 1)
    body_ratio = body_hits / max(row_count, 1)
    gutter_score = min(1.0, gutter_ratio * 1.25) * min(1.0, body_ratio * 1.10)

    # High contrast: code/editor screenshots are usually crisp.
    contrast_score = min(1.0, _scale(std_luma, low=22, high=74) * 0.75 + _scale(profile.edge_density, low=0.075, high=0.22) * 0.25)

    # Syntax colors: colored text pixels on a low-color canvas are a strong editor hint.
    bright_or_dark_text = 0
    saturated_text = 0
    for (r, g, b), is_ink in zip(rgb_values, ink_mask):
        if not is_ink:
            continue
        bright_or_dark_text += 1
        mx = max(r, g, b)
        mn = min(r, g, b)
        if mx > 0 and (mx - mn) / mx > 0.28:
            saturated_text += 1
    syntax_color_score = min(1.0, (saturated_text / max(bright_or_dark_text, 1)) * 7.5)

    # Monospace-ish vertical peaks. We do not need perfect OCR; just regular columns.
    col_counts: List[int] = []
    for x in range(width):
        count = 0
        for y in range(height):
            idx = y * width + x
            if edge_mask[idx] or ink_mask[idx]:
                count += 1
        col_counts.append(count)
    peak_threshold = max(2, int(height * 0.035))
    peaks = sum(1 for count in col_counts[body_start:] if count >= peak_threshold)
    mono_column_score = min(1.0, peaks / max(width * 0.22, 1))

    photo_penalty = profile.photo_score * 0.40
    low_color_bonus = (1.0 - _scale(profile.colorfulness, low=10, high=54)) * 0.08

    score = 0.0
    score += line_score * 0.34
    score += gutter_score * 0.20
    score += editor_canvas_score * 0.17
    score += contrast_score * 0.13
    score += syntax_color_score * 0.08
    score += mono_column_score * 0.08
    score += low_color_bonus
    score -= photo_penalty
    score = min(1.0, max(0.0, score))

    reasons: List[str] = []
    if row_count >= 8:
        reasons.append(f"{row_count} text rows")
    if gutter_score >= 0.35:
        reasons.append("line-number/gutter pattern")
    if editor_canvas_score >= 0.60:
        reasons.append("editor-like canvas")
    if syntax_color_score >= 0.18:
        reasons.append("syntax-color pixels")
    if mono_column_score >= 0.22:
        reasons.append("monospace column rhythm")
    if not reasons:
        reasons.append(f"visual score {score:.2f}")

    return CodeVisionProfile(
        score=score,
        line_score=line_score,
        gutter_score=gutter_score,
        contrast_score=contrast_score,
        editor_canvas_score=editor_canvas_score,
        syntax_color_score=syntax_color_score,
        mono_column_score=mono_column_score,
        row_count=row_count,
        active_row_ratio=active_row_ratio,
        reason=tuple(reasons),
    )



def _is_credible_visual_code(evidence: CodeVisionProfile, profile: ImageProfile) -> bool:
    """Guardrail for visual code detection.

    Long plain UI lists can also have rows and columns. A code/editor decision needs
    either an editor-like canvas or real contrast, plus repeated rows. v2 also
    supports small snippets when the code signals are very strong.
    """

    if profile.photo_score >= 0.52 and evidence.score < 0.86:
        return False

    has_editor_surface = evidence.editor_canvas_score >= 0.42 or evidence.contrast_score >= 0.34
    has_code_structure = (
        evidence.gutter_score >= 0.28
        or evidence.syntax_color_score >= 0.22
        or evidence.mono_column_score >= 0.25
    )

    if evidence.row_count >= 7:
        return has_editor_surface and has_code_structure

    # Cropped snippets: fewer rows, but high-confidence syntax coloring + monospace
    # rhythm + editor canvas is enough to classify as Code.
    if 3 <= evidence.row_count < 7:
        return (
            evidence.editor_canvas_score >= 0.55
            and evidence.syntax_color_score >= 0.45
            and evidence.mono_column_score >= 0.45
            and evidence.active_row_ratio >= 0.05
            and profile.photo_score < 0.30
        )

    return False

def _visual_code_subcategory(path: Path, app_name: str, detected_text: str = "") -> str:
    language, confidence, _reason = detect_code_language(detected_text, filename=path.name, app_name=app_name)
    if confidence >= 0.58 and language != "Projects":
        return language
    text = _normalized_text(" ".join([path.stem, app_name, detected_text]))
    if _has_any_keyword(text, ["terminal", "iterm", "windows terminal", "powershell", "bash", "zsh", "shell", "chmod", "sudo", "brew", "pip install"]):
        return "Terminal"
    if _has_any_keyword(text, ["github", "pull request", "commit", "branch", "repository"]):
        return "GitHub"
    return language or "Projects"

def _classify_specific_text(text: str, profile: ImageProfile, app_name: str) -> Optional[ClassificationResult]:
    scores: Dict[str, float] = {}
    reasons: Dict[str, List[str]] = {}

    def add(category: str, amount: float, reason: str) -> None:
        scores[category] = scores.get(category, 0.0) + amount
        reasons.setdefault(category, []).append(reason)

    app_category = _category_from_app(app_name)
    if app_category and app_category != "Code":
        add(app_category, 0.25, f"app hint: {app_name}")

    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "Photos":
            # Photos is handled earlier with visual signals. Text-only photo words are weaker.
            weight = 0.12
        else:
            weight = 0.16

        matches = [word for word in keywords if _contains_phrase(text, word)]
        if matches:
            add(category, min(0.52, len(matches) * weight), f"matched: {', '.join(matches[:4])}")

    if _has_url_signal(text):
        add("Browser", 0.45, "URL/domain signal")

    if profile.photo_score >= 0.55:
        add("Photos", 0.40, f"photo-like image ({profile.photo_score:.2f})")

    # UI/content shape helpers.
    if profile.ratio >= 1.65 and profile.photo_score >= 0.45:
        add("Video", 0.18, "wide visual layout")
    if profile.text_ui_score >= 0.70 and profile.photo_score < 0.45:
        add("Documents", 0.16, "text-heavy non-photo image")

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_category, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Convert score to confidence but require a margin. Better Other than wrong.
    confidence = min(0.90, 0.46 + best_score)
    margin = best_score - second_score

    if best_score < 0.30 or confidence < 0.68 or margin < 0.10:
        return None

    return ClassificationResult(
        category=best_category,
        confidence=confidence,
        reason="; ".join(reasons.get(best_category, [])),
        app_name=app_name,
        source="content",
    )


def _image_profile(path: Path) -> ImageProfile:
    try:
        from PIL import Image, ImageFilter, ImageStat
    except ImportError:
        return ImageProfile()

    try:
        with Image.open(path) as image:
            width, height = image.size
            sample = image.convert("RGB").resize((160, 160))
            stat = ImageStat.Stat(sample)
            stddev = sum(stat.stddev) / max(len(stat.stddev), 1)
            ratio = width / max(height, 1)
            colorfulness = _colorfulness(sample)
            edge_density = _edge_density(sample, ImageFilter)
            layout_text_score = _layout_text_score(image, ImageFilter)
            mean_luma = sum(ImageStat.Stat(sample.convert("L")).mean) / 1
            dark_score = 1.0 - _scale(mean_luma, low=35, high=165)
    except Exception:
        return ImageProfile()

    # Photo-like: colorful + varied. UI/code tends to be low colorfulness or mostly flat regions.
    photo_score = 0.0
    photo_score += _scale(stddev, low=28, high=72) * 0.42
    photo_score += _scale(colorfulness, low=18, high=62) * 0.48
    photo_score += _scale(edge_density, low=0.05, high=0.20) * 0.10
    photo_score = min(1.0, max(0.0, photo_score))

    # Text/UI-like: edges without much color variation. Useful as a weak signal only.
    text_ui_score = 0.0
    text_ui_score += _scale(edge_density, low=0.08, high=0.28) * 0.55
    text_ui_score += (1.0 - _scale(colorfulness, low=12, high=55)) * 0.25
    text_ui_score += (1.0 - _scale(stddev, low=30, high=72)) * 0.20
    text_ui_score = min(1.0, max(0.0, text_ui_score))

    return ImageProfile(
        width=width,
        height=height,
        ratio=ratio,
        stddev=stddev,
        colorfulness=colorfulness,
        edge_density=edge_density,
        photo_score=photo_score,
        text_ui_score=text_ui_score,
        layout_text_score=layout_text_score,
        dark_score=dark_score,
    )


def _layout_text_score(image: Any, image_filter_module: Any) -> float:
    gray = image.convert("L")
    width, height = gray.size
    max_width = 420
    if width > max_width:
        new_height = max(1, round(height * max_width / width))
        gray = gray.resize((max_width, new_height))
    edges = gray.filter(image_filter_module.FIND_EDGES)
    width, height = edges.size
    values = list(edges.getdata())
    if not values or width <= 0 or height <= 0:
        return 0.0

    bands = 0
    band_length = 0
    active_rows = 0
    for y in range(height):
        row = values[y * width:(y + 1) * width]
        edge_fraction = sum(1 for value in row if value > 35) / width
        active = edge_fraction > 0.018
        if active:
            active_rows += 1
            band_length += 1
        elif band_length:
            if band_length >= 1:
                bands += 1
            band_length = 0
    if band_length:
        bands += 1

    band_score = min(1.0, bands / 12)
    row_score = min(1.0, (active_rows / max(height, 1)) * 3)
    return min(1.0, band_score * 0.65 + row_score * 0.35)


def _colorfulness(image: Any) -> float:
    pixels = list(image.getdata())
    if not pixels:
        return 0.0
    rg = [r - g for r, g, _b in pixels]
    yb = [0.5 * (r + g) - b for r, g, b in pixels]
    rg_mean, rg_std = _mean_std(rg)
    yb_mean, yb_std = _mean_std(yb)
    return math.sqrt(rg_std**2 + yb_std**2) + 0.3 * math.sqrt(rg_mean**2 + yb_mean**2)


def _edge_density(image: Any, image_filter_module: Any) -> float:
    gray = image.convert("L")
    edges = gray.filter(image_filter_module.FIND_EDGES)
    values = list(edges.getdata())
    if not values:
        return 0.0
    return sum(1 for value in values if value > 35) / len(values)


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return min(1.0, max(0.0, (value - low) / (high - low)))


def _has_url_signal(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in URL_PATTERNS)


def _has_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    return any(_contains_phrase(text, keyword) for keyword in keywords)


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = _normalized_text(phrase)
    if not phrase:
        return False

    # Phrase with punctuation/symbols: use escaped relaxed whitespace.
    if re.search(r"[^a-z0-9\s]", phrase):
        pattern = re.escape(phrase).replace(r"\ ", r"\s+")
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    words = phrase.split()
    if not words:
        return False
    pattern = r"(?<![a-z0-9])" + r"\s+".join(re.escape(word) for word in words) + r"(?![a-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _read_text_with_optional_ocr(path: Path, config: Dict[str, Any]) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    try:
        with Image.open(path) as image:
            text = pytesseract.image_to_string(image, timeout=3)
    except Exception:
        return ""
    return _normalized_text(text)


def _normalized_text(value: str) -> str:
    text = str(value).replace("_", " ").replace("-", " ").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------
# CleanShot HyperSort / Smart Mode
# -------------------------

def classify_smart_screenshot(path: Path, config: Dict[str, Any]) -> ClassificationResult:
    """Hybrid local classifier with visual memory, content rules, app context and review fallback.

    This is intentionally local-first. It does not upload screenshots. It combines:
    1) ScreenshotDNA visual memory for previously seen visual layouts.
    2) Strict content classifier.
    3) App + window title context.
    4) Session memory for screenshot bursts.
    5) Conservative fallback to _Review/Other when unsure.
    """
    from .brain import find_neural_match, find_recent_session, find_visual_match

    app_name = get_frontmost_app_name() or "Unknown App"
    window_title = get_frontmost_window_title() or ""
    filename_text = _normalized_text(path.stem)
    ocr_text = _read_text_with_optional_ocr(path, config) if config.get("enable_ocr", False) else ""
    full_text = " ".join(part for part in [filename_text, _normalized_text(window_title), ocr_text] if part).strip()
    profile = _image_profile(path)

    signals: List[str] = []

    if config.get("enable_neuro_learning", True):
        threshold = float(config.get("neuro_similarity_threshold", 0.86))
        neural = find_neural_match(path, threshold=threshold)
        if neural and neural.confidence >= threshold:
            return ClassificationResult(
                category=neural.category,
                subcategory=neural.subcategory,
                confidence=neural.confidence,
                reason=neural.reason or "NeuroVector learned visual match",
                app_name=app_name,
                source="neurovector",
                signals=(f"neuro:{round(neural.confidence * 100)}",),
            )

    if config.get("enable_visual_memory", True):
        distance = int(config.get("visual_similarity_distance", 8))
        match = find_visual_match(path, max_distance=distance)
        if match and match.confidence >= 0.78:
            return ClassificationResult(
                category=match.category,
                subcategory=match.subcategory,
                confidence=match.confidence,
                reason=match.reason or f"ScreenshotDNA memory match, distance {match.distance}",
                app_name=app_name,
                source="screenshot-dna",
                signals=(f"memory:{match.distance}",),
            )

    # Run the stricter v0.4 content engine, then add subcategory intelligence.
    content_result = classify_screenshot(path, {**config, "enable_ocr": bool(config.get("enable_ocr", False))})
    if content_result.confidence > 0:
        signals.append(f"content:{content_result.category}:{round(content_result.confidence * 100)}")

    app_category = _category_from_app(app_name)
    if app_category:
        signals.append(f"app:{app_category}")

    title_category = _category_from_text(full_text, profile)
    if title_category:
        signals.append(f"text:{title_category}")

    category_scores: Dict[str, float] = {}
    reasons: Dict[str, List[str]] = {}

    def add(category: str, amount: float, reason: str) -> None:
        if not category or category == "Other":
            return
        category_scores[category] = category_scores.get(category, 0.0) + amount
        reasons.setdefault(category, []).append(reason)

    if content_result.category != "Other" and content_result.confidence > 0:
        add(content_result.category, content_result.confidence, f"content engine: {content_result.reason}")

    if app_category:
        app_weight = 0.42 if app_category in {"Code", "Design", "Chat", "Games"} else 0.30
        # Browser apps are noisy; a Chrome screenshot might show a family photo website.
        if app_category == "Browser" and not _has_url_signal(full_text):
            app_weight = 0.18
        add(app_category, app_weight, f"frontmost app: {app_name}")

    if title_category:
        add(title_category, 0.36, "window title / OCR category hint")

    visual_category = _visual_category(path, profile, full_text)
    if visual_category:
        if visual_category == "Photos":
            visual_weight = 0.52
        elif visual_category == "Code":
            visual_weight = 0.42
        else:
            visual_weight = 0.24
        add(visual_category, visual_weight, f"visual profile: photo={profile.photo_score:.2f}, text={profile.text_ui_score:.2f}, layout={profile.layout_text_score:.2f}")

    # Agreement bonus: if two independent signals agree, trust it more.
    for category, score in list(category_scores.items()):
        evidence_count = len(reasons.get(category, []))
        if evidence_count >= 2:
            category_scores[category] = score + 0.12
            reasons[category].append("multi-signal agreement")

    if category_scores:
        ranked = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
        best_category, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        confidence = min(0.96, 0.36 + best_score)

        # Prevent Browser/App from beating a strong photo-like screenshot unless Browser has URL evidence.
        if best_category == "Browser" and profile.photo_score >= 0.65 and not _has_url_signal(full_text):
            confidence -= 0.22
            reasons[best_category].append("browser penalty: photo-like screenshot with no URL signal")

        # Prevent weak Code errors on visual/photo screenshots.
        if best_category == "Code" and profile.photo_score >= 0.55:
            confidence -= 0.18
            reasons[best_category].append("code penalty: image is photo-like")

        # If another category is very close, lower confidence and let review catch it.
        if margin < 0.16:
            confidence -= 0.12
            reasons[best_category].append("low category margin")

        if confidence >= 0.64:
            # Preserve the content engine's subfolder when it made a strong direct
            # decision. This matters for cropped code snippets where VisionCore v2
            # can identify JavaScript from OCR/punctuation, but the window title is
            # unknown.
            if content_result.category == best_category and getattr(content_result, "subcategory", ""):
                subcategory = content_result.subcategory
            else:
                subcategory = _smart_subcategory(best_category, full_text, app_name, window_title, profile, config)
            if not bool(config.get("smart_subfolders", True)):
                subcategory = ""
            return ClassificationResult(
                category=best_category,
                subcategory=subcategory,
                confidence=max(0.0, min(0.96, confidence)),
                reason="; ".join(reasons.get(best_category, [])),
                app_name=app_name,
                source="hypersort",
                signals=tuple(signals),
            )

    # Burst/session memory: good for taking several screenshots from the same app quickly.
    if config.get("enable_session_memory", True):
        session = find_recent_session(app_name, within_minutes=int(config.get("session_context_minutes", 4)))
        if session:
            return ClassificationResult(
                category=session.category,
                subcategory=session.subcategory,
                confidence=session.confidence,
                reason="Recent screenshot burst from same app",
                app_name=app_name,
                source="session-memory",
                signals=("session",),
            )

    review_folder = str(config.get("review_folder", "_Review")).strip() or "_Review"
    other_folder = str(config.get("other_folder", "Other")).strip() or "Other"
    fallback = review_folder if bool(config.get("review_low_confidence", True)) else other_folder
    return ClassificationResult(
        category=fallback,
        confidence=0.0,
        reason="No safe multi-signal decision; sent to review" if fallback == review_folder else "No safe multi-signal decision",
        app_name=app_name,
        source="review" if fallback == review_folder else "fallback",
        signals=tuple(signals),
    )


def _category_from_text(text: str, profile: ImageProfile) -> Optional[str]:
    if not text:
        return None
    if _has_url_signal(text):
        return "Browser"
    code = _classify_code(text, profile, "Unknown App")
    if code:
        return "Code"
    best: Optional[Tuple[str, int]] = None
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if _contains_phrase(text, keyword))
        if hits and (best is None or hits > best[1]):
            best = (category, hits)
    if best and best[1] >= 2:
        return best[0]
    return None


def _visual_category(path: Path, profile: ImageProfile, text: str) -> Optional[str]:
    if profile.photo_score >= 0.72:
        return "Photos"
    # OCR-free VisionCore code hint: editor/terminal/README code layouts.
    code_evidence = _code_vision_profile(path, profile)
    if code_evidence.score >= 0.64 and _is_credible_visual_code(code_evidence, profile):
        return "Code"
    if profile.text_ui_score >= 0.78 and profile.photo_score < 0.42:
        return "Documents"
    if profile.ratio >= 1.65 and profile.photo_score >= 0.52 and _has_any_keyword(text, CATEGORY_KEYWORDS["Video"]):
        return "Video"
    return None


def _smart_subcategory(
    category: str,
    text: str,
    app_name: str,
    window_title: str,
    profile: ImageProfile,
    config: Dict[str, Any],
) -> str:
    combined = _normalized_text(" ".join([text, app_name, window_title]))
    lowered_app = _normalized_text(app_name)

    if category == "Code":
        language, language_confidence, _language_reason = detect_code_language(combined, filename=window_title, app_name=app_name)
        if language_confidence >= 0.56 and language != "Projects":
            return language
        if _has_any_keyword(combined, ["github", "pull request", "commit", "git", "repository"]):
            return "GitHub"
        if _has_any_keyword(lowered_app, ["terminal", "iterm", "windows terminal", "powershell"]):
            return "Terminal"
        return language or "Projects"

    if category == "Browser":
        if _has_any_keyword(combined, ["github", "pull request", "repository"]):
            return "GitHub"
        if _has_any_keyword(combined, ["youtube", "netflix", "video"]):
            return "Video"
        if _has_any_keyword(combined, ["docs", "documentation", "stackoverflow", "stack overflow"]):
            return "Research"
        if _has_any_keyword(combined, ["google", "search"]):
            return "Search"
        return normalize_app_folder(app_name) if lowered_app not in {"unknown app", "unknown"} else "Web"

    if category == "Photos":
        if _has_any_keyword(combined, ["trip", "travel", "vacation", "beach", "hotel", "flight", "dec"]):
            return "Trips"
        if _has_any_keyword(combined, ["family", "birthday", "wedding", "portrait", "baby", "kids"]):
            return "Family"
        if profile.photo_score >= 0.78:
            return "Photo Library"
        return "Personal"

    if category == "Chat":
        folder = normalize_app_folder(app_name)
        return folder if folder != "Unknown App" else "Messages"

    if category == "Video":
        if _has_any_keyword(combined, ["youtube"]):
            return "YouTube"
        if _has_any_keyword(combined, ["netflix", "movie", "episode", "trailer"]):
            return "Movies"
        return "Clips"

    if category == "Design":
        if _has_any_keyword(combined, ["figma"]):
            return "Figma"
        if _has_any_keyword(combined, ["photoshop", "illustrator", "adobe"]):
            return "Adobe"
        return "Design Files"

    if category == "Documents":
        if _has_any_keyword(combined, ["invoice", "receipt"]):
            return "Receipts"
        if _has_any_keyword(combined, ["pdf", "preview", "acrobat"]):
            return "PDFs"
        if _has_any_keyword(combined, ["notion", "notes", "obsidian"]):
            return "Notes"
        return "Documents"

    if category == "Games":
        if _has_any_keyword(combined, ["fivem", "gta"]):
            return "FiveM"
        if _has_any_keyword(combined, ["minecraft"]):
            return "Minecraft"
        if _has_any_keyword(combined, ["roblox"]):
            return "Roblox"
        return "Games"

    return ""
