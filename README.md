# CleanShot HyperSort

A clean, modern screenshot organizer for **macOS and Windows**.

CleanShot watches your screenshot folder, understands what the screenshot is, and files it automatically. It runs locally, has a polished PySide6 desktop UI, and can learn from your corrections.

![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![License](https://img.shields.io/badge/license-MIT-black)
![Local First](https://img.shields.io/badge/privacy-local--first-purple)

---

## Why CleanShot?

Screenshots pile up fast. CleanShot turns a messy Desktop into organized folders like this:

```txt
CleanShot/
  Code/
    JavaScript/
    Python/
    TypeScript/
    PHP/
    HTML/
    CSS/
    SQL/
    Shell/
    Rust/
    Go/
    Java/
    C++/
    Markdown/
  Browser/
    GitHub/
    Research/
    Search/
  Photos/
    Family/
    Trips/
    Photo Library/
  Documents/
    PDFs/
    Receipts/
    Notes/
  Chat/
    Discord/
    Slack/
  _Review/
```

If CleanShot is not confident, it sends the screenshot to `_Review` instead of guessing wrong.

---

## Features

- **Smart Mode** with local visual recognition
- **App Mode** for folders per active app
- **Day Mode** for folders per date
- **Teach Mode** so the app learns from your corrections
- **LanguageMind** code-language detection
- **VisionCore** visual code/editor detection without OCR
- **ScreenshotDNA** near-duplicate visual memory
- **NeuroVector** local learning vectors from examples you teach
- **OCR support** with Tesseract / pytesseract
- **macOS and Windows app detection**
- **Duplicate detection**
- **Move or copy mode**
- **Clean activity log** with confidence, destination, and reason
- **No Electron**
- **No cloud upload**
- **No account**

---

## Organization modes

### Smart Mode

Best mode. Smart Mode combines:

- image structure
- OCR text when enabled
- active app detection
- active window title when available
- code/editor visual patterns
- language fingerprints
- session memory
- examples you taught it
- review fallback

Example outputs:

```txt
Code/JavaScript
Code/Python
Photos/Family
Browser/GitHub
Documents/PDFs
_Review
```

### App Mode

Sorts by the active app when the screenshot was taken:

```txt
Chrome/
VS Code/
Discord/
Safari/
Unknown App/
```

OS detection is automatic:

- macOS uses AppleScript/System Events
- Windows uses standard-library Win32 calls through `ctypes`
- unsupported systems gracefully fall back to `Unknown App`

### Day Mode

Sorts by date:

```txt
2026-05-30/
2026-05-31/
```

**Specific Mode was removed** because Smart Mode does the same job better.

---

## Smart recognition tech

CleanShot uses several local engines together:

### VisionCore

OCR-free visual analysis. It detects things like:

- editor-like dark or light canvases
- code row rhythm
- line-number gutters
- syntax-colored pixels
- terminal/editor layouts
- photo-like vs UI-like image structure

### LanguageMind

Recognizes common coding languages from OCR, filename, title, and syntax fingerprints.

Supported examples include:

```txt
JavaScript, TypeScript, Python, PHP, HTML, CSS, SQL, JSON, Shell,
C, C++, C#, Java, Go, Rust, Swift, Kotlin, Ruby, Lua, Dart, R,
Markdown, YAML, XML, Docker
```

### ScreenshotDNA

A compact visual hash for near-duplicate and similar screenshot layouts.

### NeuroVector

A local learning vector created for CleanShot. It lets the app learn from your folders and future corrections.

### Teach Mode

You can correct CleanShot and it learns locally.

Example:

```txt
Correct category: Code
Correct subfolder: JavaScript
```

Learning data is stored here:

```txt
~/.cleanshot/brain.json
```

Nothing is uploaded.

---

## Install and run

### macOS

```bash
python3 -m pip install -r requirements.txt
python3 run.py
```

Or double-click:

```txt
start.command
```

### Windows

```bat
py -m pip install -r requirements.txt
py run.py
```

Or double-click:

```txt
start.bat
```

---

## Optional OCR

CleanShot works without OCR, but OCR improves Smart Mode for screenshots with readable text.

### macOS

Install Tesseract:

```bash
brew install tesseract
python3 -m pip install pytesseract
```

Optional Apple Vision packages:

```bash
python3 -m pip install -r requirements-ocr-macos.txt
```

Then enable OCR in CleanShot Settings.

### Windows

Install Tesseract for Windows, then run:

```bat
py -m pip install pytesseract
```

Then enable OCR in CleanShot Settings.

---

## Test classification manually

macOS:

```bash
python3 tools/classify_image.py "/path/to/screenshot.png" --debug --ocr
```

Windows:

```bat
py tools\classify_image.py "C:\path\to\screenshot.png" --debug --ocr
```

This prints:

- category
- subcategory
- confidence
- classifier source
- active app
- reason
- VisionCore debug values
- LanguageMind result

---

## Teach CleanShot

### Teach one screenshot

Open the **Teach** tab:

1. Choose the screenshot that was sorted wrong.
2. Enter the correct category.
3. Enter an optional subcategory.
4. Click **Teach CleanShot**.

Example:

```txt
Category: Code
Subcategory: JavaScript
```

### Learn from folders

Manually move screenshots into the right folders, then click **Learn From Folders**.

Example:

```txt
CleanShot/Code/JavaScript/example.png
CleanShot/Photos/Family/photo.png
CleanShot/Documents/Receipts/receipt.png
```

CleanShot reads those folder names and learns the pattern.

---

## Settings location

CleanShot stores settings and learning data in:

```txt
~/.cleanshot/
```

Important files:

```txt
~/.cleanshot/config.json
~/.cleanshot/brain.json
~/.cleanshot/activity.jsonl
~/.cleanshot/hashes.json
```

---

## Start CleanShot when your computer starts

### macOS startup

For development mode, create a LaunchAgent.

First, find your CleanShot folder path:

```bash
pwd
```

Then create this file:

```bash
nano ~/Library/LaunchAgents/com.cleanshot.hypersort.plist
```

Paste this and replace `/ABSOLUTE/PATH/TO/CleanShot_HyperSort` with your real folder path:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.cleanshot.hypersort</string>

    <key>ProgramArguments</key>
    <array>
      <string>/usr/bin/python3</string>
      <string>/ABSOLUTE/PATH/TO/CleanShot_HyperSort/run.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/ABSOLUTE/PATH/TO/CleanShot_HyperSort</string>

    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/tmp/cleanshot.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/cleanshot-error.log</string>
  </dict>
</plist>
```

Enable it:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cleanshot.hypersort.plist
launchctl enable gui/$(id -u)/com.cleanshot.hypersort
launchctl kickstart gui/$(id -u)/com.cleanshot.hypersort
```

Disable it:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.cleanshot.hypersort.plist
rm ~/Library/LaunchAgents/com.cleanshot.hypersort.plist
```

### Windows startup

For development mode, add CleanShot to the current-user startup registry key.

Replace the paths with your real Python and CleanShot paths:

```bat
reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v CleanShotHyperSort /t REG_SZ /d "\"C:\Path\To\pythonw.exe\" \"C:\Path\To\CleanShot_HyperSort\run.py\"" /f
```

Use `pythonw.exe`, not `python.exe`, so it does not open a terminal window.

Disable it:

```bat
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Run" /v CleanShotHyperSort /f
```

After packaging into a real `.app` or `.exe`, point startup to that app instead of `run.py`.

---

## Package for release

### macOS `.app`

```bash
python3 -m pip install pyinstaller
pyinstaller --windowed --name CleanShot run.py
```

Result:

```txt
dist/CleanShot.app
```

### Windows `.exe`

```bat
py -m pip install pyinstaller
pyinstaller --windowed --name CleanShot run.py
```

Result:

```txt
dist\CleanShot\CleanShot.exe
```

---

## Publish to GitHub

```bash
git init
git add .
git commit -m "Initial release of CleanShot HyperSort"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cleanshot-hypersort.git
git push -u origin main
```

Recommended repository description:

```txt
A local-first smart screenshot organizer for macOS and Windows with teachable recognition.
```

Recommended topics:

```txt
screenshots, organizer, desktop-app, pyside6, python, macos, windows, productivity, ocr, local-first
```

---

## Privacy

CleanShot is local-first:

- no screenshots are uploaded
- no account is required
- learning data stays in `~/.cleanshot/brain.json`
- OCR runs locally when enabled

---

## Official Source

CleanShot HyperSort is developed and maintained by Rafael H.

Official GitHub repository:

https://github.com/rafaelHai/cleanshot-hypersort

CleanShot is free and open-source software.

If you obtained CleanShot from a third-party website, marketplace, or reseller, verify that it matches the official repository before installing it.

The author does not sell CleanShot through third-party marketplaces.

---

## License

MIT License

Copyright (c) 2026 Rafael H

See the LICENSE file for details.
