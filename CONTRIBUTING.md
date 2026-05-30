# Contributing

Thanks for helping improve CleanShot HyperSort.

## Local setup

```bash
python3 -m pip install -r requirements.txt
python3 run.py
```

Windows:

```bat
py -m pip install -r requirements.txt
py run.py
```

## Test a screenshot

```bash
python3 tools/classify_image.py "/path/to/screenshot.png" --debug --ocr
```

## Pull requests

Good PRs include:

- a clear explanation of the change
- screenshots for UI changes
- before/after classifier examples for recognition changes

CleanShot should stay local-first. Do not add cloud uploads by default.
