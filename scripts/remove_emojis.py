"""Replace emojis and non-ASCII symbols in project files with ASCII equivalents."""
from __future__ import annotations

import pathlib
from typing import Iterable


REPLACEMENTS = {
    "\u26a0\ufe0f": "WARNING",
    "\u26a0": "WARNING",
    "\u274c": "ERROR",
    "\u2705": "",
    "\U0001f4c2": "",
    "\U0001f4c6": "",
    "\u23f3": "",
    "\U0001f4be": "",
    "\U0001f3af": "",
    "\U0001f389": "",
    "\U0001f324\ufe0f": "",
    "\U0001f9fe": "",
    "\U0001f4ca": "",
    "\U0001f4c8": "",
    "\U0001f4e6": "",
    "\U0001f525": "",
    "\U0001f32b\ufe0f": "",
    "\U0001f9e0": "",
    "\U0001f916": "",
    "\U0001f52e": "",
    "\U0001f680": "",
    "\U0001f4c5": "",
    "\U0001f4cc": "",
    "\U0001f504": "",
    "\U0001f4c9": "",
    "\U0001f4d5": "",
    "\u2194": "<->",
    "\u2192": "->",
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
}

CHAR_MAP = {
    ord("\u00b0"): "deg",
    ord("\u00b5"): "u",
    ord("\u00b3"): "^3",
    ord("\u00b2"): "^2",
}


def sanitize_text(text: str) -> str:
    """Return text with emojis removed and symbols converted to ASCII."""
    for target, replacement in REPLACEMENTS.items():
        text = text.replace(target, replacement)
    result_chars = []
    for char in text:
        code = ord(char)
        if code < 128:
            result_chars.append(char)
            continue
        mapped = CHAR_MAP.get(code)
        if mapped is not None:
            result_chars.append(mapped)
    return "".join(result_chars)


def iter_python_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield project python files excluding virtual environments."""
    for path in root.rglob("*.py"):
        if any(part.startswith(".venv") for part in path.parts):
            continue
        yield path


def process_file(path: pathlib.Path) -> None:
    """Read file, sanitize content, and overwrite if any change."""
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    sanitized = sanitize_text(original)
    if original != sanitized:
        path.write_text(sanitized, encoding="utf-8")


def main() -> None:
    """Traverse repository and sanitize python files."""
    root = pathlib.Path(__file__).resolve().parents[1]
    for file_path in iter_python_files(root):
        process_file(file_path)


if __name__ == "__main__":
    main()

