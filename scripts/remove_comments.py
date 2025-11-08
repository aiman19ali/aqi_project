"""Remove Python comments from project files."""
from __future__ import annotations

import io
import os
import pathlib
import tokenize


def strip_comments_from_text(text: str) -> str:
    """Return text with all comment tokens removed."""
    buffer = io.StringIO(text)
    tokens = tokenize.generate_tokens(buffer.readline)
    cleaned = []
    for token_info in tokens:
        if token_info.type == tokenize.COMMENT:
            continue
        cleaned.append(token_info)
    reconstructed = tokenize.untokenize(cleaned)
    lines = reconstructed.splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip() == "\\":
            cleaned_lines.append("")
        else:
            cleaned_lines.append(line)
    ending = "\n" if reconstructed.endswith("\n") else ""
    return "\n".join(cleaned_lines) + ending
def process_file(path: pathlib.Path) -> None:
    """Strip comments from a single file."""
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    updated = strip_comments_from_text(original)
    if original != updated:
        path.write_text(updated, encoding="utf-8")
def main() -> None:
    """Walk repository and remove comments from Python files."""
    root = pathlib.Path(__file__).resolve().parents[1]
    for file_path in root.rglob("*.py"):
        if any(part.startswith(".venv") for part in file_path.parts):
            continue
        process_file(file_path)
if __name__ == "__main__":
    main()
