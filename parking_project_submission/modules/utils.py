"""Utility helpers for the submission package.

These will host logging, argument validation, and other cross-cutting helpers
in later batches. For now they provide simple stubs so imports can be wired.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path) -> Dict[str, Any]:
    """Read a UTF-8 encoded JSON file into a dictionary."""

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, content: Dict[str, Any]) -> None:
    """Serialize a dictionary to JSON with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
