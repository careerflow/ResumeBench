"""Resume extraction schema module."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def _load_schema() -> dict[str, Any]:
    """Load the resume schema from the JSON file."""
    schema_path = files(__package__) / "resume.json"
    schema_text = schema_path.read_text(encoding="utf-8")

    return json.loads(schema_text)


RESUME_SCHEMA: dict[str, Any] = _load_schema()


def get_schema() -> dict[str, Any]:
    """Get the resume extraction schema."""
    return RESUME_SCHEMA.copy()
