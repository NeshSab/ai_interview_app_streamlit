"""Utilities for robustly extracting JSON from LLM responses."""

from __future__ import annotations
import json
import re
from typing import Any

_JSON_BLOCK = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Remove surrounding ```...``` fences (with or without 'json') if present."""
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", t, flags=re.DOTALL)
        t = re.sub(r"\s*```$", "", t, flags=re.DOTALL)
    return t.strip()


def extract_json(text: str) -> Any:
    """
    Extract and parse the first JSON object/array from an LLM response.
    - Safely handles code fences and leading/trailing prose.
    - Returns dict / list on success, or {} / [] on failure (keeping the type intuitive).
    """
    if not text:
        return {}
    t = _strip_code_fences(text)

    try:
        return json.loads(t)
    except Exception:
        pass

    m = _JSON_BLOCK.search(t)
    if not m:
        return {}
    block = m.group(1)
    try:
        return json.loads(block)
    except Exception:
        return [] if block.lstrip().startswith("[") else {}


def require_object(text: str, err: str = "Expected a JSON object.") -> dict:
    """Strict: must return an object, else raise."""
    data = extract_json(text)
    if not isinstance(data, dict):
        raise ValueError(err)
    return data


def require_array(text: str, err: str = "Expected a JSON array.") -> list:
    """Strict: must return an array, else raise."""
    data = extract_json(text)
    if not isinstance(data, list):
        raise ValueError(err)
    return data
