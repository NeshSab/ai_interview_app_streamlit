# core/utils/llm_json.py
from __future__ import annotations
import json
import re
from typing import Any

# Match the first {...} or [...] block, across newlines, as compactly as we can.
_JSON_BLOCK = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Remove surrounding ```...``` fences (with or without 'json') if present."""
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        # Remove starting fence with optional language tag
        t = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", t, flags=re.DOTALL)
        # Remove trailing fence
        t = re.sub(r"\s*```$", "", t, flags=re.DOTALL)
    return t.strip()


def extract_json(text: str) -> Any:
    """
    Extract and parse the first JSON object/array from an LLM response.
    - Safely handles code fences and leading/trailing prose.
    - Returns dict / list on success, or {} / [] on failure (keeping the type intuitive).

    Examples:
      >>> extract_json('```json\\n{"a":1}\\n```') -> {"a":1}
      >>> extract_json('noise {"a":1, "b":[2]} tail') -> {"a":1,"b":[2]}
      >>> extract_json('no json here') -> {}
    """
    if not text:
        return {}
    t = _strip_code_fences(text)

    # Fast path: the whole thing might already be clean JSON
    try:
        return json.loads(t)
    except Exception:
        pass

    # Fallback: find the first JSON-looking block
    m = _JSON_BLOCK.search(t)
    if not m:
        return {}
    block = m.group(1)
    try:
        return json.loads(block)
    except Exception:
        # Final fallback: if it *looks* like an array, return []; else {}
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
