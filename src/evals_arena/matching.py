"""Structural matching used by tool mocks and trajectory assertions.

Rules (applied recursively):

* ``dict`` expected values match *partially*: every expected key must be present
  in the actual value and match; extra keys in the actual value are ignored.
* ``list`` values must have the same length and match element-wise.
* ``str`` values support three forms:
    - ``"re:<pattern>"``  -> regular expression (``re.search``, case-insensitive)
    - a string containing ``*``, ``?`` or ``[`` -> shell-style glob (case-insensitive)
    - anything else -> exact equality (case-insensitive when the actual is a string)
* the sentinel string ``"<any>"`` matches any value, including ``None``.
* other scalars use ``==``; ints and floats compare numerically.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any

ANY = "<any>"
_GLOB_CHARS = set("*?[")


def _match_str(expected: str, actual: Any) -> bool:
    if expected == ANY:
        return True
    if not isinstance(actual, str):
        # Allow "42" to match 42 so YAML authors don't have to care about quoting.
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and (
            expected == str(actual)
        )
    if expected.startswith("re:"):
        return re.search(expected[3:], actual, flags=re.IGNORECASE) is not None
    if _GLOB_CHARS & set(expected):
        return fnmatch.fnmatchcase(actual.lower(), expected.lower())
    return expected.casefold() == actual.casefold()


def matches(expected: Any, actual: Any) -> bool:
    """Return ``True`` when ``actual`` satisfies the ``expected`` pattern."""
    if isinstance(expected, str):
        return _match_str(expected, actual)
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(k in actual and matches(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        return all(matches(e, a) for e, a in zip(expected, actual, strict=True))
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(expected) == float(actual)
    return bool(expected == actual)


def explain_mismatch(expected: Any, actual: Any, path: str = "$") -> str | None:
    """Return a human readable reason for the first mismatch, or ``None``."""
    if matches(expected, actual):
        return None
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key, value in expected.items():
            if key not in actual:
                return f"{path}.{key} missing"
            reason = explain_mismatch(value, actual[key], f"{path}.{key}")
            if reason:
                return reason
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path} length {len(actual)} != {len(expected)}"
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            reason = explain_mismatch(e, a, f"{path}[{i}]")
            if reason:
                return reason
    return f"{path}: expected {expected!r}, got {actual!r}"
