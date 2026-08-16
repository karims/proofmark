"""JSON extraction that survives what models actually emit."""

import json
import re

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_object(text: str | None) -> dict | None:
    """Pull the first plausible JSON object out of model output.

    Handles, in order: clean JSON, fenced code blocks, and objects surrounded by
    prose. Returns ``None`` rather than raising -- an unparseable response is a
    routine outcome here, not an exception.

    Only ``dict`` is accepted. A bare array or scalar is a schema violation for
    every provider's structured-output mode, so treating it as unparseable keeps
    the failure taxonomy honest.
    """
    if not text:
        return None

    cleaned = text.strip()
    parsed = _try_load(cleaned)
    if parsed is not None:
        return parsed

    fenced = _FENCE.search(cleaned)
    if fenced:
        parsed = _try_load(fenced.group(1).strip())
        if parsed is not None:
            return parsed

    return _try_load(_widest_braced_span(cleaned))


def _try_load(candidate: str | None) -> dict | None:
    if not candidate:
        return None
    try:
        loaded = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _widest_braced_span(text: str) -> str | None:
    """Return the span from the first ``{`` to the last ``}``.

    Deliberately greedy: models tend to wrap one object in commentary rather than
    emit several, so the outermost span is far more often the intended payload
    than the first balanced one.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
