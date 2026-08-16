"""A sanitized, diffable record of what happened.

The trace is the feature. Everything else in this library also exists somewhere
else in some form; being able to answer *why did this fail* without re-running
anything is what makes proofmark worth installing.

Two rules it must never break:

1. **Diffable.** Two runs of the same input produce traces that differ only where
   behaviour differed. No timestamps, no object ids, no set iteration order.
2. **Safe to attach.** Traces get pasted into bug reports. Credentials are redacted
   on the way in, not on the way out, and prompts are excluded by default because
   they routinely carry customer data.
"""

import re
from dataclasses import dataclass, field
from typing import Any

_SECRET_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "token", "access_token", "secret", "password"}
)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
)
_REDACTED = "[redacted]"
MAX_TEXT = 4000


@dataclass
class Trace:
    """An ordered list of events. Append-only."""

    include_prompts: bool = False
    max_text: int = MAX_TEXT
    events: list[dict] = field(default_factory=list)

    def record(self, event: str, **fields: Any) -> None:
        entry = {"event": event}
        for key, value in fields.items():
            if value is None:
                continue
            if key in {"prompt", "repair_prompt"} and not self.include_prompts:
                entry[f"{key}_chars"] = len(str(value))
                continue
            entry[key] = sanitize(value, self.max_text)
        self.events.append(entry)

    def to_dict(self) -> dict:
        return {"events": list(self.events)}

    def summary(self) -> list[str]:
        return [str(entry.get("event")) for entry in self.events]


def sanitize(value: Any, max_text: int = MAX_TEXT) -> Any:
    """Recursively redact credentials and truncate long strings."""
    if isinstance(value, str):
        return _sanitize_text(value, max_text)
    if isinstance(value, dict):
        return {
            key: (_REDACTED if str(key).lower() in _SECRET_KEYS else sanitize(item, max_text))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(item, max_text) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return getattr(value, "value")  # Enum
    return _sanitize_text(str(value), max_text)


def _sanitize_text(text: str, max_text: int) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    if len(text) > max_text:
        return text[:max_text] + f"... [truncated {len(text) - max_text} chars]"
    return text
