from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from proofmark.issues import Issue


class Tier(str, Enum):
    """How the payload was requested, strongest first.

    The pipeline walks these in order and degrades when a provider rejects one, so
    the tier that actually produced the payload is a property of the *result*, not
    of the configuration. Recording it is the difference between "the model is bad
    at JSON" and "your model silently fell back to prompt-and-pray."
    """

    NATIVE_STRUCTURED = "native_structured"
    JSON_MODE = "json_mode"
    TEXT_JSON = "text_json"
    NONE = "none"


TIER_ORDER: tuple[Tier, ...] = (Tier.NATIVE_STRUCTURED, Tier.JSON_MODE, Tier.TEXT_JSON)


class Outcome(str, Enum):
    """What it took to get a usable payload."""

    OK = "ok"
    """First generation validated clean."""

    NORMALIZED = "normalized"
    """Deterministic normalizers fixed it. No extra model call."""

    REPAIRED = "repaired"
    """A model repair round fixed it."""

    FALLBACK = "fallback"
    """Generation or validation failed; the caller's fallback was used."""

    FAILED = "failed"
    """Nothing usable was produced."""


class Failure(str, Enum):
    """Why a run did not produce a clean payload on its own.

    Present on any result that is not ``Outcome.OK`` -- including successful ones,
    where it records what had to be recovered from.
    """

    INVALID_SCHEMA = "invalid_schema"
    """Caught by preflight. No provider call was made."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_REFUSAL = "provider_refusal"
    INCOMPLETE_RESPONSE = "incomplete_response"
    """Truncated -- usually the token limit, not the model's fault."""

    EMPTY_RESPONSE = "empty_response"
    UNPARSEABLE_RESPONSE = "unparseable_response"
    STRUCTURAL_INVALID = "structural_invalid"
    """Parsed, but does not match the schema."""

    SEMANTIC_INVALID = "semantic_invalid"
    """Matches the schema and is still wrong."""


RECOVERABLE_BY_REPAIR: frozenset[Failure] = frozenset(
    {
        Failure.STRUCTURAL_INVALID,
        Failure.SEMANTIC_INVALID,
        Failure.UNPARSEABLE_RESPONSE,
        Failure.INCOMPLETE_RESPONSE,
    }
)
"""Failures where showing the model its own mistake is worth a second call.

A refusal or a provider error will not be fixed by asking again with the same
prompt, so the pipeline does not spend a repair call on them.
"""


@dataclass
class Result:
    """The outcome of one :func:`proofmark.compose` call.

    Truthiness follows ``data is not None``, so ``if result:`` reads correctly.
    """

    data: dict | None
    outcome: Outcome
    tier: Tier
    failure: Failure | None = None
    issues: list[Issue] = field(default_factory=list)
    initial_issues: list[Issue] = field(default_factory=list)
    repair_issues: list[Issue] = field(default_factory=list)
    normalized: bool = False
    normalizations: list[str] = field(default_factory=list)
    repair_attempted: bool = False
    provider_calls: int = 0
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.data is not None

    @property
    def clean(self) -> bool:
        """True only if the model got it right unaided."""
        return self.outcome is Outcome.OK

    def __bool__(self) -> bool:
        return self.ok

    def unwrap(self) -> dict:
        """Return the payload, or raise if there is none."""
        if self.data is None:
            from proofmark.errors import CompositionFailed

            raise CompositionFailed(self)
        return self.data

    def summary(self) -> str:
        parts = [f"outcome={self.outcome.value}", f"tier={self.tier.value}"]
        if self.failure is not None:
            parts.append(f"failure={self.failure.value}")
        if self.normalized:
            parts.append(f"normalized={','.join(self.normalizations)}")
        if self.repair_attempted:
            parts.append("repair=attempted")
        parts.append(f"provider_calls={self.provider_calls}")
        if self.issues:
            parts.append(f"unresolved_issues={len(self.issues)}")
        return " ".join(parts)
