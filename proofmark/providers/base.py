from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from proofmark.result import Tier


@dataclass
class Generation:
    """One provider response, before any validation.

    ``parsed`` is populated only when the provider itself returns structured data.
    Otherwise the pipeline extracts it from ``text``, so providers never need to
    implement JSON recovery.
    """

    text: str | None = None
    parsed: dict | None = None
    tier: Tier = Tier.TEXT_JSON
    provider: str = "unknown"
    model: str | None = None
    refusal: str | None = None
    incomplete: bool = False
    debug: dict = field(default_factory=dict)


@runtime_checkable
class Provider(Protocol):
    """What proofmark needs from a model backend.

    Providers are deliberately dumb: they issue one request at one tier and report
    what happened. Tier degradation, JSON recovery, validation, and repair all live
    in the pipeline, so that behaviour is identical across backends and testable
    without a network.

    Signal the difference between "this tier is not available" and "this call
    failed" by raising :class:`~proofmark.errors.TierUnsupported` for the former
    and :class:`~proofmark.errors.ProviderError` for the latter. The pipeline
    degrades on the first and gives up on the second.
    """

    name: str
    model: str | None
    supported_tiers: tuple[Tier, ...]

    def generate(self, prompt: str, schema: dict, schema_name: str, tier: Tier) -> Generation:
        ...


def schema_instruction(schema: dict, schema_name: str) -> str:
    """Prompt text describing the schema, for tiers with no native enforcement."""
    import json

    return (
        f"\n\nReturn a single JSON object named {schema_name} matching this JSON Schema exactly. "
        "Output only the JSON object -- no prose, no markdown fences.\n"
        f"{json.dumps(schema, indent=2)}"
    )
