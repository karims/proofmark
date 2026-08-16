"""Providers that answer from a script instead of a network.

These make the whole pipeline testable offline -- tier degradation, repair rounds,
refusals, truncation -- without mocking HTTP. Every behaviour proofmark claims to
handle is reproducible in a unit test with one of these.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from proofmark.errors import ProviderError, TierUnsupported
from proofmark.providers.base import Generation
from proofmark.result import Tier


@dataclass
class StaticProvider:
    """Replays a fixed list of responses, one per call.

    Each element of ``responses`` may be a ``dict`` (returned as parsed output), a
    ``str`` (returned as raw text for the pipeline to parse), a ``Generation``, or
    an exception instance to raise. The last element repeats once exhausted, so a
    single-element list models a provider that always says the same thing.
    """

    responses: list[Any] = field(default_factory=list)
    name: str = "static"
    model: str | None = "static-model"
    supported_tiers: tuple[Tier, ...] = (Tier.NATIVE_STRUCTURED, Tier.JSON_MODE, Tier.TEXT_JSON)
    calls: list[dict] = field(default_factory=list, init=False)

    def generate(self, prompt: str, schema: dict, schema_name: str, tier: Tier) -> Generation:
        self.calls.append({"prompt": prompt, "tier": tier, "schema_name": schema_name})

        if not self.responses:
            raise ProviderError("StaticProvider has no configured responses")

        index = min(len(self.calls) - 1, len(self.responses) - 1)
        response = self.responses[index]

        if isinstance(response, BaseException):
            raise response
        if isinstance(response, Generation):
            return response
        if isinstance(response, dict):
            return Generation(
                text=json.dumps(response),
                parsed=response,
                tier=tier,
                provider=self.name,
                model=self.model,
            )
        return Generation(text=str(response), tier=tier, provider=self.name, model=self.model)


@dataclass
class TierLimitedProvider:
    """Wraps another provider and refuses everything above ``highest_tier``.

    Models a backend whose model does not support native structured output, so
    degradation can be tested without inventing HTTP error payloads.
    """

    inner: Any
    highest_tier: Tier = Tier.JSON_MODE
    name: str = "tier-limited"

    def __post_init__(self) -> None:
        self.model = getattr(self.inner, "model", None)
        order = [Tier.NATIVE_STRUCTURED, Tier.JSON_MODE, Tier.TEXT_JSON]
        cutoff = order.index(self.highest_tier)
        self.supported_tiers = tuple(order[cutoff:])

    def generate(self, prompt: str, schema: dict, schema_name: str, tier: Tier) -> Generation:
        if tier not in self.supported_tiers:
            raise TierUnsupported(tier.value, f"{self.name} does not support {tier.value}")
        generation = self.inner.generate(prompt, schema, schema_name, tier)
        generation.tier = tier
        return generation


@dataclass
class CallableProvider:
    """Adapts a plain function into a provider. Useful for one-off fakes."""

    fn: Callable[[str, dict, str, Tier], Any]
    name: str = "callable"
    model: str | None = None
    supported_tiers: tuple[Tier, ...] = (Tier.NATIVE_STRUCTURED, Tier.JSON_MODE, Tier.TEXT_JSON)

    def generate(self, prompt: str, schema: dict, schema_name: str, tier: Tier) -> Generation:
        result = self.fn(prompt, schema, schema_name, tier)
        if isinstance(result, Generation):
            return result
        if isinstance(result, dict):
            return Generation(
                text=json.dumps(result), parsed=result, tier=tier, provider=self.name, model=self.model
            )
        return Generation(text=str(result), tier=tier, provider=self.name, model=self.model)
