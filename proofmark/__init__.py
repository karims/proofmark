"""proofmark -- structured LLM output that has been tested, not just typed.

A proof mark is the stamp struck into a barrel or an ingot after it has passed a
proof test: evidence the piece was put under load and survived, applied by the
proof house rather than the maker.

That is this library's job. Getting a model to emit JSON matching a schema is the
easy half and several libraries do it well. proofmark covers the rest:

- **Preflight** the schema against provider constraints, before spending a call.
- **Degrade** across generation tiers deliberately, and record which one was used.
- **Validate semantically** -- totals that add up, no leftover placeholders, no
  duplicate keys -- because a schema-valid document can still be wrong.
- **Repair deterministically** in code first, and only then spend a model call.
- **Explain the failure** with a category, the specific issues, and a sanitized,
  diffable trace.

Basic use::

    from proofmark import compose, checks, normalize

    result = compose(
        prompt="Draft a 3-day budget for Lisbon, total 1500 EUR.",
        schema=BUDGET_SCHEMA,
        provider=provider,
        normalizers=[
            normalize.coerce_numbers("budget.items[].amount"),
            normalize.rebalance_to_total(
                items="budget.items",
                amount_field="amount",
                total="budget.total",
                slack_match={"category": "contingency"},
            ),
        ],
        checks=[
            checks.sums_to("budget.items", "amount", "budget.total"),
            checks.currency_consistent("EUR"),
            checks.no_placeholders(),
        ],
    )

    if result.ok:
        use(result.data)
    else:
        log(result.failure, result.issues, result.trace)
"""

from proofmark import checks, normalize, paths
from proofmark.errors import (
    CompositionFailed,
    ProofmarkError,
    ProviderError,
    TierUnsupported,
)
from proofmark.issues import Issue, Stage
from proofmark.pipeline import compose
from proofmark.providers.base import Generation, Provider
from proofmark.providers.ollama import OllamaProvider
from proofmark.providers.openai_compatible import OpenAICompatibleProvider
from proofmark.providers.static import CallableProvider, StaticProvider, TierLimitedProvider
from proofmark.result import Failure, Outcome, Result, Tier
from proofmark.schema.preflight import (
    GENERIC,
    OPENAI_STRICT,
    PreflightReport,
    Profile,
    preflight,
)
from proofmark.schema.validate import validate_instance
from proofmark.trace import Trace

__version__ = "0.1.0"

__all__ = [
    "compose",
    "Result",
    "Outcome",
    "Failure",
    "Tier",
    "Issue",
    "Stage",
    "checks",
    "normalize",
    "paths",
    "preflight",
    "PreflightReport",
    "Profile",
    "OPENAI_STRICT",
    "GENERIC",
    "validate_instance",
    "Provider",
    "Generation",
    "OpenAICompatibleProvider",
    "OllamaProvider",
    "StaticProvider",
    "TierLimitedProvider",
    "CallableProvider",
    "Trace",
    "ProofmarkError",
    "CompositionFailed",
    "ProviderError",
    "TierUnsupported",
    "__version__",
]
