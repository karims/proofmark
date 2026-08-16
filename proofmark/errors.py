from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proofmark.result import Result


class ProofmarkError(Exception):
    """Base class for every error this library raises."""


class CompositionFailed(ProofmarkError):
    """Raised by ``compose(on_failure="raise")`` and by ``Result.unwrap()``.

    Carries the whole :class:`~proofmark.result.Result`, so the failure category,
    the unresolved issues, and the trace survive the raise. Callers should not have
    to re-run anything to find out what went wrong.
    """

    def __init__(self, result: "Result"):
        self.result = result
        self.failure = result.failure
        self.issues = result.issues
        detail = result.failure.value if result.failure else "unknown"
        if result.issues:
            shown = "; ".join(item.format() for item in result.issues[:3])
            more = f" (+{len(result.issues) - 3} more)" if len(result.issues) > 3 else ""
            detail = f"{detail}: {shown}{more}"
        super().__init__(detail)


class TierUnsupported(ProofmarkError):
    """Raised by a provider when it cannot serve the requested tier.

    This is a *control-flow* signal, not a fault: the pipeline catches it and
    degrades to the next tier. Providers should raise it for "this model does not
    support json_schema response format" and raise :class:`ProviderError` for
    anything that a retry at a lower tier would not fix.
    """

    def __init__(self, tier: str, detail: str, debug: dict | None = None):
        self.tier = tier
        self.detail = detail
        self.debug = debug or {}
        super().__init__(f"{tier}: {detail}")


class ProviderError(ProofmarkError):
    """Transport, auth, or server-side failure. Degrading tiers will not help."""

    def __init__(self, detail: str, debug: dict | None = None):
        self.detail = detail
        self.debug = debug or {}
        super().__init__(detail)
