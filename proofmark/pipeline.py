"""The composition pipeline.

The order of operations is the whole design:

1. **Preflight the schema.** Free, and catches the failures that would otherwise
   cost a call and return an opaque 400.
2. **Generate**, walking down the tiers, degrading only when a provider says a tier
   is unavailable -- never when a call simply failed.
3. **Parse**, tolerating fences and surrounding prose.
4. **Normalize deterministically.** Arithmetic and formatting get fixed in code.
5. **Validate**, structurally and then semantically.
6. **Repair with the model** only for what survived steps 4 and 5, and only for
   failures a retry could plausibly fix.
7. **Fall back**, or fail with a category and a trace.

Steps 4 and 6 in that order are the point. Asking a model to redo arithmetic it
just got wrong is slower, more expensive, and less reliable than doing the
arithmetic.
"""

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from proofmark._json import extract_json_object
from proofmark.checks import Check
from proofmark.errors import CompositionFailed, ProviderError, TierUnsupported
from proofmark.issues import Issue, Stage
from proofmark.normalize import Normalization, Normalizer
from proofmark.providers.base import Generation, Provider
from proofmark.result import (
    RECOVERABLE_BY_REPAIR,
    TIER_ORDER,
    Failure,
    Outcome,
    Result,
    Tier,
)
from proofmark.schema.preflight import OPENAI_STRICT, Profile, preflight
from proofmark.schema.validate import validate_instance
from proofmark.trace import Trace

StructuralValidator = Callable[[dict, dict], list[Issue]]


@dataclass
class _Attempt:
    payload: dict | None
    issues: list[Issue]
    failure: Failure | None
    normalizations: list[Normalization]

    @property
    def normalized(self) -> bool:
        return any(item.changed for item in self.normalizations)


def compose(
    prompt: str,
    schema: dict,
    provider: Provider | None = None,
    *,
    schema_name: str = "response",
    checks: Iterable[Check] = (),
    normalizers: Iterable[Normalizer] = (),
    repair_attempts: int = 1,
    profile: Profile = OPENAI_STRICT,
    max_tier: Tier = Tier.NATIVE_STRUCTURED,
    fallback: dict | Callable[[], dict] | None = None,
    on_failure: str = "return",
    structural_validator: StructuralValidator | None = None,
    include_prompts_in_trace: bool = False,
) -> Result:
    """Produce a payload that satisfies ``schema`` both structurally and semantically.

    Args:
        prompt: What to ask for. The schema is described separately per tier.
        schema: JSON Schema for the payload. Must describe an object.
        provider: Model backend. ``None`` goes straight to ``fallback``, which makes
            offline tests and no-key environments a supported path rather than a crash.
        checks: Semantic checks. This is where correctness that schemas cannot
            express belongs.
        normalizers: Deterministic repairs, applied in order before validation.
        repair_attempts: Model repair rounds after normalization fails. ``0``
            disables. Above ``1`` has sharply diminishing returns.
        profile: Provider constraints for preflight.
        max_tier: Highest tier to attempt.
        fallback: Payload, or callable returning one, used when generation fails.
        on_failure: ``"return"`` for a failed :class:`Result`, ``"raise"`` for
            :class:`~proofmark.errors.CompositionFailed`.
        structural_validator: Override the built-in schema validator, e.g. to plug
            in ``jsonschema``. Takes ``(payload, schema)`` and returns issues.
        include_prompts_in_trace: Prompts are excluded by default because they carry
            user data.

    Returns:
        A :class:`~proofmark.result.Result`. Check ``.ok``, or call ``.unwrap()``.
    """
    trace = Trace(include_prompts=include_prompts_in_trace)
    check_list = list(checks)
    normalizer_list = list(normalizers)
    validate = structural_validator or (lambda payload, target: validate_instance(payload, target))

    report = preflight(schema, profile)
    trace.record(
        "preflight",
        profile=profile.name,
        errors=[item.to_dict() for item in report.errors],
        warnings=[item.to_dict() for item in report.warnings],
    )
    if not report.ok:
        return _finish(
            _fallback_or_fail(
                fallback, Failure.INVALID_SCHEMA, report.errors, trace, tier=Tier.NONE, provider_calls=0
            ),
            on_failure,
        )

    if provider is None:
        trace.record("provider_missing")
        return _finish(
            _fallback_or_fail(
                fallback, Failure.PROVIDER_UNAVAILABLE, [], trace, tier=Tier.NONE, provider_calls=0
            ),
            on_failure,
        )

    calls = 0
    try:
        generation, calls = _generate(provider, prompt, schema, schema_name, max_tier, trace, calls)
    except ProviderError as exc:
        trace.record("provider_error", detail=str(exc), debug=exc.debug)
        return _finish(
            _fallback_or_fail(
                fallback, Failure.PROVIDER_ERROR, [], trace, tier=Tier.NONE, provider_calls=calls
            ),
            on_failure,
        )
    except _AllTiersRejected as exc:
        trace.record("all_tiers_rejected", detail=str(exc))
        return _finish(
            _fallback_or_fail(
                fallback, Failure.PROVIDER_ERROR, [], trace, tier=Tier.NONE, provider_calls=calls
            ),
            on_failure,
        )

    tier = generation.tier
    attempt = _evaluate(generation, schema, validate, check_list, normalizer_list, trace, "initial")
    initial_issues = list(attempt.issues)

    if attempt.payload is not None and not attempt.issues:
        return _finish(
            Result(
                data=attempt.payload,
                outcome=Outcome.NORMALIZED if attempt.normalized else Outcome.OK,
                tier=tier,
                failure=_failure_for(initial_issues) if attempt.normalized else None,
                initial_issues=initial_issues,
                normalized=attempt.normalized,
                normalizations=[item.name for item in attempt.normalizations if item.changed],
                provider_calls=calls,
                trace=trace.to_dict(),
            ),
            on_failure,
        )

    repairable = attempt.failure in RECOVERABLE_BY_REPAIR
    if repair_attempts <= 0 or not repairable:
        trace.record(
            "repair_skipped",
            reason="disabled" if repair_attempts <= 0 else f"{attempt.failure} is not repairable",
        )
        return _finish(
            _fallback_or_fail(
                fallback,
                attempt.failure or Failure.STRUCTURAL_INVALID,
                attempt.issues,
                trace,
                tier=tier,
                provider_calls=calls,
                initial_issues=initial_issues,
                normalizations=attempt.normalizations,
            ),
            on_failure,
        )

    repair_issues: list[Issue] = []
    for round_index in range(repair_attempts):
        repair_prompt = _repair_prompt(prompt, attempt.payload, generation.text, attempt.issues, schema_name)
        trace.record("repair_request", round=round_index + 1, repair_prompt=repair_prompt, tier=tier.value)
        try:
            generation = provider.generate(repair_prompt, schema, schema_name, tier)
            calls += 1
        except (ProviderError, TierUnsupported) as exc:
            trace.record("repair_failed", round=round_index + 1, detail=str(exc))
            break

        attempt = _evaluate(
            generation, schema, validate, check_list, normalizer_list, trace, f"repair_{round_index + 1}"
        )
        repair_issues = list(attempt.issues)
        if attempt.payload is not None and not attempt.issues:
            return _finish(
                Result(
                    data=attempt.payload,
                    outcome=Outcome.REPAIRED,
                    tier=tier,
                    failure=_failure_for(initial_issues),
                    initial_issues=initial_issues,
                    repair_issues=[],
                    normalized=attempt.normalized,
                    normalizations=[item.name for item in attempt.normalizations if item.changed],
                    repair_attempted=True,
                    provider_calls=calls,
                    trace=trace.to_dict(),
                ),
                on_failure,
            )

    return _finish(
        _fallback_or_fail(
            fallback,
            attempt.failure or Failure.SEMANTIC_INVALID,
            attempt.issues,
            trace,
            tier=tier,
            provider_calls=calls,
            initial_issues=initial_issues,
            repair_issues=repair_issues,
            repair_attempted=True,
            normalizations=attempt.normalizations,
        ),
        on_failure,
    )


class _AllTiersRejected(Exception):
    pass


def _generate(
    provider: Provider,
    prompt: str,
    schema: dict,
    schema_name: str,
    max_tier: Tier,
    trace: Trace,
    calls: int,
) -> tuple[Generation, int]:
    """Walk tiers downward, degrading only on an explicit TierUnsupported."""
    supported = tuple(getattr(provider, "supported_tiers", TIER_ORDER))
    start = TIER_ORDER.index(max_tier) if max_tier in TIER_ORDER else 0
    candidates = [tier for tier in TIER_ORDER[start:] if tier in supported]

    if not candidates:
        raise _AllTiersRejected(f"provider supports none of {[tier.value for tier in TIER_ORDER[start:]]}")

    last_detail = ""
    for tier in candidates:
        trace.record("generate", tier=tier.value, prompt=prompt)
        try:
            generation = provider.generate(prompt, schema, schema_name, tier)
        except TierUnsupported as exc:
            last_detail = exc.detail
            trace.record("tier_unsupported", tier=tier.value, detail=exc.detail, debug=exc.debug)
            continue
        calls += 1
        generation.tier = tier
        trace.record(
            "generated",
            tier=tier.value,
            provider=generation.provider,
            model=generation.model,
            refusal=generation.refusal,
            incomplete=generation.incomplete,
            debug=generation.debug,
        )
        return generation, calls

    raise _AllTiersRejected(last_detail or "every tier was rejected")


def _evaluate(
    generation: Generation,
    schema: dict,
    validate: StructuralValidator,
    checks: list[Check],
    normalizers: list[Normalizer],
    trace: Trace,
    label: str,
) -> _Attempt:
    if generation.refusal:
        trace.record("refusal", stage=label, detail=generation.refusal)
        return _Attempt(None, [], Failure.PROVIDER_REFUSAL, [])

    payload = generation.parsed or extract_json_object(generation.text)

    if payload is None:
        if not (generation.text or "").strip():
            trace.record("empty_response", stage=label)
            return _Attempt(None, [], Failure.EMPTY_RESPONSE, [])
        failure = Failure.INCOMPLETE_RESPONSE if generation.incomplete else Failure.UNPARSEABLE_RESPONSE
        trace.record("unparseable", stage=label, failure=failure.value, raw=generation.text)
        return _Attempt(None, [], failure, [])

    if generation.incomplete:
        trace.record("truncated_but_parsed", stage=label)

    payload = copy.deepcopy(payload)

    applied: list[Normalization] = []
    for normalizer in normalizers:
        outcome = normalizer(payload)
        applied.append(outcome)
        if outcome.changed:
            trace.record("normalized", stage=label, normalizer=outcome.name, detail=outcome.detail)

    structural = validate(payload, schema)
    semantic: list[Issue] = []
    if not structural:
        # Semantic checks assume a well-formed payload. Running them on a payload
        # that failed structurally produces noise addressed to fields that may not
        # exist, which then pollutes the repair prompt.
        for check in checks:
            semantic.extend(check(payload))

    issues = structural + semantic
    trace.record(
        "validated",
        stage=label,
        structural=[item.to_dict() for item in structural],
        semantic=[item.to_dict() for item in semantic],
    )

    failure = _failure_for(issues)
    return _Attempt(payload, issues, failure, applied)


def _failure_for(issues: list[Issue]) -> Failure | None:
    if not issues:
        return None
    if any(item.stage is Stage.STRUCTURAL for item in issues):
        return Failure.STRUCTURAL_INVALID
    return Failure.SEMANTIC_INVALID


def _repair_prompt(
    original: str,
    payload: dict | None,
    raw_text: str | None,
    issues: list[Issue],
    schema_name: str,
) -> str:
    body = json.dumps(payload, indent=2) if payload is not None else (raw_text or "")
    listed = "\n".join(f"- {item.format()}" for item in issues)
    return (
        f"{original}\n\n"
        f"A previous attempt produced this {schema_name}:\n\n{body}\n\n"
        f"It has the following problems:\n{listed}\n\n"
        "Return a corrected version of the complete object. Fix only the listed "
        "problems and keep everything else identical. Output only the JSON object."
    )


def _fallback_or_fail(
    fallback: dict | Callable[[], dict] | None,
    failure: Failure,
    issues: list[Issue],
    trace: Trace,
    tier: Tier,
    provider_calls: int,
    initial_issues: list[Issue] | None = None,
    repair_issues: list[Issue] | None = None,
    repair_attempted: bool = False,
    normalizations: list[Normalization] | None = None,
) -> Result:
    applied = normalizations or []
    common: dict[str, Any] = {
        "tier": tier,
        "failure": failure,
        "issues": issues,
        "initial_issues": initial_issues or [],
        "repair_issues": repair_issues or [],
        "repair_attempted": repair_attempted,
        "normalized": any(item.changed for item in applied),
        "normalizations": [item.name for item in applied if item.changed],
        "provider_calls": provider_calls,
    }

    if fallback is not None:
        data = fallback() if callable(fallback) else copy.deepcopy(fallback)
        trace.record("fallback_used", failure=failure.value)
        return Result(data=data, outcome=Outcome.FALLBACK, trace=trace.to_dict(), **common)

    trace.record("failed", failure=failure.value, issue_count=len(issues))
    return Result(data=None, outcome=Outcome.FAILED, trace=trace.to_dict(), **common)


def _finish(result: Result, on_failure: str) -> Result:
    if on_failure == "raise" and result.data is None:
        raise CompositionFailed(result)
    return result
