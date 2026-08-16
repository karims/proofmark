import json

import pytest

from proofmark import checks, compose, normalize
from proofmark.errors import CompositionFailed, ProviderError, TierUnsupported
from proofmark.providers.base import Generation
from proofmark.providers.static import StaticProvider, TierLimitedProvider
from proofmark.result import Failure, Outcome, Tier
from tests.conftest import BUDGET_SCHEMA, budget_payload


def run(responses, **kwargs):
    provider = StaticProvider(responses=responses)
    kwargs.setdefault("schema", BUDGET_SCHEMA)
    result = compose(prompt="build a budget", provider=provider, **kwargs)
    return result, provider


def test_clean_generation_is_reported_as_ok():
    result, provider = run([budget_payload()])

    assert result.ok
    assert result.outcome is Outcome.OK
    assert result.failure is None
    assert result.tier is Tier.NATIVE_STRUCTURED
    assert result.provider_calls == 1
    assert len(provider.calls) == 1


def test_result_is_truthy_and_unwraps():
    result, _ = run([budget_payload()])

    assert result
    assert result.unwrap()["title"] == "Lisbon"


def test_invalid_schema_never_reaches_the_provider():
    provider = StaticProvider(responses=[budget_payload()])
    open_schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}}

    result = compose(prompt="x", schema=open_schema, provider=provider)

    assert not result.ok
    assert result.failure is Failure.INVALID_SCHEMA
    assert result.tier is Tier.NONE
    assert provider.calls == []
    assert result.provider_calls == 0


def test_missing_provider_uses_the_fallback():
    result = compose(prompt="x", schema=BUDGET_SCHEMA, provider=None, fallback=budget_payload())

    assert result.ok
    assert result.outcome is Outcome.FALLBACK
    assert result.failure is Failure.PROVIDER_UNAVAILABLE


def test_missing_provider_without_fallback_fails_cleanly():
    result = compose(prompt="x", schema=BUDGET_SCHEMA, provider=None)

    assert not result.ok
    assert result.failure is Failure.PROVIDER_UNAVAILABLE


def test_json_wrapped_in_fences_and_prose_is_recovered():
    text = "Sure! Here you go:\n```json\n" + json.dumps(budget_payload()) + "\n```\nHope that helps."

    result, _ = run([text])

    assert result.ok
    assert result.outcome is Outcome.OK


def test_empty_response_is_categorised():
    result, _ = run(["   "], repair_attempts=0)

    assert result.failure is Failure.EMPTY_RESPONSE


def test_unparseable_response_is_categorised():
    result, _ = run(["I'd rather describe it in words."], repair_attempts=0)

    assert result.failure is Failure.UNPARSEABLE_RESPONSE


def test_refusal_is_categorised_and_not_repaired():
    refusal = Generation(text=None, refusal="I can't help with that.", tier=Tier.NATIVE_STRUCTURED)

    result, provider = run([refusal], repair_attempts=1)

    assert result.failure is Failure.PROVIDER_REFUSAL
    assert not result.repair_attempted
    assert len(provider.calls) == 1


def test_truncated_unparseable_response_is_reported_as_incomplete():
    truncated = Generation(text='{"title": "Lis', incomplete=True, tier=Tier.NATIVE_STRUCTURED)

    result, _ = run([truncated], repair_attempts=0)

    assert result.failure is Failure.INCOMPLETE_RESPONSE


def test_structural_failure_is_distinguished_from_semantic():
    broken = budget_payload()
    broken["surprise"] = 1

    result, _ = run([broken], repair_attempts=0)

    assert result.failure is Failure.STRUCTURAL_INVALID


def test_semantic_failure_is_reported_when_the_schema_is_satisfied():
    result, _ = run(
        [budget_payload(items=[{"category": "a", "amount": 1.0}])],
        repair_attempts=0,
        checks=[checks.sums_to("budget.items", "amount", "budget.total")],
    )

    assert result.failure is Failure.SEMANTIC_INVALID
    assert result.issues[0].code == "total_mismatch"


def test_semantic_checks_are_skipped_when_the_payload_is_structurally_broken():
    broken = budget_payload(items=[{"category": "a", "amount": 1.0}])
    broken["surprise"] = 1

    result, _ = run(
        [broken],
        repair_attempts=0,
        checks=[checks.sums_to("budget.items", "amount", "budget.total")],
    )

    assert [item.code for item in result.issues] == ["additional_property"]


def test_normalizers_fix_it_without_a_second_call():
    drifted = budget_payload(
        items=[
            {"category": "lodging", "amount": 500.0},
            {"category": "food", "amount": 300.0},
            {"category": "contingency", "amount": 150.0},
        ]
    )

    result, provider = run(
        [drifted],
        checks=[checks.sums_to("budget.items", "amount", "budget.total")],
        normalizers=[
            normalize.rebalance_to_total(
                items="budget.items",
                amount_field="amount",
                total="budget.total",
                slack_match={"category": "contingency"},
            )
        ],
    )

    assert result.ok
    assert result.outcome is Outcome.NORMALIZED
    assert result.normalizations == ["rebalance_to_total"]
    assert len(provider.calls) == 1, "normalization must not cost a provider call"
    assert result.data["budget"]["items"][2]["amount"] == 200.0


def test_normalizers_run_before_validation_so_coercion_prevents_a_repair():
    stringly = budget_payload(
        items=[
            {"category": "lodging", "amount": "500.00"},
            {"category": "food", "amount": "300.00"},
            {"category": "contingency", "amount": "200.00"},
        ]
    )

    result, provider = run(
        [stringly],
        normalizers=[normalize.coerce_numbers("budget.items[].amount")],
    )

    assert result.ok
    assert result.outcome is Outcome.NORMALIZED
    assert len(provider.calls) == 1


def test_model_repair_is_used_when_normalization_cannot_fix_it():
    broken = budget_payload()
    del broken["title"]

    result, provider = run([broken, budget_payload()], repair_attempts=1)

    assert result.ok
    assert result.outcome is Outcome.REPAIRED
    assert result.repair_attempted
    assert result.failure is Failure.STRUCTURAL_INVALID, "records what had to be recovered from"
    assert len(provider.calls) == 2


def test_repair_prompt_quotes_the_specific_issues():
    broken = budget_payload()
    del broken["title"]

    _, provider = run([broken, budget_payload()], repair_attempts=1)

    repair_prompt = provider.calls[1]["prompt"]
    assert "$.title: required property is missing" in repair_prompt
    assert "build a budget" in repair_prompt


def test_repair_attempts_zero_disables_the_second_call():
    broken = budget_payload()
    del broken["title"]

    result, provider = run([broken], repair_attempts=0)

    assert not result.ok
    assert not result.repair_attempted
    assert len(provider.calls) == 1


def test_failed_repair_falls_back_when_a_fallback_is_supplied():
    broken = budget_payload()
    del broken["title"]

    result, _ = run([broken, broken], repair_attempts=1, fallback=budget_payload())

    assert result.ok
    assert result.outcome is Outcome.FALLBACK
    assert result.repair_attempted
    assert result.initial_issues and result.repair_issues


def test_on_failure_raise_carries_the_whole_result():
    with pytest.raises(CompositionFailed) as caught:
        run(["not json"], repair_attempts=0, on_failure="raise")

    assert caught.value.failure is Failure.UNPARSEABLE_RESPONSE
    assert caught.value.result.trace["events"]


def test_unwrap_raises_on_a_failed_result():
    result, _ = run(["not json"], repair_attempts=0)

    with pytest.raises(CompositionFailed):
        result.unwrap()


def test_tier_degrades_when_the_provider_rejects_native_structured():
    inner = StaticProvider(responses=[budget_payload()])
    provider = TierLimitedProvider(inner=inner, highest_tier=Tier.JSON_MODE)

    result = compose(prompt="x", schema=BUDGET_SCHEMA, provider=provider)

    assert result.ok
    assert result.tier is Tier.JSON_MODE


def test_declared_tier_support_is_used_to_skip_doomed_calls():
    inner = StaticProvider(responses=[budget_payload()])
    provider = TierLimitedProvider(inner=inner, highest_tier=Tier.TEXT_JSON)

    result = compose(prompt="x", schema=BUDGET_SCHEMA, provider=provider)

    assert result.tier is Tier.TEXT_JSON
    assert len(inner.calls) == 1, "a provider that declares its tiers should not be probed"


def test_runtime_rejection_degrades_and_is_recorded():
    provider = StaticProvider(
        responses=[
            TierUnsupported("native_structured", "model does not support json_schema"),
            TierUnsupported("json_mode", "model does not support response_format"),
            budget_payload(),
        ]
    )

    result = compose(prompt="x", schema=BUDGET_SCHEMA, provider=provider)

    assert result.ok
    assert result.tier is Tier.TEXT_JSON
    events = [entry["event"] for entry in result.trace["events"]]
    assert events.count("tier_unsupported") == 2
    assert result.provider_calls == 1, "rejected attempts are not billable generations"


def test_max_tier_can_skip_native_structured():
    provider = StaticProvider(responses=[budget_payload()])

    result = compose(prompt="x", schema=BUDGET_SCHEMA, provider=provider, max_tier=Tier.JSON_MODE)

    assert result.tier is Tier.JSON_MODE
    assert provider.calls[0]["tier"] is Tier.JSON_MODE


def test_provider_error_does_not_degrade_tiers():
    provider = StaticProvider(responses=[ProviderError("401 invalid api key")])

    result = compose(prompt="x", schema=BUDGET_SCHEMA, provider=provider)

    assert result.failure is Failure.PROVIDER_ERROR
    assert len(provider.calls) == 1, "an auth failure must not be retried at a lower tier"


def test_tier_unsupported_on_every_tier_is_a_provider_error():
    provider = StaticProvider(
        responses=[TierUnsupported("native_structured", "nope")],
    )

    result = compose(prompt="x", schema=BUDGET_SCHEMA, provider=provider)

    assert result.failure is Failure.PROVIDER_ERROR
    assert len(provider.calls) == 3


def test_full_pipeline_combines_normalization_and_repair():
    first = budget_payload(
        items=[
            {"category": "lodging", "amount": "500.00"},
            {"category": "food", "amount": "300.00"},
            {"category": "contingency", "amount": "150.00"},
        ],
        title="TBD",
    )
    second = budget_payload(
        items=[
            {"category": "lodging", "amount": "500.00"},
            {"category": "food", "amount": "300.00"},
            {"category": "contingency", "amount": "150.00"},
        ],
        title="Three days in Lisbon",
    )

    result, provider = run(
        [first, second],
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
            checks.no_placeholders(),
            checks.currency_consistent("EUR"),
        ],
        repair_attempts=1,
    )

    assert result.ok
    assert result.outcome is Outcome.REPAIRED
    assert result.data["budget"]["items"][2]["amount"] == 200.0
    assert result.data["title"] == "Three days in Lisbon"
    assert len(provider.calls) == 2


def test_summary_is_a_single_readable_line():
    result, _ = run([budget_payload()])

    assert result.summary() == "outcome=ok tier=native_structured provider_calls=1"
