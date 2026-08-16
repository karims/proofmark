from proofmark import compose
from proofmark.providers.static import StaticProvider
from proofmark.trace import Trace, sanitize
from tests.conftest import BUDGET_SCHEMA, budget_payload


def test_api_keys_are_redacted_from_nested_structures():
    sanitized = sanitize({"headers": {"api_key": "sk-abcdef123456", "accept": "json"}})

    assert sanitized["headers"]["api_key"] == "[redacted]"
    assert sanitized["headers"]["accept"] == "json"


def test_key_shaped_strings_are_redacted_wherever_they_appear():
    sanitized = sanitize("failed with token sk-abcdef1234567890 in the body")

    assert "sk-abcdef1234567890" not in sanitized
    assert "[redacted]" in sanitized


def test_bearer_tokens_are_redacted():
    assert "[redacted]" in sanitize("Authorization: Bearer abcdefgh12345678")


def test_long_text_is_truncated_with_a_count():
    sanitized = sanitize("x" * 5000, max_text=100)

    assert sanitized.startswith("x" * 100)
    assert "truncated 4900 chars" in sanitized


def test_prompts_are_excluded_by_default():
    trace = Trace()

    trace.record("generate", prompt="customer name is Jane Doe")

    assert "prompt" not in trace.events[0]
    assert trace.events[0]["prompt_chars"] == 25


def test_prompts_are_included_when_explicitly_requested():
    trace = Trace(include_prompts=True)

    trace.record("generate", prompt="hello")

    assert trace.events[0]["prompt"] == "hello"


def test_pipeline_trace_records_the_sequence_of_events():
    provider = StaticProvider(responses=[budget_payload()])

    result = compose(prompt="x", schema=BUDGET_SCHEMA, provider=provider)

    events = [entry["event"] for entry in result.trace["events"]]
    assert events == ["preflight", "generate", "generated", "validated"]


def test_pipeline_trace_is_deterministic_across_identical_runs():
    def once():
        return compose(
            prompt="x", schema=BUDGET_SCHEMA, provider=StaticProvider(responses=[budget_payload()])
        ).trace

    assert once() == once()


def test_trace_records_preflight_warnings_even_on_success():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["amount"],
        "properties": {"amount": {"type": "number", "minimum": 0}},
    }
    provider = StaticProvider(responses=[{"amount": 5}])

    result = compose(prompt="x", schema=schema, provider=provider)

    assert result.ok
    assert result.trace["events"][0]["warnings"]
