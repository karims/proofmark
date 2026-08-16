"""End-to-end demonstration. Runs offline -- no API key, no network.

    python examples/budget.py

Uses StaticProvider to replay a response of the kind models actually return: right
types, wrong arithmetic, amounts as strings, and a chatty extra key. Watch it get
fixed without a second call.

To run against a real model, replace the provider:

    from proofmark import OpenAICompatibleProvider
    provider = OpenAICompatibleProvider(api_key=os.environ["OPENAI_API_KEY"],
                                        model="gpt-4o-mini")
"""

import json

from proofmark import Outcome, StaticProvider, checks, compose, normalize, preflight

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "currency", "budget"],
    "properties": {
        "title": {"type": "string"},
        "currency": {"type": "string"},
        "budget": {
            "type": "object",
            "additionalProperties": False,
            "required": ["total", "items"],
            "properties": {
                "total": {"type": "number"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["category", "amount"],
                        "properties": {
                            "category": {"type": "string"},
                            "amount": {"type": "number"},
                        },
                    },
                },
            },
        },
    },
}

# What a model plausibly returns: schema-shaped, arithmetically wrong (items sum to
# 950, not 1000), amounts as strings, and an extra key the schema forbids.
MODEL_RESPONSE = {
    "title": "Three days in Lisbon",
    "currency": "EUR",
    "commentary": "Let me know if you'd like me to adjust anything!",
    "budget": {
        "total": 1000.0,
        "items": [
            {"category": "lodging", "amount": "500.00"},
            {"category": "food", "amount": "300.00"},
            {"category": "contingency", "amount": "150.00"},
        ],
    },
}


def main() -> None:
    report = preflight(SCHEMA)
    print(f"preflight: {'ok' if report.ok else 'FAILED'}, {len(report.warnings)} warning(s)")

    provider = StaticProvider(responses=[MODEL_RESPONSE])

    result = compose(
        prompt="Draft a 3-day budget for Lisbon. Total 1000 EUR.",
        schema=SCHEMA,
        provider=provider,
        normalizers=[
            normalize.drop_unknown_properties(SCHEMA),
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

    print(f"result:    {result.summary()}")
    print(f"calls:     {len(provider.calls)} (the model was never asked to try again)")
    print()
    print(json.dumps(result.data, indent=2))
    print()

    assert result.ok
    assert result.outcome is Outcome.NORMALIZED, "the model was wrong; code fixed it"
    assert sum(item["amount"] for item in result.data["budget"]["items"]) == 1000.0
    assert "commentary" not in result.data
    print("Line items now sum to exactly 1000.00, in one provider call.")
    print("Trace events:", " -> ".join(entry["event"] for entry in result.trace["events"]))


if __name__ == "__main__":
    main()
