from decimal import Decimal

from proofmark import normalize
from tests.conftest import BUDGET_SCHEMA, budget_payload


def total_of(payload):
    return sum(Decimal(str(item["amount"])) for item in payload["budget"]["items"])


def test_coerce_numbers_parses_formatted_strings():
    payload = budget_payload(
        items=[
            {"category": "a", "amount": "1,200.50"},
            {"category": "b", "amount": "$300"},
            {"category": "c", "amount": " 42 "},
        ]
    )

    outcome = normalize.coerce_numbers("budget.items[].amount")(payload)

    assert outcome.changed
    assert [item["amount"] for item in payload["budget"]["items"]] == [1200.5, 300.0, 42.0]


def test_coerce_numbers_leaves_real_text_alone():
    payload = budget_payload(items=[{"category": "a", "amount": "unknown"}])

    outcome = normalize.coerce_numbers("budget.items[].amount")(payload)

    assert not outcome.changed
    assert payload["budget"]["items"][0]["amount"] == "unknown"


def test_rebalance_pushes_the_discrepancy_into_the_slack_item():
    payload = budget_payload(
        items=[
            {"category": "lodging", "amount": 500.0},
            {"category": "food", "amount": 300.0},
            {"category": "contingency", "amount": 150.0},
        ]
    )

    outcome = normalize.rebalance_to_total(
        items="budget.items",
        amount_field="amount",
        total="budget.total",
        slack_match={"category": "contingency"},
    )(payload)

    assert outcome.changed
    assert total_of(payload) == Decimal("1000.00")
    assert payload["budget"]["items"][0]["amount"] == 500.0
    assert payload["budget"]["items"][2]["amount"] == 200.0


def test_rebalance_distributes_proportionally_without_a_slack_item():
    payload = budget_payload(
        items=[
            {"category": "a", "amount": 100.0},
            {"category": "b", "amount": 100.0},
            {"category": "c", "amount": 100.0},
        ],
        total=1000.0,
    )

    normalize.rebalance_to_total(items="budget.items", amount_field="amount", total="budget.total")(payload)

    assert total_of(payload) == Decimal("1000.00")


def test_rebalance_leaves_no_rounding_residual_on_an_indivisible_split():
    payload = budget_payload(
        items=[
            {"category": "a", "amount": 1.0},
            {"category": "b", "amount": 1.0},
            {"category": "c", "amount": 1.0},
        ],
        total=100.0,
    )

    normalize.rebalance_to_total(items="budget.items", amount_field="amount", total="budget.total")(payload)

    assert total_of(payload) == Decimal("100.00")


def test_rebalance_falls_back_to_proportional_when_slack_would_go_negative():
    payload = budget_payload(
        items=[
            {"category": "lodging", "amount": 900.0},
            {"category": "contingency", "amount": 50.0},
        ],
        total=500.0,
    )

    normalize.rebalance_to_total(
        items="budget.items",
        amount_field="amount",
        total="budget.total",
        slack_match={"category": "contingency"},
    )(payload)

    assert total_of(payload) == Decimal("500.00")
    assert all(item["amount"] >= 0 for item in payload["budget"]["items"])


def test_rebalance_is_a_no_op_when_already_exact():
    payload = budget_payload()

    outcome = normalize.rebalance_to_total(
        items="budget.items", amount_field="amount", total="budget.total"
    )(payload)

    assert not outcome.changed


def test_rebalance_declines_to_guess_at_non_numeric_amounts():
    payload = budget_payload(items=[{"category": "a", "amount": "lots"}])

    outcome = normalize.rebalance_to_total(
        items="budget.items", amount_field="amount", total="budget.total"
    )(payload)

    assert not outcome.changed
    assert "coerce_numbers" in (outcome.detail or "")


def test_drop_unknown_properties_removes_forbidden_keys():
    payload = budget_payload()
    payload["commentary"] = "I hope this helps!"
    payload["budget"]["items"][0]["reasoning"] = "because hotels are expensive"

    outcome = normalize.drop_unknown_properties(BUDGET_SCHEMA)(payload)

    assert outcome.changed
    assert "commentary" not in payload
    assert "reasoning" not in payload["budget"]["items"][0]


def test_drop_unknown_properties_keeps_declared_keys():
    payload = budget_payload()

    outcome = normalize.drop_unknown_properties(BUDGET_SCHEMA)(payload)

    assert not outcome.changed
    assert payload == budget_payload()


def test_round_amounts_quantizes_half_up():
    payload = budget_payload(items=[{"category": "a", "amount": 10.005}])

    normalize.round_amounts("budget.items[].amount")(payload)

    assert payload["budget"]["items"][0]["amount"] == 10.01


def test_deduplicate_drops_repeated_elements():
    payload = budget_payload(
        items=[
            {"category": "food", "amount": 1.0},
            {"category": "food", "amount": 2.0},
            {"category": "lodging", "amount": 3.0},
        ]
    )

    outcome = normalize.deduplicate("budget.items", key="category")(payload)

    assert outcome.changed
    assert [item["category"] for item in payload["budget"]["items"]] == ["food", "lodging"]
