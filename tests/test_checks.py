from proofmark import checks
from tests.conftest import budget_payload


def test_sums_to_passes_when_items_add_up():
    assert checks.sums_to("budget.items", "amount", "budget.total")(budget_payload()) == []


def test_sums_to_reports_the_discrepancy():
    payload = budget_payload(
        items=[
            {"category": "lodging", "amount": 500.0},
            {"category": "food", "amount": 300.0},
        ]
    )

    issues = checks.sums_to("budget.items", "amount", "budget.total")(payload)

    assert issues[0].code == "total_mismatch"
    assert "800" in issues[0].message
    assert "1000" in issues[0].message


def test_sums_to_tolerates_float_dust():
    payload = budget_payload(
        items=[
            {"category": "a", "amount": 333.33},
            {"category": "b", "amount": 333.33},
            {"category": "c", "amount": 333.34},
        ]
    )

    assert checks.sums_to("budget.items", "amount", "budget.total")(payload) == []


def test_sums_to_accepts_a_literal_total():
    payload = budget_payload()

    assert checks.sums_to("budget.items", "amount", 1000.0)(payload) == []
    assert checks.sums_to("budget.items", "amount", 900.0)(payload)


def test_sums_to_reports_non_numeric_amounts_clearly():
    payload = budget_payload(items=[{"category": "a", "amount": "lots"}])

    issues = checks.sums_to("budget.items", "amount", "budget.total")(payload)

    assert issues[0].code == "non_numeric_amount"


def test_no_placeholders_flags_filler_values():
    payload = budget_payload(title="TBD")

    issues = checks.no_placeholders()(payload)

    assert issues[0].code == "placeholder"
    assert issues[0].path == "$.title"


def test_no_placeholders_does_not_flag_prose_containing_the_word():
    payload = budget_payload(title="For example, the Alfama district is worth a morning.")

    assert checks.no_placeholders()(payload) == []


def test_no_placeholders_accepts_extra_vocabulary():
    payload = budget_payload(title="FIXME")

    assert checks.no_placeholders()(payload) == []
    assert checks.no_placeholders(extra=["fixme"])(payload)


def test_currency_consistent_flags_a_foreign_amount():
    payload = budget_payload()
    payload["title"] = "Lisbon on about $1,200 all in"

    issues = checks.currency_consistent("EUR")(payload)

    assert issues[0].code == "currency_mismatch"
    assert "USD" in issues[0].message


def test_currency_consistent_accepts_the_declared_currency():
    payload = budget_payload()
    payload["title"] = "Lisbon for EUR 1000"

    assert checks.currency_consistent("EUR")(payload) == []


def test_unique_flags_duplicate_keys():
    payload = budget_payload(
        items=[
            {"category": "food", "amount": 500.0},
            {"category": "food", "amount": 500.0},
        ]
    )

    issues = checks.unique("budget.items", key="category")

    assert issues(payload)[0].code == "duplicate"


def test_bounded_backs_up_a_schema_constraint_the_provider_may_ignore():
    payload = budget_payload(items=[{"category": "food", "amount": -50.0}])

    issues = checks.bounded("budget.items[].amount", minimum=0)(payload)

    assert issues[0].code == "below_min"
    assert issues[0].path == "$.budget.items[0].amount"


def test_non_empty_flags_blank_and_absent_values():
    payload = budget_payload(title="")

    assert checks.non_empty("title")(payload)[0].code == "empty"
    assert checks.non_empty("missing.field")(payload)[0].code == "empty"


def test_all_of_combines_checks():
    payload = budget_payload(title="TBD", items=[{"category": "a", "amount": 1.0}])
    combined = checks.all_of(
        checks.no_placeholders(),
        checks.sums_to("budget.items", "amount", "budget.total"),
    )

    codes = {item.code for item in combined(payload)}

    assert codes == {"placeholder", "total_mismatch"}
