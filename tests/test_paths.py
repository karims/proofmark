from proofmark import paths
from tests.conftest import budget_payload


def test_resolves_a_simple_field():
    assert paths.first(budget_payload(), "budget.total") == 1000.0


def test_resolves_a_list_wildcard():
    values = paths.values(budget_payload(), "budget.items[].category")

    assert values == ["lodging", "food", "contingency"]


def test_resolves_a_specific_index():
    assert paths.first(budget_payload(), "budget.items[1].category") == "food"


def test_negative_index_reads_from_the_end():
    assert paths.first(budget_payload(), "budget.items[-1].category") == "contingency"


def test_missing_path_returns_nothing_rather_than_raising():
    assert paths.resolve(budget_payload(), "budget.nonexistent.deeply") == []
    assert paths.first(budget_payload(), "nope", default="fallback") == "fallback"


def test_match_can_write_back():
    payload = budget_payload()

    paths.resolve(payload, "budget.items[0].amount")[0].set(42.0)

    assert payload["budget"]["items"][0]["amount"] == 42.0


def test_match_path_is_reported_with_indices():
    matches = paths.resolve(budget_payload(), "budget.items[].amount")

    assert [match.path for match in matches] == [
        "$.budget.items[0].amount",
        "$.budget.items[1].amount",
        "$.budget.items[2].amount",
    ]


def test_walk_strings_finds_every_string_with_its_location():
    matches = paths.walk_strings({"a": "x", "b": [{"c": "y"}], "n": 1})

    assert {match.path: match.value for match in matches} == {"$.a": "x", "$.b[0].c": "y"}
