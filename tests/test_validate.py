from proofmark.schema.validate import validate_instance
from tests.conftest import budget_payload


def test_valid_payload_produces_no_issues(schema):
    assert validate_instance(budget_payload(), schema) == []


def test_missing_required_property_is_reported(schema):
    payload = budget_payload()
    del payload["currency"]

    issues = validate_instance(payload, schema)

    assert [item.code for item in issues] == ["missing_required"]
    assert issues[0].path == "$.currency"


def test_additional_property_is_reported(schema):
    payload = budget_payload()
    payload["commentary"] = "here is my reasoning"

    issues = validate_instance(payload, schema)

    assert [item.code for item in issues] == ["additional_property"]
    assert issues[0].path == "$.commentary"


def test_type_mismatch_is_reported_with_a_useful_path(schema):
    payload = budget_payload()
    payload["budget"]["items"][1]["amount"] = "300.00"

    issues = validate_instance(payload, schema)

    assert issues[0].code == "type_mismatch"
    assert issues[0].path == "$.budget.items[1].amount"
    assert "string" in issues[0].message


def test_all_issues_are_collected_not_just_the_first(schema):
    payload = budget_payload()
    del payload["title"]
    payload["budget"]["items"][0]["amount"] = "oops"
    payload["extra"] = 1

    issues = validate_instance(payload, schema)

    assert len(issues) == 3


def test_booleans_are_not_accepted_as_numbers():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}

    issues = validate_instance({"n": True}, schema)

    assert issues[0].code == "type_mismatch"


def test_integral_float_satisfies_integer():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}

    assert validate_instance({"n": 3.0}, schema) == []
    assert validate_instance({"n": 3.5}, schema)[0].code == "type_mismatch"


def test_nullable_union_accepts_both_branches():
    schema = {
        "type": "object",
        "required": ["note"],
        "properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
    }

    assert validate_instance({"note": "hello"}, schema) == []
    assert validate_instance({"note": None}, schema) == []
    assert validate_instance({"note": 5}, schema)


def test_enum_and_const_are_enforced():
    schema = {
        "type": "object",
        "required": ["kind", "version"],
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "version": {"const": 1},
        },
    }

    assert validate_instance({"kind": "a", "version": 1}, schema) == []
    assert validate_instance({"kind": "c", "version": 1}, schema)[0].code == "enum_mismatch"
    assert validate_instance({"kind": "a", "version": 2}, schema)[0].code == "const_mismatch"


def test_local_ref_is_followed():
    schema = {
        "type": "object",
        "required": ["child"],
        "properties": {"child": {"$ref": "#/$defs/Child"}},
        "$defs": {
            "Child": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            }
        },
    }

    assert validate_instance({"child": {"name": "x"}}, schema) == []
    assert validate_instance({"child": {}}, schema)[0].code == "missing_required"


def test_numeric_and_length_constraints():
    schema = {
        "type": "object",
        "required": ["n", "s", "xs"],
        "properties": {
            "n": {"type": "number", "minimum": 0, "maximum": 10},
            "s": {"type": "string", "minLength": 2},
            "xs": {"type": "array", "items": {"type": "number"}, "minItems": 1},
        },
    }

    assert validate_instance({"n": 5, "s": "ok", "xs": [1]}, schema) == []
    codes = {item.code for item in validate_instance({"n": 99, "s": "x", "xs": []}, schema)}
    assert codes == {"maximum", "min_length", "min_items"}


def test_array_element_paths_are_indexed():
    schema = {
        "type": "object",
        "required": ["xs"],
        "properties": {"xs": {"type": "array", "items": {"type": "number"}}},
    }

    issues = validate_instance({"xs": [1, "two", 3]}, schema)

    assert issues[0].path == "$.xs[1]"
