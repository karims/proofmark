from proofmark.schema.preflight import GENERIC, OPENAI_STRICT, Profile, preflight


def test_valid_strict_schema_passes(schema):
    report = preflight(schema, OPENAI_STRICT)

    assert report.ok
    assert report.errors == []


def test_open_object_is_rejected_under_strict():
    schema = {
        "type": "object",
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
    }

    report = preflight(schema, OPENAI_STRICT)

    assert not report.ok
    assert any(item.code == "open_object" for item in report.errors)


def test_open_object_is_fine_under_generic():
    schema = {
        "type": "object",
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
    }

    assert preflight(schema, GENERIC).ok


def test_optional_property_is_rejected_under_strict():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name"],
        "properties": {"name": {"type": "string"}, "nickname": {"type": "string"}},
    }

    report = preflight(schema, OPENAI_STRICT)

    assert any(item.code == "optional_property" for item in report.errors)
    assert "nickname" in report.errors[0].message


def test_required_naming_unknown_property_is_rejected():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "ghost"],
        "properties": {"name": {"type": "string"}},
    }

    report = preflight(schema, OPENAI_STRICT)

    assert any(item.code == "required_unknown" for item in report.errors)


def test_const_contradicting_its_type_is_rejected():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind"],
        "properties": {"kind": {"type": "string", "const": 7}},
    }

    report = preflight(schema, OPENAI_STRICT)

    assert any(item.code == "const_type_mismatch" for item in report.errors)


def test_missing_type_is_rejected():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["mystery"],
        "properties": {"mystery": {"description": "no type here"}},
    }

    report = preflight(schema, OPENAI_STRICT)

    assert any(item.code == "missing_type" for item in report.errors)


def test_array_without_items_is_rejected():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tags"],
        "properties": {"tags": {"type": "array"}},
    }

    report = preflight(schema, OPENAI_STRICT)

    assert any(item.code == "array_without_items" for item in report.errors)


def test_unresolved_ref_is_rejected():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["child"],
        "properties": {"child": {"$ref": "#/$defs/Missing"}},
    }

    report = preflight(schema, OPENAI_STRICT)

    assert any(item.code == "unresolved_ref" for item in report.errors)


def test_resolvable_ref_passes():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["child"],
        "properties": {"child": {"$ref": "#/$defs/Child"}},
        "$defs": {
            "Child": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            }
        },
    }

    assert preflight(schema, OPENAI_STRICT).ok


def test_advisory_keywords_warn_but_do_not_block():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["amount"],
        "properties": {"amount": {"type": "number", "minimum": 0}},
    }

    report = preflight(schema, OPENAI_STRICT)

    assert report.ok
    assert any(item.code == "advisory_keyword" for item in report.warnings)


def test_depth_limit_is_configurable():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["a"],
        "properties": {
            "a": {
                "type": "object",
                "additionalProperties": False,
                "required": ["b"],
                "properties": {"b": {"type": "string"}},
            }
        },
    }
    shallow = Profile(name="shallow", max_depth=1)

    assert preflight(schema, shallow).errors
    assert preflight(schema, Profile(name="deep", max_depth=9)).ok


def test_property_budget_counts_the_whole_schema(schema):
    tiny = Profile(name="tiny", max_properties=2)

    report = preflight(schema, tiny)

    assert any(item.code == "too_many_properties" for item in report.errors)
