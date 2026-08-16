import pytest

BUDGET_SCHEMA = {
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


def budget_payload(items=None, total=1000.0, currency="EUR", title="Lisbon"):
    if items is None:
        items = [
            {"category": "lodging", "amount": 500.0},
            {"category": "food", "amount": 300.0},
            {"category": "contingency", "amount": 200.0},
        ]
    return {
        "title": title,
        "currency": currency,
        "budget": {"total": total, "items": items},
    }


@pytest.fixture
def schema():
    return BUDGET_SCHEMA


@pytest.fixture
def payload():
    return budget_payload()
