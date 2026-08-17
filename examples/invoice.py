"""Invoice extraction -- the case proofmark was built for.

    python examples/invoice.py

Runs offline.

Pulling structured data out of documents is where a wrong number is expensive and
nobody is reading the JSON before it hits a ledger. It is also where schema
conformance is worth the least: every field on a bad invoice extraction has the
right type. The document is wrong in *arithmetic*, and arithmetic is exactly what
a language model is worst at and what code is best at.

The lesson this example is built around:

    Normalize what is DERIVABLE. Check what is EVIDENCE.

``line_total`` is derivable -- it is ``quantity x unit_price`` and nothing else, so
recompute it in code and never bother the model. ``subtotal`` is evidence: it is
printed on the document, and if it disagrees with the line items that most likely
means a row was *missed* during extraction. Silently "fixing" the subtotal to match
the rows you found would erase the only signal that a row is missing.

That distinction is a data-loss bug in most extraction pipelines. Here it is the
difference between a normalizer and a check.

Three scenarios run below:

    1. Derivable arithmetic wrong  -> fixed in code, one call.
    2. A line item dropped         -> NOT auto-fixed; repaired by the model.
    3. Still inconsistent          -> fails with a category, for human review.
"""

import json
from datetime import date
from decimal import Decimal

from proofmark import (
    Failure,
    Issue,
    Outcome,
    Stage,
    StaticProvider,
    checks,
    compose,
    normalize,
    paths,
)

CENTS = Decimal("0.01")

INVOICE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "invoice_number",
        "invoice_date",
        "due_date",
        "currency",
        "vendor",
        "line_items",
        "subtotal",
        "tax_rate",
        "tax_amount",
        "total",
    ],
    "properties": {
        "invoice_number": {"type": "string"},
        "invoice_date": {"type": "string"},
        "due_date": {"type": "string"},
        "currency": {"type": "string"},
        "vendor": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "tax_id"],
            "properties": {
                "name": {"type": "string"},
                "tax_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        },
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["description", "quantity", "unit_price", "line_total"],
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"},
                    "line_total": {"type": "number"},
                },
            },
        },
        "subtotal": {"type": "number"},
        "tax_rate": {"type": "number"},
        "tax_amount": {"type": "number"},
        "total": {"type": "number"},
    },
}


# --- normalizers: only for values that are DERIVABLE ------------------------


def recompute_line_totals(data: dict) -> normalize.Normalization:
    """``line_total = quantity x unit_price``. Pure arithmetic, so do it in code."""
    changed: list[str] = []
    for match in paths.resolve(data, "line_items[]"):
        item = match.value
        if not isinstance(item, dict):
            continue
        quantity, price = _decimal(item.get("quantity")), _decimal(item.get("unit_price"))
        if quantity is None or price is None:
            continue
        expected = (quantity * price).quantize(CENTS)
        if _decimal(item.get("line_total")) != expected:
            item["line_total"] = float(expected)
            changed.append(match.path)
    return normalize.Normalization(
        name="recompute_line_totals",
        changed=bool(changed),
        detail=f"recomputed {len(changed)} line total(s)" if changed else None,
    )


def recompute_tax_and_total(data: dict) -> normalize.Normalization:
    """``tax_amount`` and ``total`` follow from subtotal and rate.

    Note what this deliberately does *not* touch: ``subtotal``. See the module
    docstring -- that one is evidence, not a derived value.
    """
    subtotal = _decimal(data.get("subtotal"))
    rate = _decimal(data.get("tax_rate"))
    if subtotal is None or rate is None:
        return normalize.Normalization(name="recompute_tax_and_total")

    changed: list[str] = []
    tax = (subtotal * rate).quantize(CENTS)
    if _decimal(data.get("tax_amount")) != tax:
        data["tax_amount"] = float(tax)
        changed.append("tax_amount")

    total = (subtotal + tax).quantize(CENTS)
    if _decimal(data.get("total")) != total:
        data["total"] = float(total)
        changed.append("total")

    return normalize.Normalization(
        name="recompute_tax_and_total",
        changed=bool(changed),
        detail=f"recomputed {changed}" if changed else None,
    )


# --- checks: for values that are EVIDENCE -----------------------------------


def subtotal_matches_line_items(data: dict) -> list[Issue]:
    """The printed subtotal must equal the rows we extracted.

    A mismatch is the single most valuable signal in invoice extraction: it usually
    means a line item was missed. Recomputing the subtotal to match would destroy
    that signal and produce a confidently wrong invoice.
    """
    stated = _decimal(data.get("subtotal"))
    if stated is None:
        return []
    rows = [_decimal(value) for value in paths.values(data, "line_items[].line_total")]
    if any(value is None for value in rows):
        return []
    extracted = sum(rows, Decimal("0")).quantize(CENTS)
    if extracted == stated:
        return []
    return [
        Issue(
            path="$.subtotal",
            message=(
                f"document states a subtotal of {stated} but the {len(rows)} extracted "
                f"line items sum to {extracted} (short by {stated - extracted}). "
                "A line item was probably missed -- re-read the table."
            ),
            stage=Stage.SEMANTIC,
            code="subtotal_mismatch",
        )
    ]


def due_date_is_not_before_invoice_date(data: dict) -> list[Issue]:
    try:
        issued = date.fromisoformat(str(data.get("invoice_date")))
        due = date.fromisoformat(str(data.get("due_date")))
    except ValueError:
        return [
            Issue(
                path="$.due_date",
                message="invoice_date and due_date must both be ISO 8601 (YYYY-MM-DD)",
                stage=Stage.SEMANTIC,
                code="bad_date",
            )
        ]
    if due < issued:
        return [
            Issue(
                path="$.due_date",
                message=f"due date {due} precedes the invoice date {issued}",
                stage=Stage.SEMANTIC,
                code="due_before_issue",
            )
        ]
    return []


def _decimal(value) -> Decimal | None:
    """Exact conversion, deliberately NOT quantized.

    Rounding belongs at the money boundary, not on every read. Quantizing here would
    turn a ``tax_rate`` of 0.0875 into 0.09 and quietly inflate every invoice.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    return None


NORMALIZERS = [
    normalize.drop_unknown_properties(INVOICE_SCHEMA),
    normalize.coerce_numbers(
        [
            "line_items[].quantity",
            "line_items[].unit_price",
            "line_items[].line_total",
            "subtotal",
            "tax_rate",
            "tax_amount",
            "total",
        ]
    ),
    recompute_line_totals,
    recompute_tax_and_total,
]

CHECKS = [
    subtotal_matches_line_items,
    due_date_is_not_before_invoice_date,
    checks.bounded("line_items[].quantity", minimum=0),
    checks.bounded("tax_rate", minimum=0, maximum=1),
    checks.currency_consistent("USD"),
    checks.no_placeholders(["invoice_number", "vendor.name"]),
]

PROMPT = "Extract the invoice in the attached scan as JSON."


def base_invoice(**overrides) -> dict:
    invoice = {
        "invoice_number": "INV-2044",
        "invoice_date": "2026-07-02",
        "due_date": "2026-08-01",
        "currency": "USD",
        "vendor": {"name": "Northwind Fabrication", "tax_id": "US-88-4410912"},
        "line_items": [
            {"description": "CNC bracket, 6061-T6", "quantity": 120, "unit_price": 18.50, "line_total": 2220.00},
            {"description": "Anodizing, clear", "quantity": 120, "unit_price": 3.25, "line_total": 390.00},
            {"description": "Tooling setup", "quantity": 1, "unit_price": 450.00, "line_total": 450.00},
        ],
        "subtotal": 3060.00,
        "tax_rate": 0.0875,
        "tax_amount": 267.75,
        "total": 3327.75,
    }
    invoice.update(overrides)
    return invoice


def run(label: str, responses: list, **kwargs):
    print("=" * 74)
    print(label)
    print("=" * 74)

    provider = StaticProvider(responses=responses)
    result = compose(
        prompt=PROMPT,
        schema=INVOICE_SCHEMA,
        provider=provider,
        normalizers=NORMALIZERS,
        checks=CHECKS,
        **kwargs,
    )

    print(f"result: {result.summary()}")
    if result.normalizations:
        print(f"fixed in code: {', '.join(result.normalizations)}")
    if result.initial_issues:
        print("issues the model had to answer for:")
        for problem in result.initial_issues:
            print(f"    {problem.format()}")
    if result.issues:
        print("UNRESOLVED:")
        for problem in result.issues:
            print(f"    {problem.format()}")
    print()
    return result


def main() -> None:
    # 1. Derivable arithmetic is wrong: the model multiplied 120 x 18.50 as 2200,
    #    and the tax follows from a subtotal it never checked. All recomputable.
    wrong_arithmetic = base_invoice()
    wrong_arithmetic["line_items"][0]["line_total"] = 2200.00
    wrong_arithmetic["tax_amount"] = 260.00
    wrong_arithmetic["total"] = 3320.00
    wrong_arithmetic["extraction_notes"] = "Table was slightly skewed in the scan."

    first = run("1. DERIVABLE ARITHMETIC WRONG -- fixed in code, one call", [wrong_arithmetic])
    assert first.ok and first.outcome is Outcome.NORMALIZED
    assert first.data["line_items"][0]["line_total"] == 2220.00
    assert first.data["total"] == 3327.75
    assert "extraction_notes" not in first.data
    print("   The model got three numbers wrong and cost one call to fix none of them.")
    print("   No repair round: arithmetic does not need a language model.\n")

    # 2. A line item was missed. The subtotal is evidence that says so, and no
    #    normalizer touches it -- the discrepancy survives to become a repair prompt.
    dropped_row = base_invoice()
    dropped_row["line_items"] = dropped_row["line_items"][:2]

    second = run(
        "2. LINE ITEM DROPPED -- detected, not papered over",
        [dropped_row, base_invoice()],
        repair_attempts=1,
    )
    assert second.ok and second.outcome is Outcome.REPAIRED
    assert len(second.data["line_items"]) == 3
    print("   Had subtotal been a normalizer instead of a check, this would have")
    print("   returned a tidy, internally consistent, WRONG invoice for $2,610.\n")

    # 3. The model cannot reconcile it. Better to fail loudly than to guess.
    third = run(
        "3. IRRECONCILABLE -- fails with a category, for human review",
        [dropped_row, dropped_row],
        repair_attempts=1,
    )
    assert not third.ok
    assert third.failure is Failure.SEMANTIC_INVALID
    print(f"   route to review queue: failure={third.failure.value}, "
          f"{len(third.issues)} issue(s), trace has {len(third.trace['events'])} events")
    print("   No payload is returned. An invoice you cannot verify is worse than none.\n")

    print("=" * 74)
    print("Final invoice from scenario 2:")
    print("=" * 74)
    print(json.dumps(second.data, indent=2))


if __name__ == "__main__":
    main()
