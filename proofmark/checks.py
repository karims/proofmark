"""Semantic checks -- the part schema validation cannot do.

A payload can satisfy its schema completely and still be wrong: a budget whose
line items do not sum to its stated total, a plan with two "day 3" entries, a
field left as ``"TBD"``, an amount in the wrong currency. Every field has the right
type; the document is still unusable.

A check is any ``Callable[[dict], list[Issue]]``. The built-ins below cover the
recurring cases; anything domain-specific is a plain function.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable

from proofmark import paths
from proofmark.issues import Issue, Stage, issue

Check = Callable[[dict], list[Issue]]

DEFAULT_PLACEHOLDERS: tuple[str, ...] = (
    "tbd",
    "todo",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "xxx",
    "lorem ipsum",
    "insert here",
    "your text here",
    "placeholder",
    "example",
    "...",
)

_CURRENCY_CODE = re.compile(r"\b([A-Z]{3})\s?[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s?([A-Z]{3})\b")
_CURRENCY_SYMBOL = re.compile(r"([$£€¥₹])\s?[\d,]+")
_SYMBOL_TO_CODE = {"$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY", "₹": "INR"}


def sums_to(
    items: str,
    amount_field: str,
    total: str | float,
    tolerance: float = 0.01,
    label: str | None = None,
) -> Check:
    """Line items must add up to a stated total.

    The single most common way a schema-valid document is still wrong. ``total``
    may be a path into the payload or a literal the caller already knows.
    """

    def check(data: dict) -> list[Issue]:
        item_matches = paths.resolve(data, f"{items}[].{amount_field}")
        if not item_matches:
            return []

        expected = _expected_total(data, total)
        if expected is None:
            return []

        actual = Decimal("0")
        for match in item_matches:
            amount = _to_decimal(match.value)
            if amount is None:
                return [
                    issue(
                        match.path,
                        f"amount {match.value!r} is not numeric, so the total cannot be checked",
                        Stage.SEMANTIC,
                        "non_numeric_amount",
                    )
                ]
            actual += amount

        if abs(actual - expected) <= Decimal(str(tolerance)):
            return []

        name = label or items
        return [
            issue(
                items,
                f"{name} line items sum to {actual}, but the stated total is {expected} "
                f"(off by {actual - expected})",
                Stage.SEMANTIC,
                "total_mismatch",
            )
        ]

    return check


def no_placeholders(
    targets: str | Iterable[str] | None = None,
    extra: Iterable[str] = (),
) -> Check:
    """Reject leftover filler text.

    With no ``targets``, every string in the payload is scanned. Matching is on the
    whole trimmed value, not substrings, so a sentence mentioning "for example" is
    not flagged while a field containing only ``"example"`` is.
    """
    vocabulary = {word.lower() for word in (*DEFAULT_PLACEHOLDERS, *extra)}

    def check(data: dict) -> list[Issue]:
        if targets is None:
            matches = paths.walk_strings(data)
        else:
            selected = [targets] if isinstance(targets, str) else list(targets)
            matches = [match for path in selected for match in paths.resolve(data, path)]

        found: list[Issue] = []
        for match in matches:
            if not isinstance(match.value, str):
                continue
            normalized = match.value.strip().lower().rstrip(".")
            if normalized in vocabulary or (not normalized and targets is not None):
                found.append(
                    issue(
                        match.path,
                        f"placeholder value {match.value!r} was left in the output",
                        Stage.SEMANTIC,
                        "placeholder",
                    )
                )
        return found

    return check


def non_empty(targets: str | Iterable[str]) -> Check:
    """Required text fields must actually contain text."""
    selected = [targets] if isinstance(targets, str) else list(targets)

    def check(data: dict) -> list[Issue]:
        found: list[Issue] = []
        for path in selected:
            matches = paths.resolve(data, path)
            if not matches:
                found.append(issue(path, "expected a value here, found nothing", Stage.SEMANTIC, "empty"))
                continue
            for match in matches:
                if match.value is None or (isinstance(match.value, (str, list, dict)) and not match.value):
                    found.append(issue(match.path, "value is empty", Stage.SEMANTIC, "empty"))
        return found

    return check


def unique(target: str, key: str | None = None) -> Check:
    """No duplicates among a list's values, or among one field of its elements."""

    def check(data: dict) -> list[Issue]:
        path = f"{target}[].{key}" if key else f"{target}[]"
        matches = paths.resolve(data, path)
        seen: dict[str, str] = {}
        found: list[Issue] = []
        for match in matches:
            marker = repr(match.value)
            if marker in seen:
                found.append(
                    issue(
                        match.path,
                        f"duplicate value {match.value!r} (first seen at {seen[marker]})",
                        Stage.SEMANTIC,
                        "duplicate",
                    )
                )
            else:
                seen[marker] = match.path
        return found

    return check


def bounded(target: str, minimum: float | None = None, maximum: float | None = None) -> Check:
    """Range check as a *semantic* rule.

    Worth stating explicitly even when the schema already declares ``minimum``:
    strict structured-output modes may ignore numeric constraints, which preflight
    warns about. This is the check that makes the warning actionable.
    """

    def check(data: dict) -> list[Issue]:
        found: list[Issue] = []
        for match in paths.resolve(data, target):
            amount = _to_decimal(match.value)
            if amount is None:
                continue
            if minimum is not None and amount < Decimal(str(minimum)):
                found.append(
                    issue(match.path, f"{amount} is below the minimum of {minimum}", Stage.SEMANTIC, "below_min")
                )
            if maximum is not None and amount > Decimal(str(maximum)):
                found.append(
                    issue(match.path, f"{amount} exceeds the maximum of {maximum}", Stage.SEMANTIC, "above_max")
                )
        return found

    return check


def currency_consistent(expected: str, targets: str | Iterable[str] | None = None) -> Check:
    """Every monetary amount mentioned in prose must use the expected currency.

    Models routinely narrate a budget in dollars after being asked for euros. The
    schema cannot see it, because the field is a string either way.
    """
    expected_code = expected.upper()

    def check(data: dict) -> list[Issue]:
        if targets is None:
            matches = paths.walk_strings(data)
        else:
            selected = [targets] if isinstance(targets, str) else list(targets)
            matches = [match for path in selected for match in paths.resolve(data, path)]

        found: list[Issue] = []
        for match in matches:
            if not isinstance(match.value, str):
                continue
            for code in _currencies_in(match.value):
                if code != expected_code:
                    found.append(
                        issue(
                            match.path,
                            f"mentions an amount in {code} but the document currency is {expected_code}",
                            Stage.SEMANTIC,
                            "currency_mismatch",
                        )
                    )
                    break
        return found

    return check


def all_of(*checks: Check) -> Check:
    """Combine checks into one. Useful for passing a single named bundle around."""

    def check(data: dict) -> list[Issue]:
        return [found for single in checks for found in single(data)]

    return check


def _currencies_in(text: str) -> set[str]:
    codes: set[str] = set()
    for match in _CURRENCY_CODE.finditer(text):
        code = match.group(1) or match.group(2)
        if code:
            codes.add(code)
    for match in _CURRENCY_SYMBOL.finditer(text):
        mapped = _SYMBOL_TO_CODE.get(match.group(1))
        if mapped:
            codes.add(mapped)
    return codes


def _expected_total(data: dict, total: str | float) -> Decimal | None:
    if isinstance(total, str):
        return _to_decimal(paths.first(data, total))
    return _to_decimal(total)


def _to_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").lstrip("$£€¥₹").strip()
        try:
            return Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return None
    return None
