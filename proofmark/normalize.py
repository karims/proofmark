"""Deterministic repair -- fix it in code before asking the model again.

A second model call is slow, costs money, and may return something new to be wrong
about. Most validation failures do not need one: a total that is off by two cents,
an amount returned as ``"1,200.00"``, a stray property the schema forbids. These
are arithmetic and parsing problems, and code fixes them exactly.

Normalizers run before any repair round. Only what survives them is worth a call.

A normalizer is ``Callable[[dict], Normalization]``. It mutates the payload in
place and reports whether it changed anything.
"""

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable, Iterable

from proofmark import paths

DEFAULT_QUANTUM = Decimal("0.01")


@dataclass
class Normalization:
    """What a normalizer did, for the trace and the result summary."""

    name: str
    changed: bool = False
    detail: str | None = None


Normalizer = Callable[[dict], Normalization]


def coerce_numbers(targets: str | Iterable[str]) -> Normalizer:
    """Turn numeric-looking strings into numbers.

    ``"1,200.00"``, ``"$1200"``, ``" 1200 "`` all become ``1200.0``. Models emit
    these constantly under json mode, where nothing enforces the declared type.
    Run this before any arithmetic normalizer.
    """
    selected = [targets] if isinstance(targets, str) else list(targets)

    def normalize(data: dict) -> Normalization:
        changed: list[str] = []
        for path in selected:
            for match in paths.resolve(data, path):
                if not isinstance(match.value, str):
                    continue
                amount = _to_decimal(match.value)
                if amount is None:
                    continue
                match.set(float(amount))
                changed.append(match.path)
        return Normalization(
            name="coerce_numbers",
            changed=bool(changed),
            detail=f"coerced {len(changed)} value(s)" if changed else None,
        )

    return normalize


def round_amounts(targets: str | Iterable[str], quantum: str | Decimal = DEFAULT_QUANTUM) -> Normalizer:
    """Quantize numbers to a fixed precision, half-up."""
    selected = [targets] if isinstance(targets, str) else list(targets)
    step = Decimal(str(quantum))

    def normalize(data: dict) -> Normalization:
        changed: list[str] = []
        for path in selected:
            for match in paths.resolve(data, path):
                amount = _to_decimal(match.value)
                if amount is None:
                    continue
                rounded = amount.quantize(step, rounding=ROUND_HALF_UP)
                if rounded != amount or not isinstance(match.value, (int, float)):
                    match.set(float(rounded))
                    changed.append(match.path)
        return Normalization(
            name="round_amounts",
            changed=bool(changed),
            detail=f"rounded {len(changed)} value(s)" if changed else None,
        )

    return normalize


def rebalance_to_total(
    items: str,
    amount_field: str,
    total: str | float,
    slack_match: dict | None = None,
    quantum: str | Decimal = DEFAULT_QUANTUM,
    allow_negative: bool = False,
) -> Normalizer:
    """Force line items to sum exactly to their stated total.

    Two strategies, in order:

    1. If ``slack_match`` identifies an absorbing item (say
       ``{"category": "contingency"}``), the whole discrepancy is pushed there.
       This is what a human would do, and it leaves every other figure untouched.
    2. Otherwise the total is redistributed across all items in proportion to their
       current values, using the largest-remainder method so the quantized parts
       sum to the target exactly rather than drifting by a cent per item.

    Arithmetic is in :class:`~decimal.Decimal` throughout. Doing this in floats is
    how you end up with a "corrected" total that is still off by 0.000000001.
    """
    step = Decimal(str(quantum))

    def normalize(data: dict) -> Normalization:
        matches = paths.resolve(data, f"{items}[].{amount_field}")
        if not matches:
            return Normalization(name="rebalance_to_total")

        target = _resolve_total(data, total)
        if target is None:
            return Normalization(name="rebalance_to_total")
        target = target.quantize(step, rounding=ROUND_HALF_UP)

        amounts: list[Decimal] = []
        for match in matches:
            amount = _to_decimal(match.value)
            if amount is None:
                return Normalization(
                    name="rebalance_to_total",
                    detail="skipped: a non-numeric amount is present (run coerce_numbers first)",
                )
            amounts.append(amount.quantize(step, rounding=ROUND_HALF_UP))

        original = list(amounts)
        difference = target - sum(amounts)

        if difference != 0 and slack_match is not None:
            index = _find_slack(data, items, slack_match)
            if index is not None and 0 <= index < len(amounts):
                adjusted = amounts[index] + difference
                if allow_negative or adjusted >= 0:
                    amounts[index] = adjusted
                    difference = Decimal("0")

        if difference != 0:
            amounts = _largest_remainder(target, amounts, step)

        if amounts == original and all(isinstance(match.value, (int, float)) for match in matches):
            return Normalization(name="rebalance_to_total")

        for match, amount in zip(matches, amounts):
            match.set(float(amount))

        return Normalization(
            name="rebalance_to_total",
            changed=True,
            detail=f"adjusted {items} to sum to {target}",
        )

    return normalize


def deduplicate(target: str, key: str | None = None) -> Normalizer:
    """Drop later list elements that repeat an earlier one."""

    def normalize(data: dict) -> Normalization:
        containers = paths.resolve(data, target)
        removed = 0
        for container in containers:
            if not isinstance(container.value, list):
                continue
            seen: set[str] = set()
            kept: list[Any] = []
            for element in container.value:
                marker = repr(element.get(key) if key and isinstance(element, dict) else element)
                if marker in seen:
                    removed += 1
                    continue
                seen.add(marker)
                kept.append(element)
            if removed:
                container.set(kept)
        return Normalization(
            name="deduplicate",
            changed=bool(removed),
            detail=f"removed {removed} duplicate(s)" if removed else None,
        )

    return normalize


def drop_unknown_properties(schema: dict) -> Normalizer:
    """Remove properties the schema forbids.

    Strict structured output should make this impossible, but json mode and text
    tiers routinely add a chatty ``"notes"`` or ``"explanation"`` key. Deleting it
    is exact and needs no model call -- a textbook case for fixing in code.
    """

    def normalize(data: dict) -> Normalization:
        removed: list[str] = []
        _prune(data, schema, schema, "$", removed)
        return Normalization(
            name="drop_unknown_properties",
            changed=bool(removed),
            detail=f"removed {sorted(removed)}" if removed else None,
        )

    return normalize


def _prune(node: Any, schema: Any, root: dict, path: str, removed: list[str]) -> None:
    if not isinstance(schema, dict):
        return
    schema = _deref(schema, root)
    if schema is None:
        return

    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            for branch in branches:
                resolved = _deref(branch, root) or {}
                if isinstance(node, dict) and resolved.get("type") == "object":
                    _prune(node, resolved, root, path, removed)
                    return

    if isinstance(node, dict) and schema.get("type") == "object":
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        if schema.get("additionalProperties") is False:
            for name in [key for key in node if key not in properties]:
                del node[name]
                removed.append(f"{path}.{name}")
        for name, subschema in properties.items():
            if name in node:
                _prune(node[name], subschema, root, f"{path}.{name}", removed)
        return

    if isinstance(node, list) and schema.get("type") == "array":
        items = schema.get("items")
        for index, element in enumerate(node):
            _prune(element, items, root, f"{path}[{index}]", removed)


def _deref(schema: dict, root: dict) -> dict | None:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    target: Any = root
    for segment in ref[2:].split("/"):
        if not isinstance(target, dict) or segment not in target:
            return None
        target = target[segment]
    return target if isinstance(target, dict) else None


def _largest_remainder(target: Decimal, amounts: list[Decimal], step: Decimal) -> list[Decimal]:
    """Redistribute ``target`` across ``amounts`` so the parts sum to it exactly.

    Proportional split, floored to the quantum, with the leftover units handed out
    by descending fractional remainder -- the standard apportionment method. Naive
    rounding leaves a residual of up to half a unit per item; this leaves none.
    """
    count = len(amounts)
    if count == 0:
        return amounts

    weight_total = sum(amounts)
    if weight_total > 0:
        raw = [target * amount / weight_total for amount in amounts]
    else:
        raw = [target / count] * count

    result = [value.quantize(step, rounding=ROUND_DOWN) for value in raw]
    remainders = [raw[index] - result[index] for index in range(count)]

    residual = target - sum(result)
    units = int((residual / step).to_integral_value(rounding=ROUND_HALF_UP))

    if units > 0:
        order = sorted(range(count), key=lambda index: remainders[index], reverse=True)
        for position in range(units):
            result[order[position % count]] += step
    elif units < 0:
        order = sorted(range(count), key=lambda index: remainders[index])
        for position in range(-units):
            index = order[position % count]
            if result[index] >= step:
                result[index] -= step

    return result


def _find_slack(data: dict, items: str, slack_match: dict) -> int | None:
    containers = paths.resolve(data, items)
    for container in containers:
        if not isinstance(container.value, list):
            continue
        for index, element in enumerate(container.value):
            if isinstance(element, dict) and all(
                str(element.get(key, "")).lower() == str(value).lower() for key, value in slack_match.items()
            ):
                return index
    return None


def _resolve_total(data: dict, total: str | float) -> Decimal | None:
    if isinstance(total, str):
        return _to_decimal(paths.first(data, total))
    return _to_decimal(total)


def _to_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").lstrip("$£€¥₹").strip()
        try:
            return Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return None
    return None
