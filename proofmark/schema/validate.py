"""A focused JSON Schema validator.

Deliberately a subset, not a conformant implementation. It covers the constructs
that :mod:`proofmark.schema.preflight` permits -- which is to say, the constructs
providers actually support in structured-output mode. Anything exotic enough to be
missing here would have been rejected by the provider anyway.

Keeping this in-tree rather than depending on ``jsonschema`` is what lets proofmark
install with zero required dependencies. If you need full Draft 2020-12 semantics,
pass your own structural checker to :func:`proofmark.compose`.
"""

import re
from typing import Any

from proofmark.issues import Issue, Stage, issue

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def validate_instance(instance: Any, schema: dict, root: dict | None = None) -> list[Issue]:
    """Validate ``instance`` against ``schema``, returning every issue found.

    Collects all issues rather than stopping at the first, because the whole list
    is what gets handed to a repair call. Reporting one error per round trip would
    turn a single repair into five.
    """
    issues: list[Issue] = []
    _validate(instance, schema, root if root is not None else schema, "$", issues)
    return issues


def _validate(node: Any, schema: Any, root: dict, path: str, issues: list[Issue]) -> None:
    if schema is True or schema == {}:
        return
    if schema is False:
        issues.append(issue(path, "schema forbids any value here", Stage.STRUCTURAL, "forbidden"))
        return
    if not isinstance(schema, dict):
        return

    resolved = _resolve(schema, root, path, issues)
    if resolved is None:
        return
    schema = resolved

    if "const" in schema:
        if node != schema["const"]:
            issues.append(
                issue(
                    path,
                    f"expected constant {schema['const']!r}, got {_brief(node)}",
                    Stage.STRUCTURAL,
                    "const_mismatch",
                )
            )
        return

    if "enum" in schema:
        if node not in schema["enum"]:
            issues.append(
                issue(
                    path,
                    f"{_brief(node)} is not one of {_brief_list(schema['enum'])}",
                    Stage.STRUCTURAL,
                    "enum_mismatch",
                )
            )
        return

    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            _validate_any_of(node, schema[keyword], root, path, issues, keyword)
            return

    if "allOf" in schema:
        for branch in schema["allOf"]:
            _validate(node, branch, root, path, issues)

    declared = schema.get("type")
    if declared is not None and not _type_matches(node, declared):
        issues.append(
            issue(
                path,
                f"expected type {_type_name(declared)}, got {_python_type_name(node)}",
                Stage.STRUCTURAL,
                "type_mismatch",
            )
        )
        return

    if isinstance(node, dict):
        _validate_object(node, schema, root, path, issues)
    elif isinstance(node, list):
        _validate_array(node, schema, root, path, issues)
    elif isinstance(node, str):
        _validate_string(node, schema, path, issues)
    elif isinstance(node, bool):
        pass
    elif isinstance(node, (int, float)):
        _validate_number(node, schema, path, issues)


def _validate_any_of(
    node: Any,
    branches: Any,
    root: dict,
    path: str,
    issues: list[Issue],
    keyword: str,
) -> None:
    if not isinstance(branches, list) or not branches:
        return
    branch_issues: list[list[Issue]] = []
    for branch in branches:
        collected: list[Issue] = []
        _validate(node, branch, root, path, collected)
        if not collected:
            return
        branch_issues.append(collected)

    # Every branch failed. Report the one that got closest rather than all of them:
    # a repair prompt listing five contradictory demands is worse than no hint.
    best = min(branch_issues, key=len)
    if len(best) == 1:
        issues.append(best[0])
        return
    issues.append(
        issue(
            path,
            f"does not match any {keyword} branch ({_brief(node)})",
            Stage.STRUCTURAL,
            "any_of_mismatch",
        )
    )


def _validate_object(node: dict, schema: dict, root: dict, path: str, issues: list[Issue]) -> None:
    properties = schema.get("properties")
    properties = properties if isinstance(properties, dict) else {}

    for name in schema.get("required", []) or []:
        if name not in node:
            issues.append(
                issue(f"{path}.{name}", "required property is missing", Stage.STRUCTURAL, "missing_required")
            )

    if schema.get("additionalProperties") is False:
        for name in node:
            if name not in properties:
                issues.append(
                    issue(
                        f"{path}.{name}",
                        "property is not allowed by the schema",
                        Stage.STRUCTURAL,
                        "additional_property",
                    )
                )

    for name, subschema in properties.items():
        if name in node:
            _validate(node[name], subschema, root, f"{path}.{name}", issues)


def _validate_array(node: list, schema: dict, root: dict, path: str, issues: list[Issue]) -> None:
    minimum = schema.get("minItems")
    maximum = schema.get("maxItems")
    if isinstance(minimum, int) and len(node) < minimum:
        issues.append(
            issue(path, f"expected at least {minimum} items, got {len(node)}", Stage.STRUCTURAL, "min_items")
        )
    if isinstance(maximum, int) and len(node) > maximum:
        issues.append(
            issue(path, f"expected at most {maximum} items, got {len(node)}", Stage.STRUCTURAL, "max_items")
        )

    items = schema.get("items")
    if items is None:
        return
    for index, element in enumerate(node):
        _validate(element, items, root, f"{path}[{index}]", issues)


def _validate_string(node: str, schema: dict, path: str, issues: list[Issue]) -> None:
    minimum = schema.get("minLength")
    maximum = schema.get("maxLength")
    if isinstance(minimum, int) and len(node) < minimum:
        issues.append(
            issue(path, f"expected at least {minimum} characters, got {len(node)}", Stage.STRUCTURAL, "min_length")
        )
    if isinstance(maximum, int) and len(node) > maximum:
        issues.append(
            issue(path, f"expected at most {maximum} characters, got {len(node)}", Stage.STRUCTURAL, "max_length")
        )
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        try:
            matched = re.search(pattern, node) is not None
        except re.error:
            return
        if not matched:
            issues.append(issue(path, f"does not match pattern {pattern!r}", Stage.STRUCTURAL, "pattern"))


def _validate_number(node: float, schema: dict, path: str, issues: list[Issue]) -> None:
    checks = (
        ("minimum", lambda value, bound: value >= bound, "must be >= {bound}"),
        ("maximum", lambda value, bound: value <= bound, "must be <= {bound}"),
        ("exclusiveMinimum", lambda value, bound: value > bound, "must be > {bound}"),
        ("exclusiveMaximum", lambda value, bound: value < bound, "must be < {bound}"),
    )
    for keyword, predicate, template in checks:
        bound = schema.get(keyword)
        if isinstance(bound, (int, float)) and not isinstance(bound, bool) and not predicate(node, bound):
            issues.append(
                issue(path, f"{template.format(bound=bound)}, got {node}", Stage.STRUCTURAL, keyword)
            )

    multiple = schema.get("multipleOf")
    if isinstance(multiple, (int, float)) and not isinstance(multiple, bool) and multiple > 0:
        quotient = node / multiple
        if abs(quotient - round(quotient)) > 1e-9:
            issues.append(
                issue(path, f"must be a multiple of {multiple}, got {node}", Stage.STRUCTURAL, "multiple_of")
            )


def _resolve(schema: dict, root: dict, path: str, issues: list[Issue]) -> dict | None:
    """Follow a local ``$ref``. Remote refs are not supported by design."""
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return schema
    if not ref.startswith("#/"):
        issues.append(issue(path, f"unsupported non-local $ref {ref!r}", Stage.SCHEMA, "remote_ref"))
        return None

    target: Any = root
    for segment in ref[2:].split("/"):
        segment = segment.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or segment not in target:
            issues.append(issue(path, f"$ref {ref!r} does not resolve", Stage.SCHEMA, "unresolved_ref"))
            return None
        target = target[segment]

    if not isinstance(target, dict):
        issues.append(issue(path, f"$ref {ref!r} does not point at a schema", Stage.SCHEMA, "unresolved_ref"))
        return None

    merged = {key: value for key, value in schema.items() if key != "$ref"}
    return {**target, **merged}


def _type_matches(node: Any, declared: Any) -> bool:
    names = declared if isinstance(declared, list) else [declared]
    return any(_single_type_matches(node, name) for name in names)


def _single_type_matches(node: Any, name: Any) -> bool:
    expected = _JSON_TYPES.get(name)
    if expected is None:
        return True
    # JSON booleans are not numbers, but Python's bool subclasses int.
    if isinstance(node, bool):
        return name == "boolean"
    if name == "integer" and isinstance(node, float):
        return node.is_integer()
    return isinstance(node, expected)


def _type_name(declared: Any) -> str:
    return " or ".join(declared) if isinstance(declared, list) else str(declared)


def _python_type_name(node: Any) -> str:
    if node is None:
        return "null"
    if isinstance(node, bool):
        return "boolean"
    if isinstance(node, str):
        return "string"
    if isinstance(node, int):
        return "integer"
    if isinstance(node, float):
        return "number"
    if isinstance(node, list):
        return "array"
    if isinstance(node, dict):
        return "object"
    return type(node).__name__


def _brief(value: Any, limit: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _brief_list(values: Any, limit: int = 5) -> str:
    if not isinstance(values, list):
        return _brief(values)
    shown = ", ".join(repr(value) for value in values[:limit])
    return f"[{shown}]" if len(values) <= limit else f"[{shown}, ... {len(values) - limit} more]"
