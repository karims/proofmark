"""Catch schemas a provider will reject -- before spending a call on them.

Structured-output modes impose constraints that ordinary JSON Schema does not:
every object must close itself to extra properties, every property must be
required, nesting is capped. A schema that violates these comes back as an opaque
400 *after* you have paid for the request and waited for the round trip, and the
error rarely names the offending path.

Preflight is the cheapest check in the library and the only one that runs before
any network call.
"""

from dataclasses import dataclass, field
from typing import Any

from proofmark.issues import Issue, Stage, issue


@dataclass(frozen=True)
class Profile:
    """Provider constraints to check against.

    Limits are named and overridable rather than baked in, because providers move
    them. A hardcoded ``max_depth`` becomes a false rejection the week after the
    vendor raises the cap.
    """

    name: str
    require_closed_objects: bool = False
    require_all_properties_required: bool = False
    max_depth: int | None = None
    max_properties: int | None = None
    advisory_keywords: frozenset[str] = frozenset()


OPENAI_STRICT = Profile(
    name="openai_strict",
    require_closed_objects=True,
    require_all_properties_required=True,
    max_depth=5,
    max_properties=100,
    advisory_keywords=frozenset(
        {
            "minLength",
            "maxLength",
            "pattern",
            "format",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minItems",
            "maxItems",
            "default",
        }
    ),
)
"""Constraints for OpenAI-style strict ``json_schema`` response format.

``advisory_keywords`` are reported as warnings, not errors. Support for them has
expanded over time and varies by model, so proofmark tells you they may be ignored
rather than refusing to send. Note the consequence: a schema relying on ``minimum``
for correctness should back it with a semantic check, since the provider may not
enforce it.
"""

GENERIC = Profile(name="generic")
"""No provider-specific constraints. Structural sanity only."""


@dataclass
class PreflightReport:
    """Errors block the call. Warnings are recorded in the trace and sent anyway."""

    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
        }


def preflight(schema: dict, profile: Profile = OPENAI_STRICT) -> PreflightReport:
    """Check ``schema`` against ``profile`` without contacting a provider."""
    report = PreflightReport()

    if not isinstance(schema, dict):
        report.errors.append(issue("$", "schema must be an object", Stage.SCHEMA, "not_an_object"))
        return report

    if schema.get("type") != "object" and "$ref" not in schema and "properties" not in schema:
        report.errors.append(
            issue("$", "root schema must be an object type", Stage.SCHEMA, "root_not_object")
        )

    counter = _Counter()
    _walk(schema, schema, "$", profile, report, depth=0, counter=counter, seen=set())

    if profile.max_properties is not None and counter.properties > profile.max_properties:
        report.errors.append(
            issue(
                "$",
                f"schema declares {counter.properties} properties; {profile.name} allows "
                f"{profile.max_properties}",
                Stage.SCHEMA,
                "too_many_properties",
            )
        )

    return report


@dataclass
class _Counter:
    properties: int = 0


def _walk(
    node: Any,
    root: dict,
    path: str,
    profile: Profile,
    report: PreflightReport,
    depth: int,
    counter: _Counter,
    seen: set[int],
) -> None:
    if not isinstance(node, dict):
        return

    # $defs are walked through their references; recursion guards against cycles.
    marker = id(node)
    if marker in seen:
        return
    seen = seen | {marker}

    if profile.max_depth is not None and depth > profile.max_depth:
        report.errors.append(
            issue(
                path,
                f"nesting depth {depth} exceeds the {profile.name} limit of {profile.max_depth}",
                Stage.SCHEMA,
                "too_deep",
            )
        )
        return

    ref = node.get("$ref")
    if isinstance(ref, str):
        target = _resolve(ref, root)
        if target is None:
            report.errors.append(issue(path, f"$ref {ref!r} does not resolve", Stage.SCHEMA, "unresolved_ref"))
        else:
            _walk(target, root, path, profile, report, depth, counter, seen)
        return

    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = node.get(keyword)
        if isinstance(branches, list):
            for index, branch in enumerate(branches):
                _walk(branch, root, f"{path}/{keyword}[{index}]", profile, report, depth, counter, seen)

    _check_const_type(node, path, report)
    _check_advisory(node, path, profile, report)

    declared = node.get("type")
    types = declared if isinstance(declared, list) else [declared] if declared else []

    if "object" in types:
        _check_object(node, root, path, profile, report, depth, counter, seen)
    elif "array" in types:
        items = node.get("items")
        if items is None:
            report.errors.append(
                issue(path, "array schema must declare 'items'", Stage.SCHEMA, "array_without_items")
            )
        else:
            _walk(items, root, f"{path}[]", profile, report, depth + 1, counter, seen)
    elif not types and not any(key in node for key in ("enum", "const", "anyOf", "oneOf", "allOf")):
        report.errors.append(
            issue(path, "schema node declares no 'type'", Stage.SCHEMA, "missing_type")
        )


def _check_object(
    node: dict,
    root: dict,
    path: str,
    profile: Profile,
    report: PreflightReport,
    depth: int,
    counter: _Counter,
    seen: set[int],
) -> None:
    properties = node.get("properties")
    if not isinstance(properties, dict) or not properties:
        report.errors.append(
            issue(path, "object schema must declare at least one property", Stage.SCHEMA, "empty_object")
        )
        return

    counter.properties += len(properties)

    if profile.require_closed_objects and node.get("additionalProperties") is not False:
        report.errors.append(
            issue(
                path,
                "object must set 'additionalProperties': false",
                Stage.SCHEMA,
                "open_object",
            )
        )

    if profile.require_all_properties_required:
        required = node.get("required")
        required_names = set(required) if isinstance(required, list) else set()
        missing = [name for name in properties if name not in required_names]
        if missing:
            report.errors.append(
                issue(
                    path,
                    f"every property must be listed in 'required'; missing {sorted(missing)}. "
                    "Model optional fields as a nullable union instead.",
                    Stage.SCHEMA,
                    "optional_property",
                )
            )
        unknown = sorted(required_names - set(properties))
        if unknown:
            report.errors.append(
                issue(
                    path,
                    f"'required' names properties that do not exist: {unknown}",
                    Stage.SCHEMA,
                    "required_unknown",
                )
            )

    for name, subschema in properties.items():
        _walk(subschema, root, f"{path}.{name}", profile, report, depth + 1, counter, seen)


def _check_const_type(node: dict, path: str, report: PreflightReport) -> None:
    """A ``const`` that contradicts its own ``type`` can never be satisfied."""
    if "const" not in node:
        return
    declared = node.get("type")
    if not isinstance(declared, str):
        return
    value = node["const"]
    matches = {
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "null": value is None,
    }.get(declared)
    if matches is False:
        report.errors.append(
            issue(
                path,
                f"const {value!r} contradicts declared type {declared!r}",
                Stage.SCHEMA,
                "const_type_mismatch",
            )
        )


def _check_advisory(node: dict, path: str, profile: Profile, report: PreflightReport) -> None:
    present = sorted(profile.advisory_keywords & set(node))
    if present:
        report.warnings.append(
            issue(
                path,
                f"{profile.name} may ignore {present}; back these with a semantic check "
                "if correctness depends on them",
                Stage.SCHEMA,
                "advisory_keyword",
            )
        )


def _resolve(ref: str, root: dict) -> dict | None:
    if not ref.startswith("#/"):
        return None
    target: Any = root
    for segment in ref[2:].split("/"):
        segment = segment.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or segment not in target:
            return None
        target = target[segment]
    return target if isinstance(target, dict) else None
