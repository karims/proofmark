"""Deep nesting, arrays, $refs -- on a permissive schema.

    python examples/recipe.py

Runs offline. Two things this shows that the budget example does not:

1. **A permissive schema.** No ``additionalProperties: false``, optional properties,
   nullable unions, ``$defs``/``$ref`` reuse, and enums. This is what a schema looks
   like when you are not contorting it to satisfy strict structured output -- so it
   is checked against the ``GENERIC`` profile, and the script prints what
   ``OPENAI_STRICT`` would have said about the same schema.

2. **A model repair round.** Some defects are arithmetic and code fixes them. Others
   -- an invalid enum value, a missing required field -- are content, and no amount
   of deterministic normalization invents them. Those are what a repair call is for.

The structure exercised here is genuinely nested: arrays of objects containing
arrays of strings, a shared ``$ref`` used at two different depths, and a two-level
nested object under an optional key.
"""

import json
from decimal import Decimal

from proofmark import (
    GENERIC,
    OPENAI_STRICT,
    Issue,
    Outcome,
    Stage,
    StaticProvider,
    checks,
    compose,
    normalize,
    paths,
    preflight,
)

RECIPE_SCHEMA = {
    "type": "object",
    "required": ["name", "servings", "ingredients", "steps"],
    "properties": {
        "name": {"type": "string"},
        # Optional. Under a strict profile this would have to be a nullable union
        # listed in `required`; here it can simply be absent.
        "cuisine": {"type": "string"},
        "servings": {"type": "integer", "minimum": 1},
        "difficulty": {"enum": ["easy", "medium", "hard"]},
        "total_minutes": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "ingredients": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["item", "quantity"],
                "properties": {
                    "item": {"type": "string"},
                    "quantity": {"$ref": "#/$defs/Quantity"},
                    "notes": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    # An array of scalars nested inside an array of objects.
                    "substitutes": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["number", "instruction"],
                "properties": {
                    "number": {"type": "integer"},
                    "instruction": {"type": "string"},
                    "minutes": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    "equipment": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "nutrition_per_serving": {
            "type": "object",
            "required": ["calories", "macros"],
            "properties": {
                "calories": {"type": "number"},
                "macros": {
                    "type": "object",
                    "required": ["protein_g", "carbs_g", "fat_g"],
                    "properties": {
                        "protein_g": {"type": "number"},
                        "carbs_g": {"type": "number"},
                        "fat_g": {"type": "number"},
                    },
                },
            },
        },
    },
    "$defs": {
        # Referenced from ingredients[].quantity. A $ref keeps one definition of
        # what a measured amount looks like instead of three drifting copies.
        "Quantity": {
            "type": "object",
            "required": ["amount", "unit"],
            "properties": {
                "amount": {"type": "number"},
                "unit": {"enum": ["g", "ml", "tbsp", "tsp", "cup", "piece", "pinch"]},
            },
        }
    },
}


# --- a domain check, which is just a function -------------------------------


def steps_are_sequential(data: dict) -> list[Issue]:
    """Step numbers must run 1, 2, 3... with no gaps or repeats.

    Nothing in JSON Schema can express "this integer depends on its position in the
    array". This is the shape most real semantic checks take.
    """
    found: list[Issue] = []
    for position, match in enumerate(paths.resolve(data, "steps[].number"), start=1):
        if match.value != position:
            found.append(
                Issue(
                    path=match.path,
                    message=f"expected step number {position}, found {match.value}",
                    stage=Stage.SEMANTIC,
                    code="step_out_of_order",
                )
            )
    return found


def total_time_covers_the_steps(data: dict) -> list[Issue]:
    """Stated total time must be at least the sum of the timed steps."""
    stated = paths.first(data, "total_minutes")
    if stated is None:
        return []
    timed = [value for value in paths.values(data, "steps[].minutes") if isinstance(value, (int, float))]
    required = sum(Decimal(str(value)) for value in timed)
    if Decimal(str(stated)) < required:
        return [
            Issue(
                path="$.total_minutes",
                message=f"stated total of {stated} min is less than the {required} min of timed steps",
                stage=Stage.SEMANTIC,
                code="total_time_too_short",
            )
        ]
    return []


# --- what the model returns -------------------------------------------------

# Wrong in four ways, of two different kinds.
#   Fixable in code:   amounts as strings, a duplicated ingredient row.
#   Needs the model:   an invalid `unit` enum value, and a step missing its
#                      `instruction` entirely.
FIRST_ATTEMPT = {
    "name": "Weeknight Dal",
    "cuisine": "Indian",
    "servings": 4,
    "difficulty": "easy",
    "total_minutes": 40,
    "tags": ["vegetarian", "one-pot"],
    "ingredients": [
        {
            "item": "red lentils",
            "quantity": {"amount": "200", "unit": "g"},
            "notes": "rinsed until the water runs clear",
            "substitutes": ["yellow split peas"],
        },
        {"item": "red lentils", "quantity": {"amount": "200", "unit": "g"}, "notes": None},
        {"item": "cumin seeds", "quantity": {"amount": "1", "unit": "handful"}},
        {"item": "coconut milk", "quantity": {"amount": "400", "unit": "ml"}, "substitutes": []},
    ],
    "steps": [
        {"number": 1, "instruction": "Toast the cumin seeds.", "minutes": 2, "equipment": ["pan"]},
        {"number": 2, "minutes": 25, "equipment": ["pot", "wooden spoon"]},
        {"number": 3, "instruction": "Stir in coconut milk and simmer.", "minutes": 10},
    ],
    "nutrition_per_serving": {
        "calories": 410,
        "macros": {"protein_g": 18, "carbs_g": 52, "fat_g": 14},
    },
}

REPAIRED_ATTEMPT = json.loads(json.dumps(FIRST_ATTEMPT))
REPAIRED_ATTEMPT["ingredients"] = [
    FIRST_ATTEMPT["ingredients"][0],
    {"item": "cumin seeds", "quantity": {"amount": 1, "unit": "tsp"}},
    FIRST_ATTEMPT["ingredients"][3],
]
REPAIRED_ATTEMPT["steps"][1] = {
    "number": 2,
    "instruction": "Simmer the lentils with 600 ml water until collapsing.",
    "minutes": 25,
    "equipment": ["pot", "wooden spoon"],
}


def main() -> None:
    strict = preflight(RECIPE_SCHEMA, OPENAI_STRICT)
    generic = preflight(RECIPE_SCHEMA, GENERIC)

    print("=" * 72)
    print("PREFLIGHT: the same schema, two profiles")
    print("=" * 72)
    print(f"OPENAI_STRICT -> {len(strict.errors)} error(s). The first three:")
    for error in strict.errors[:3]:
        print(f"    {error.format()}")
    print(f"GENERIC       -> {'ok' if generic.ok else 'FAILED'}")
    print()
    print("Neither profile is 'right'. Use GENERIC when you are validating output")
    print("you already have; use OPENAI_STRICT when the schema is going to a strict")
    print("provider, and let it tell you what to change first.")
    print()

    provider = StaticProvider(responses=[FIRST_ATTEMPT, REPAIRED_ATTEMPT])

    result = compose(
        prompt="Give me a weeknight dal recipe for 4.",
        schema=RECIPE_SCHEMA,
        provider=provider,
        profile=GENERIC,
        normalizers=[
            # Reaches through an array into a $ref'd nested object.
            normalize.coerce_numbers("ingredients[].quantity.amount"),
            normalize.deduplicate("ingredients", key="item"),
        ],
        checks=[
            steps_are_sequential,
            total_time_covers_the_steps,
            checks.unique("ingredients", key="item"),
            checks.no_placeholders(),
        ],
        repair_attempts=1,
    )

    print("=" * 72)
    print("COMPOSE")
    print("=" * 72)
    print(f"result: {result.summary()}")
    print()
    # `result.normalizations` describes the payload you are holding -- the repaired
    # one. What the *first* attempt needed is in the trace, which is the point of
    # keeping one.
    first_pass_fixes = [
        entry["normalizer"]
        for entry in result.trace["events"]
        if entry["event"] == "normalized" and entry.get("stage") == "initial"
    ]

    print("Fixed in code on the first attempt, costing nothing:")
    for name in first_pass_fixes:
        print(f"    {name}")
    print()
    print("Left over -- content the model had to supply:")
    for problem in result.initial_issues:
        print(f"    {problem.format()}")
    print()
    print(f"Provider calls: {result.provider_calls} (one generation, one repair)")
    print()

    assert result.ok
    assert result.outcome is Outcome.REPAIRED

    data = result.data
    print("Deep paths resolve through every level of nesting:")
    print(f"    ingredients[].quantity.unit  -> {paths.values(data, 'ingredients[].quantity.unit')}")
    print(f"    steps[].equipment[]          -> {paths.values(data, 'steps[].equipment[]')}")
    print(f"    nutrition.macros.protein_g   -> {paths.first(data, 'nutrition_per_serving.macros.protein_g')}")
    print()
    print(f"Ingredients after dedup: {[item['item'] for item in data['ingredients']]}")
    print(f"Steps: {[step['number'] for step in data['steps']]} -- all with instructions")


if __name__ == "__main__":
    main()
