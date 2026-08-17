# proofmark

**proofmark validates and repairs structured LLM output beyond JSON Schema.**

Define semantic checks for things like totals, bounds, duplicates, and
placeholders, then deterministically fix what can be fixed before spending another
LLM call.

```bash
pip install proofmark
```

The pipeline:

```text
LLM output
    ↓
normalize what can be deterministically derived
    ↓
structural/schema validation
    ↓
application-level semantic checks
    ↓
LLM repair only if necessary
```

The differentiator is the ordering: **repair with deterministic code before
asking the model again.** A second model call costs latency and money, and it
re-rolls the dice on output the previous call already got wrong. Fixing in code
what code can fix removes those retries entirely.

Zero required dependencies, incidentally: providers speak HTTP over the standard
library and schema validation is implemented in-tree.

---

## The problem

Your schema says `budget.total` is a number and `budget.items[].amount` is a
number. The model returns:

```json
{
  "budget": {
    "total": 1000.0,
    "items": [
      { "category": "lodging", "amount": 500.0 },
      { "category": "food",    "amount": 300.0 },
      { "category": "contingency", "amount": 150.0 }
    ]
  }
}
```

Every field has the right type. JSON Schema is satisfied. **The line items add up
to 950.**

Schema conformance cannot catch this: "line items sum to the stated total" is an
application invariant, not a typing rule. proofmark lets you state that invariant
(`checks.sums_to(...)`), detects the violation, and — if you have supplied a
normalizer that says how to reconcile it — repairs it in code without another
provider call.

proofmark does not decide which number is authoritative. That policy is your
normalizer configuration: absorbing the shortfall into a contingency line is a
choice you make, not something the library infers.

### Normalize what is derivable. Check what is evidence.

This is the principle the library is organised around, and it is how you decide
whether something belongs in `normalizers=` or `checks=`.

- **Derived** values can safely be recomputed. On an invoice, `line_total` is
  `quantity × unit_price` and nothing else, so code recomputes it and the model is
  never asked.
- **Evidence** values should usually be checked, not overwritten. A `subtotal`
  printed on the source document is evidence; when it disagrees with the extracted
  rows, that often means a row was *missed*.

Auto-correcting evidence to match the rest of the document erases the only signal
that extraction went wrong, and hands you a tidy, internally consistent, wrong
invoice. That is a silent data-loss bug in a lot of extraction pipelines; here it
is just the difference between `normalizers=` and `checks=`.

---

## Usage

```python
from proofmark import compose, checks, normalize, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(api_key=KEY, model="gpt-4o-mini")

result = compose(
    prompt="Draft a 3-day budget for Lisbon. Total 1000 EUR.",
    schema=BUDGET_SCHEMA,
    provider=provider,

    # Deterministic repair, applied before anything is validated.
    normalizers=[
        normalize.coerce_numbers("budget.items[].amount"),
        normalize.rebalance_to_total(
            items="budget.items",
            amount_field="amount",
            total="budget.total",
            slack_match={"category": "contingency"},
        ),
    ],

    # Application-level rules JSON Schema cannot express.
    checks=[
        checks.sums_to("budget.items", "amount", "budget.total"),
        checks.currency_consistent("EUR"),
        checks.no_placeholders(),
    ],

    repair_attempts=1,
)

if result.ok:
    use(result.data)
else:
    log(result.failure, result.issues, result.trace)
```

That run costs **one** provider call. The 50-EUR shortfall is absorbed into
contingency by `rebalance_to_total`, and `result.outcome` is `NORMALIZED` rather
than `OK` — so you know the model got it wrong even though the document you got
satisfies the constraints you defined.

---

## What it does

The built-in checks and normalizers below are a starting set, not the point. The
abstraction is the pipeline and its extension model: checks are ordinary callables,
normalizers are deterministic, structural and semantic failures are separated,
model repair runs only after deterministic repair fails, failure types are
explicit, traces record what happened, and provider degradation is handled the
same way across backends.

### 1. Preflights the schema — before spending a call

Strict structured-output modes impose constraints ordinary JSON Schema does not:
every object must close itself to extra properties, every property must be listed
in `required`, nesting is capped. Violate one and you get an opaque 400 *after*
paying for the round trip, usually without being told which path is at fault.

```python
from proofmark import preflight

report = preflight(MY_SCHEMA)
for error in report.errors:
    print(error.format())
# $.address: object must set 'additionalProperties': false
# $.user: every property must be listed in 'required'; missing ['nickname'].
#         Model optional fields as a nullable union instead.
```

Warnings are separate from errors. `minimum`, `pattern`, and friends may be
silently ignored by a strict provider — proofmark tells you rather than refusing to
send, and `checks.bounded()` is there to enforce them yourself.

### 2. Degrades across tiers deliberately, and tells you which one ran

`native_structured` → `json_mode` → `text_json`. The distinction that matters:
a provider saying *"this model can't do json_schema"* triggers degradation, while
a provider saying *"401"* does not. Conflating the two turns an auth failure into a
mysteriously low-quality result.

`result.tier` records what actually produced your payload. "The model is bad at
JSON" and "your model silently fell back to prompt-and-pray" look identical
without it.

### 3. Runs application-level semantic checks

Schema validity is table stakes. The built-ins cover recurring ways a schema-valid
document still violates a domain invariant:

| Check | Catches |
|---|---|
| `sums_to` | line items that don't add up to the stated total |
| `currency_consistent` | a budget narrated in dollars after you asked for euros |
| `no_placeholders` | `"TBD"`, `"N/A"`, `"lorem ipsum"` left in the output |
| `unique` | two entries for day 3 |
| `bounded` | negative quantities the provider ignored your `minimum` for |
| `non_empty` | required prose the model left blank |

Anything domain-specific is a plain `Callable[[dict], list[Issue]]`.

### 4. Repairs deterministically first

Normalizers run before validation and cost nothing:

| Normalizer | Fixes |
|---|---|
| `coerce_numbers` | `"1,200.00"` → `1200.0` |
| `rebalance_to_total` | forces line items to sum exactly, in `Decimal` |
| `drop_unknown_properties` | deletes the chatty `"explanation"` key |
| `round_amounts` | quantizes to a fixed precision |
| `deduplicate` | drops repeated list entries |

`rebalance_to_total` pushes the whole discrepancy into a designated slack category
if you name one, and otherwise redistributes proportionally using the
largest-remainder method — so the parts sum to the target exactly, rather than
drifting by a cent per line. All arithmetic is `Decimal`.

Only what survives normalization is worth a model call.

### 5. Explains the failure

```python
result.outcome   # ok | normalized | repaired | fallback | failed
result.failure   # invalid_schema | provider_refusal | incomplete_response |
                 # empty_response | unparseable_response |
                 # structural_invalid | semantic_invalid | provider_error | ...
result.issues    # [Issue(path='$.budget.items', message='line items sum to 950...')]
result.trace     # sanitized, diffable event log
```

`incomplete_response` means you hit the token limit — a different fix from
`provider_refusal`, which is a different fix again from `semantic_invalid`. Most
code built on structured output collapses all of these into "it failed" and retries
blindly.

The trace is safe to paste into a bug report: credentials are redacted on the way
in, prompts are excluded by default because they carry user data, and it contains
no timestamps or object ids — so two runs of the same input produce traces that
differ only where behaviour differed.

---

## Examples

All three run offline with no API key, and are covered by the test suite so they
cannot silently rot. See [`examples/`](examples/).

```bash
python examples/budget.py    # the core pitch, on a flat strict schema
python examples/recipe.py    # deep nesting, $refs, arrays, permissive schema
python examples/invoice.py   # document extraction -- start here
```

[`invoice.py`](examples/invoice.py) is the worked version of *normalize what is
derivable, check what is evidence*: `line_total` is recomputed by a normalizer,
`subtotal` is left alone and asserted by a check.

(The name: a proof mark is the stamp something gets once it has been tested.)

## Testing without a network

Every behaviour proofmark claims to handle is reproducible offline. The library's
own suite is 100 tests in 0.04s with no network and no mocking of HTTP.

```python
from proofmark import compose, StaticProvider
from proofmark.errors import TierUnsupported

# Degradation, without inventing HTTP error payloads.
provider = StaticProvider(responses=[
    TierUnsupported("native_structured", "model does not support json_schema"),
    {"title": "Lisbon", "budget": {...}},
])

result = compose(prompt="...", schema=SCHEMA, provider=provider)
assert result.tier is Tier.JSON_MODE
```

`compose(provider=None, fallback=...)` is a supported path rather than a crash, so
a missing API key degrades to your fallback instead of taking down a test run.

---

## Design notes

**Providers are deliberately dumb.** They issue one request at one tier and report
what happened. Tier degradation, JSON recovery, validation, and repair all live in
the pipeline, so behaviour is identical across backends and testable without a
network. Writing a provider means implementing one method.

**Normalization precedes repair.** This is the ordering the whole library is built
around. A second model call is slow, costs money, and may return something new to
be wrong about.

**Semantic checks are skipped when the payload is structurally broken.** Running
them against a malformed document produces issues addressed to fields that may not
exist, which then pollutes the repair prompt.

**Repair prompts quote specific paths.** `$.title: required property is missing`,
not "the output was invalid." All issues are collected and sent at once, so a
document with five problems takes one repair round, not five.

---

## Status

`0.1.0`, alpha. The API may move before `1.0`.

Not yet implemented: Pydantic model adaptation (pass a JSON Schema for now — 
`Model.model_json_schema()` works), Anthropic-native tool-use provider, streaming,
and a response cache for deterministic replay.

Nearest neighbours: [Instructor](https://github.com/jxnl/instructor) and
[Outlines](https://github.com/dottxt-ai/outlines). proofmark is focused
specifically on the layer after structured generation — deterministic
normalization, application-level invariant checks, repair orchestration, failure
classification, and tracing — so it complements libraries that already handle
structured generation. It does not care how the JSON was produced.

## License

MIT — see [LICENSE](LICENSE).
