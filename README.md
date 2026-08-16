# proofmark

**Structured LLM output that has been tested, not just typed.**

A *proof mark* is the stamp struck into a barrel or an ingot after it has passed a
proof test — evidence the piece was put under load and survived, applied by the
proof house rather than the maker.

Getting a model to emit JSON that matches a schema is the easy half, and several
libraries do it well. proofmark covers the rest: whether the document is actually
*correct*, what to do when it isn't, and how to find out why.

```bash
pip install proofmark
```

Zero required dependencies. Providers speak HTTP over the standard library and
schema validation is implemented in-tree, so adding proofmark to a project does not
drag in a dependency tree.

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

Every field has the right type. The schema validates. **The numbers add up to 950.**

No amount of schema conformance catches this, because it is not a typing error —
it is an arithmetic error. And the standard reflex, asking the model to try again,
is the wrong move: it costs a round trip, and the model is being asked to redo the
arithmetic it just got wrong.

proofmark fixes it in code, in microseconds, and tells you it did.

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

    # Correctness the schema cannot express.
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
than `OK`, so you know the model got it wrong even though you got a correct
document.

---

## What it does

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

### 3. Validates semantically

Schema validity is table stakes. The built-ins cover the recurring ways a
schema-valid document is still wrong:

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

## Testing without a network

Every behaviour proofmark claims to handle is reproducible offline. The library's
own suite is 97 tests in 0.05s with no network and no mocking of HTTP.

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
[Outlines](https://github.com/dottxt-ai/outlines) get you schema conformance and
stop there. proofmark starts where they finish. Use both if you like — proofmark
does not care how the JSON was produced, only whether it holds up.

## License

MIT — see [LICENSE](LICENSE).
