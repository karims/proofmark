# Examples

All three run offline with no API key — they replay recorded model responses
through `StaticProvider`. Each asserts its own claims, and `tests/test_examples.py`
runs all three, so they cannot silently rot.

```bash
python examples/budget.py
python examples/recipe.py
python examples/invoice.py
```

| Example | Schema style | Shows |
|---|---|---|
| [`budget.py`](budget.py) | strict, flat | The core pitch: a schema-valid document whose numbers don't add up, fixed in code in one provider call. |
| [`recipe.py`](recipe.py) | permissive, deeply nested | `$defs`/`$ref` reuse, arrays of objects containing arrays, nullable unions, enums. Contrasts the `GENERIC` and `OPENAI_STRICT` preflight profiles on one schema, and shows a model repair round for defects code cannot fix. |
| [`invoice.py`](invoice.py) | strict, real-world | Document extraction. Custom domain checks and normalizers, chained arithmetic, and all three outcomes: normalized, repaired, and failed-for-human-review. |

## Start with `invoice.py`

It carries the idea the library is organised around:

> **Normalize what is derivable. Check what is evidence.**

`line_total` is derivable — it is `quantity × unit_price` and nothing else — so
code recomputes it and the model is never asked. `subtotal` is *evidence*: it is
printed on the document, and when it disagrees with the extracted rows, that
usually means a row was **missed**. Auto-correcting the subtotal to match the rows
you found would erase the only signal that anything is wrong, and hand you a tidy,
internally consistent, wrong invoice.

That distinction is a silent data-loss bug in a lot of extraction pipelines. In
proofmark it is the difference between passing something to `normalizers=` and
passing it to `checks=`.

Scenario 3 in that example ends with no payload at all. That is intentional: an
invoice you cannot verify is worse than no invoice.
