"""The examples are the primary documentation, so they are tested like code.

Each one asserts its own claims internally and runs entirely offline, so importing
and calling ``main()`` is a real check that the README's story still holds.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(name: str):
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"examples_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["budget", "recipe", "invoice"])
def test_example_runs_and_its_assertions_hold(name, capsys):
    load(name).main()

    assert capsys.readouterr().out.strip(), f"{name} should print something"
