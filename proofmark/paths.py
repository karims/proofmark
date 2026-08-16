"""Dotted path resolution with list wildcards.

Checks and normalizers both need to address parts of a payload without knowing its
shape in advance, and normalizers additionally need to *write* back. Every match
carries a setter bound to its parent container, so a normalizer can rewrite a value
in place without re-walking the tree.

Grammar::

    budget.total            a single field
    budget.items[].amount   every element of a list
    days[0].activities[]    a specific index, then every element
"""

import re
from dataclasses import dataclass
from typing import Any

_SEGMENT = re.compile(r"([^.\[\]]+)|\[(-?\d*)\]")


@dataclass
class Match:
    """One resolved location in a payload."""

    path: str
    value: Any
    parent: Any
    key: Any

    def set(self, value: Any) -> None:
        """Write ``value`` back into the payload at this location."""
        if self.parent is None:
            raise ValueError("cannot set the document root")
        self.parent[self.key] = value
        self.value = value

    def delete(self) -> None:
        if self.parent is None:
            raise ValueError("cannot delete the document root")
        del self.parent[self.key]


def resolve(data: Any, path: str) -> list[Match]:
    """Return every location in ``data`` matching ``path``.

    Missing intermediate keys yield no matches rather than raising -- a check
    written against an optional branch should simply find nothing there.
    """
    matches = [Match(path="$", value=data, parent=None, key=None)]

    for token, index in _tokenize(path):
        next_matches: list[Match] = []
        for match in matches:
            if token is not None:
                if isinstance(match.value, dict) and token in match.value:
                    next_matches.append(
                        Match(
                            path=f"{match.path}.{token}",
                            value=match.value[token],
                            parent=match.value,
                            key=token,
                        )
                    )
                continue

            if not isinstance(match.value, list):
                continue
            if index is None:
                next_matches.extend(
                    Match(path=f"{match.path}[{position}]", value=element, parent=match.value, key=position)
                    for position, element in enumerate(match.value)
                )
            elif -len(match.value) <= index < len(match.value):
                next_matches.append(
                    Match(
                        path=f"{match.path}[{index}]",
                        value=match.value[index],
                        parent=match.value,
                        key=index,
                    )
                )
        matches = next_matches
        if not matches:
            return []

    return matches


def first(data: Any, path: str, default: Any = None) -> Any:
    matches = resolve(data, path)
    return matches[0].value if matches else default


def values(data: Any, path: str) -> list[Any]:
    return [match.value for match in resolve(data, path)]


def walk_strings(data: Any, path: str = "$") -> list[Match]:
    """Every string in the payload, with its location. Used by text-wide checks."""
    found: list[Match] = []
    _walk_strings(data, path, None, None, found)
    return found


def _walk_strings(node: Any, path: str, parent: Any, key: Any, found: list[Match]) -> None:
    if isinstance(node, str):
        found.append(Match(path=path, value=node, parent=parent, key=key))
        return
    if isinstance(node, dict):
        for name, child in node.items():
            _walk_strings(child, f"{path}.{name}", node, name, found)
        return
    if isinstance(node, list):
        for index, child in enumerate(node):
            _walk_strings(child, f"{path}[{index}]", node, index, found)


def _tokenize(path: str) -> list[tuple[str | None, int | None]]:
    tokens: list[tuple[str | None, int | None]] = []
    cleaned = path.lstrip("$").lstrip(".")
    for match in _SEGMENT.finditer(cleaned):
        name, index = match.group(1), match.group(2)
        if name is not None:
            tokens.append((name, None))
        else:
            tokens.append((None, int(index) if index not in (None, "", "-") else None))
    return tokens
