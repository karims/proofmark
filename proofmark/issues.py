from dataclasses import dataclass
from enum import Enum


class Stage(str, Enum):
    """Which layer of the pipeline raised an issue.

    The stage determines what can fix it. ``SCHEMA`` issues are the library's own
    fault or the caller's and never reach a provider. ``STRUCTURAL`` issues mean
    the payload does not match the schema. ``SEMANTIC`` issues mean the payload
    matches the schema but is still wrong -- the case schema validation alone
    cannot catch, and the reason this library exists.
    """

    SCHEMA = "schema"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class Issue:
    """A single defect, addressed to a location in the payload.

    ``path`` uses JSON-pointer-ish dotted notation (``$.budget.items[2].amount``)
    so it can be quoted straight back to a model during repair without further
    formatting.
    """

    path: str
    message: str
    stage: Stage = Stage.SEMANTIC
    code: str | None = None

    def format(self) -> str:
        return f"{self.path}: {self.message}"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "message": self.message,
            "stage": self.stage.value,
            "code": self.code,
        }


def issue(path: str, message: str, stage: Stage = Stage.SEMANTIC, code: str | None = None) -> Issue:
    return Issue(path=path, message=message, stage=stage, code=code)


def format_issues(issues: list[Issue]) -> list[str]:
    return [item.format() for item in issues]
