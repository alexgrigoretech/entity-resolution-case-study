"""Record types and CSV formats for the resolver."""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from . import config

Kind = Literal["organizations", "people"]
KINDS: tuple[Kind, ...] = ("organizations", "people")
NEW = "NEW"
SOCIAL_FIELDS = ("linkedin", "crunchbase", "x")


@dataclass(frozen=True, slots=True)
class DatabaseRecord:
    id: str
    name: str
    description: str
    links: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InputRecord:
    input_id: str
    name: str
    text: str
    kind: Kind = "organizations"


@dataclass(slots=True)
class Decision:
    input_id: str
    match_id: str
    confidence: float
    reason: str
    dup_of: str = ""
    links: dict[str, str] = field(default_factory=dict)
    model: str = ""
    employer: str = ""
    role: str = ""
    fast_confidence: float | None = None
    strong_confidence: float | None = None
    escalated: bool = False
    models_disagree: bool = False
    is_fallback: bool = False
    link_confidences: dict[str, float] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return self.confidence < config.REVIEW_BELOW or self.models_disagree or self.is_fallback


def link_fields(kind: Kind) -> tuple[str, ...]:
    return ("domain", *SOCIAL_FIELDS) if kind == "organizations" else SOCIAL_FIELDS


def _read_csv(path: Path, id_column: str) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for raw in reader:
            row = {key.strip(): (value or "").strip() for key, value in raw.items() if key is not None}
            if row.get(id_column):
                rows.append(row)
            else:
                logging.getLogger(__name__).warning(
                    "Skipping blank %s in %s at line %s", id_column, path, reader.line_num
                )
    return rows


def read_database(path: Path, kind: Kind) -> list[DatabaseRecord]:
    fields = link_fields(kind)
    return [
        DatabaseRecord(row["id"], row["name"], row["description"], {field: row.get(field, "") for field in fields})
        for row in _read_csv(path, "id")
    ]


def read_inputs(path: Path, kind: Kind) -> list[InputRecord]:
    text_field = "description" if kind == "organizations" else "source_article"
    return [InputRecord(row["input_id"], row["name"], row[text_field], kind) for row in _read_csv(path, "input_id")]


def write_submission(path: Path, kind: Kind, decisions: list[Decision]) -> None:
    fields = link_fields(kind)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("input_id", "match_id", "dup_of", *fields))
        for decision in decisions:
            writer.writerow(
                (
                    decision.input_id,
                    decision.match_id,
                    decision.dup_of,
                    *(decision.links.get(field, "") for field in fields),
                )
            )
