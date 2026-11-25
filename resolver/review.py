"""Write decisions and borderline links that need human review."""

import csv
from collections.abc import Iterable
from pathlib import Path

from . import config
from .records import Decision, InputRecord, Kind

ReviewEntry = tuple[Kind, InputRecord, Decision, str]


def decision_flags(decision: Decision) -> list[str]:
    flags = []
    if decision.confidence < config.REVIEW_BELOW:
        flags.append("low_confidence")
    if decision.models_disagree:
        flags.append("models_disagree")
    if decision.is_fallback:
        flags.append("fallback")
    return flags


def borderline_fields(decision: Decision, fields: Iterable[str] | None = None) -> list[str]:
    return [
        field
        for field in (config.LINK_MIN_CONF if fields is None else fields)
        if decision.links.get(field)
        and decision.link_confidences.get(field, 0.0) <= config.LINK_MIN_CONF[field] + 0.1 + 1e-9
    ]


def write_review(
    path: Path, entries: list[ReviewEntry], link_flags: dict[tuple[Kind, str], list[str]] | None = None
) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(
            ("type", "input_id", "name", "decision", "decided_name", "confidence", "model", "reason", "flags")
        )
        for kind, record, decision, decided_name in sorted(entries, key=lambda entry: entry[2].confidence):
            borderline = [f"borderline_link:{field}" for field in borderline_fields(decision)]
            disagreements = (link_flags or {}).get((kind, record.input_id), [])
            if decision.needs_review or borderline or disagreements:
                writer.writerow(
                    (
                        kind,
                        record.input_id,
                        record.name,
                        decision.match_id,
                        decided_name,
                        decision.confidence,
                        decision.model,
                        decision.reason,
                        ";".join(decision_flags(decision) + disagreements + borderline),
                    )
                )
                count += 1
    return count
