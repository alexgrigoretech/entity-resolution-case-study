"""Evaluate decision confidence and link thresholds on a completed run."""

import argparse
import csv
import json
import random
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resolver import config
from resolver.normalize import link_key
from resolver.records import KINDS, NEW, Kind, link_fields
from tools.link_agreement import CsvRow, LinkAgreement, chosen_link, table

BUCKET_EDGES = (0.5, 0.7, 0.8, 0.9)
BUCKETS = ("<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", ">=0.9")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CAUSES = (
    "Existing called new",
    "New matched to existing",
    "Wrong existing ID",
    "Group too small",
    "Group too big",
    "Invalid or missing",
)
NO_SELECTIONS = "no link selections in this run"
type Observation = tuple[float, bool]
type Entity = dict[str, list[Observation]]


def _read_csv(path: Path) -> list[CsvRow]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return [
            {(key or "").strip(): (value or "").strip() for key, value in row.items()} for row in csv.DictReader(source)
        ]


def _new_groups(rows: dict[str, CsvRow]) -> dict[str, set[str]]:
    parent = {input_id: input_id for input_id, row in rows.items() if row.get("match_id") == NEW}

    def root(input_id: str) -> str:
        while parent[input_id] != input_id:
            parent[input_id] = parent[parent[input_id]]
            input_id = parent[input_id]
        return input_id

    for input_id in parent:
        target = rows[input_id].get("dup_of", "")
        if target in parent:
            parent[root(input_id)] = root(target)
    by_root = defaultdict(set)
    for input_id in parent:
        by_root[root(input_id)].add(input_id)
    return {input_id: by_root[root(input_id)] for input_id in parent}


def _decision_error(match: str, expected: str, submitted_group: set[str], true_group: set[str], valid: bool) -> str:
    if not valid:
        return "Invalid or missing"
    if expected != NEW:
        if match == expected:
            return ""
        return "Existing called new" if match == NEW else "Wrong existing ID"
    if match != NEW:
        return "New matched to existing"
    if submitted_group == true_group:
        return ""
    return "Group too big" if submitted_group - true_group else "Group too small"


def _calibration(label: str, observations: list[Observation]) -> list[tuple[str, ...]]:
    bins: list[list[bool]] = [[] for _ in BUCKETS]
    for confidence, right in observations:
        bins[bisect_right(BUCKET_EDGES, confidence)].append(right)
    rows = []
    for bucket, results in zip(BUCKETS, bins, strict=True):
        count, right = len(results), sum(results)
        percentage = f"{100 * right / count:.1f}%" if count else "n/a"
        rows.append((label, bucket, str(count), str(right), percentage))
    return rows


def _net(observations: list[Observation], threshold: float) -> int:
    return sum(1 if correct else -1 for confidence, correct in observations if confidence >= threshold)


def _best(observations: list[Observation]) -> float:
    return max(THRESHOLDS, key=lambda threshold: (_net(observations, threshold), threshold))


def _thresholds(field: str, folds: list[list[Entity]]) -> tuple[str, ...]:
    fold_links = [[link for entity in fold for link in entity.get(field, [])] for fold in folds]
    all_links = [link for fold in fold_links for link in fold]
    held_out = 0
    for index, test in enumerate(fold_links):
        training = [link for other, fold in enumerate(fold_links) if other != index for link in fold]
        held_out += _net(test, _best(training))
    chosen, current = _best(all_links), config.LINK_MIN_CONF[field]
    return (
        field,
        f"{chosen:.1f}",
        str(_net(all_links, chosen)),
        str(held_out),
        f"{current:.1f}",
        str(_net(all_links, current)),
    )


def evaluate(out: Path, key: Path, folds: int = 5, seed: int = 7) -> None:
    stats = json.loads((out / "run_stats.json").read_text(encoding="utf-8"))
    db_dir = key if (key / "existing_organizations.csv").is_file() else key.parent / "candidate"
    decisions, errors = [], []
    entities: dict[tuple[Kind, str], Entity] = {}
    has_selections = False
    agreement = LinkAgreement()
    for kind in KINDS:
        truth = {row["input_id"]: row for row in _read_csv(key / f"answer_key_{kind}.csv")}
        db_path = db_dir / f"existing_{kind}.csv"
        existing = {row["id"] for row in _read_csv(db_path)} if db_path.is_file() else None
        submission_path = out / f"submission_{kind}.csv"
        submitted: dict[str, CsvRow] = {}
        if submission_path.is_file():
            for row in _read_csv(submission_path):
                if row.get("input_id") in truth:
                    submitted.setdefault(row["input_id"], row)
        for row in submitted.values():
            if row.get("match_id", "").upper() == NEW:
                row["match_id"] = NEW
        true_groups, submitted_groups = _new_groups(truth), _new_groups(submitted)
        audits = {row["input_id"]: row for row in stats["types"].get(kind, {}).get("decisions", [])}
        agreement.add(kind, truth, audits, submitted_groups)
        confidence_rows: list[Observation] = []
        causes: Counter[str] = Counter()
        for input_id, expected in truth.items():
            submission = submitted.get(input_id, {})
            match, dup = submission.get("match_id", ""), submission.get("dup_of", "")
            valid = bool(match) and (
                (not dup or dup in truth) if match == NEW else (existing is None or match in existing)
            )
            true_group = true_groups.get(input_id, set())
            cause = _decision_error(
                match, expected["match_id"], submitted_groups.get(input_id, set()), true_group, valid
            )
            audit = audits.get(input_id, {})
            confidence_rows.append((audit.get("confidence", 0.0), not cause))
            causes[cause] += 1
            selections = audit.get("link_selection", {})
            entity = entities.setdefault((kind, min(true_group)), {}) if true_group else None
            for field in link_fields(kind):
                if chosen := chosen_link(selections.get(field, {})):
                    has_selections = True
                    if entity is not None:
                        url, confidence = chosen
                        normalized = link_key(field, url)
                        correct = bool(normalized) and normalized == link_key(field, expected.get(field, ""))
                        entity.setdefault(field, []).append((confidence, correct))
        decisions.extend(_calibration(kind, confidence_rows))
        errors.append((kind, *(str(causes[cause]) for cause in CAUSES)))

    lines = ["# Evaluation", "", "## Decision calibration", ""]
    lines += table(("Type", "Confidence", "Rows", "Right", "Accuracy"), decisions)
    lines += ["", "## Link calibration", ""]
    if has_selections:
        links = [
            row
            for field in config.LINK_MIN_CONF
            for row in _calibration(field, [link for entity in entities.values() for link in entity.get(field, [])])
        ]
        lines += table(("Field", "Confidence", "Chosen", "Correct", "Precision"), links)
    else:
        lines.append(NO_SELECTIONS)

    lines += ["", "## Threshold cross-validation", ""]
    if has_selections:
        entity_ids = sorted(entities)
        random.Random(seed).shuffle(entity_ids)
        split = [[entities[entity] for entity in entity_ids[index::folds]] for index in range(folds)]
        lines += [f"{folds} folds, seed {seed}. Ties favor the higher threshold.", ""]
        lines += table(
            ("Field", "Chosen threshold", "In-sample net", "Held-out net", "Current threshold", "Current net"),
            [_thresholds(field, split) for field in config.LINK_MIN_CONF],
        )
    else:
        lines.append(NO_SELECTIONS)
    lines += ["", "## Link agreement", "", *agreement.lines()]
    lines += ["", "## Decision errors", "", *table(("Type", *CAUSES), errors)]
    (out / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Directory containing run outputs")
    parser.add_argument("--key", required=True, type=Path, help="Directory containing the answer key")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("folds must be at least 2")
    evaluate(args.out, args.key, args.folds, args.seed)


if __name__ == "__main__":
    main()
