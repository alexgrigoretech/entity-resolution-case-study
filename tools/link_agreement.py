"""Replay link policies and report agreement on a completed run."""

from collections import defaultdict
from typing import Any

from resolver import config
from resolver.combine import AGREEMENTS, LinkAnswer, combine
from resolver.normalize import link_key
from resolver.records import NEW, Kind, link_fields

BOTH_ABOVE = "both, at least one at or above the minimum"
BOTH_BELOW = "both, each below the minimum"
AGREEMENT_CLASSES = (BOTH_ABOVE, BOTH_BELOW, *AGREEMENTS[1:])
POLICIES = ("Search alone", "Blank", "Higher", "both needs one above")
type CsvRow = dict[str, str]


def chosen_link(selection: dict[str, Any]) -> tuple[str, float] | None:
    choice, confidence = selection.get("choice"), selection.get("confidence")
    candidates = selection.get("candidates", [])
    if type(choice) is int and 1 <= choice <= len(candidates) and type(confidence) in (int, float):
        return candidates[choice - 1]["link"], confidence
    return None


def table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    return ["| " + " | ".join(row) + " |" for row in (headers, ("-",) * len(headers), *rows)]


class LinkAgreement:
    def __init__(self) -> None:
        self.answers: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
        self.points: dict[tuple[str, str], float] = defaultdict(float)
        self.totals: dict[str, float] = defaultdict(float)
        self.has_research = False

    def add(self, kind: Kind, truth: dict[str, CsvRow], audits: dict[str, Any], groups: dict[str, set[str]]) -> None:
        fields = link_fields(kind)
        roots = {item: min(group) for item, group in groups.items()}
        winners: dict[tuple[str, str, str], LinkAnswer] = {}
        for item, audit in audits.items():
            research = audit.get("research_selection", {})
            self.has_research |= any(answer.get("url") for answer in research.values())
            if item not in roots:
                continue
            expected = truth[item]
            for field in fields:
                chosen = chosen_link(audit.get("link_selection", {}).get(field, {}))
                search = LinkAnswer(*chosen) if chosen else None
                evidence = research.get(field, {})
                responder = (
                    LinkAnswer(evidence["url"], evidence["confidence"], evidence.get("confirmation_only", False))
                    if evidence.get("url")
                    else None
                )
                blank = combine(field, search, responder, "blank")
                agreement = blank.agreement if blank else None
                if agreement == "both":
                    above = max(search.confidence, responder.confidence) >= config.LINK_MIN_CONF[field]
                    agreement = BOTH_ABOVE if above else BOTH_BELOW
                if blank and expected["match_id"] == NEW:
                    target = link_key(field, expected.get(field, ""))
                    for source, answer in (("search", search), ("research", responder)):
                        if answer and blank.agreement in (source, "both", "disagree"):
                            self.answers[field, agreement, source].append(link_key(field, answer.url) == target)
                results = (
                    combine(field, search, None, "blank"),
                    blank,
                    combine(field, search, responder, "higher"),
                    None if agreement == BOTH_BELOW else blank,
                )
                for policy, result in zip(POLICIES, results, strict=True):
                    key = roots[item], field, policy
                    if (
                        result
                        and (answer := result.emitted)
                        and (key not in winners or answer.confidence > winners[key].confidence)
                    ):
                        winners[key] = answer
        for item, root in roots.items():
            expected = truth[item]
            if expected["match_id"] != NEW:
                continue
            weight = 1 / max(1, sum(bool(expected.get(field)) for field in fields))
            for policy in POLICIES:
                total = 0.0
                for field in fields:
                    if answer := winners.get((root, field, policy)):
                        correct = link_key(field, answer.url) == link_key(field, expected.get(field, ""))
                        points = weight if correct else -weight
                        self.points[field, policy] += points
                        total += points
                self.totals[policy] += max(-1.0, min(1.0, total))

    def lines(self) -> list[str]:
        if not self.has_research:
            return ["no research answers in this run"]
        rows = []
        for field in config.LINK_MIN_CONF:
            for agreement in AGREEMENT_CLASSES:
                for source in ("search", "research"):
                    if agreement in ("search", "research") and agreement != source:
                        continue
                    results = self.answers[field, agreement, source]
                    right, count = sum(results), len(results)
                    precision = f"{100 * right / count:.1f}%" if count else "n/a"
                    rows.append((field, agreement, source, str(count), str(right), precision))
        return [
            *table(("Field", "Agreement", "Answer", "Links", "Correct", "Precision"), rows),
            "",
            "Policy replay uses the submitted duplicate groups. Net link points per field precede the grader's per-row cap. The total applies the cap.",
            "",
            *table(
                ("Field", *POLICIES),
                [
                    (field, *(f"{self.points[field, policy]:.3f}" for policy in POLICIES))
                    for field in config.LINK_MIN_CONF
                ]
                + [("Total", *(f"{self.totals[policy]:.3f}" for policy in POLICIES))],
            ),
        ]
