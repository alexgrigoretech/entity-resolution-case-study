"""Build and embed run data for the review report."""

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from . import config
from .candidates import Candidate
from .combine import CombinedLink
from .links import LinkOptions, Selection, reconcile
from .records import KINDS, NEW, Decision, InputRecord, Kind, link_fields
from .research import ResearchResult
from .review import borderline_fields, decision_flags

DATA_TOKEN = "/*__REPORT_DATA__*/null"
type RowOutcome = Literal["ok", "row_failed", "link_failed", "skipped"]


@dataclass(slots=True)
class Row:
    record: InputRecord
    candidates: list[Candidate]
    offline: Decision
    decision: Decision
    ambiguous: bool
    options: LinkOptions | None = None
    selections: dict[str, Selection] = field(default_factory=dict)
    outcome: RowOutcome = "ok"
    attempted: bool = False
    research: ResearchResult | None = None
    combined: dict[str, CombinedLink] = field(default_factory=dict)
    search_links: dict[str, str] = field(default_factory=dict)

    def apply_research(self) -> None:
        self.search_links = self.decision.links.copy()
        self.combined = reconcile(self.selections, self.research.selections if self.research is not None else {})
        for name, result in self.combined.items():
            if result.agreement == "search":
                continue
            if result.emitted is None:
                self.decision.links.pop(name, None)
                self.decision.link_confidences.pop(name, None)
            else:
                self.decision.links[name] = result.emitted.url
                self.decision.link_confidences[name] = result.emitted.confidence

    def link_flags(self) -> list[str]:
        return [f"link_disagree:{name}" for name, result in self.combined.items() if result.agreement == "disagree"]

    def report(self) -> dict[str, Any]:
        options = self.options or {}
        audit = {
            name: asdict(self.selections.get(name) or Selection(options.get(name, [])))
            for name in link_fields(self.record.kind)
        }
        researched = {}
        if self.research:
            for name, selection in self.research.selections.items():
                evidence = asdict(selection)
                if not selection.confirmation_only:
                    del evidence["confirmation_only"]
                researched[name] = evidence
        details = {
            **asdict(self.decision),
            "needs_review": self.decision.needs_review,
            "link_selection": audit,
            "search_links": self.search_links,
            "research_selection": researched,
            "agreement": {name: result.agreement for name, result in self.combined.items()},
        }
        flags = self.link_flags()
        if self.outcome == "link_failed":
            flags.append("link_error")
        if flags:
            details["flags"] = flags
        return details


def _candidate(candidate: Candidate, kind: Kind) -> dict[str, Any]:
    record, features = candidate.record, candidate.features
    return {
        "id": record.id,
        "name": record.name,
        "description": record.description,
        "links": {name: record.links.get(name, "") for name in link_fields(kind)},
        "name_score": features["name_score"],
        "text_score": features["text_score"],
        "embed_score": features.get("embed_score", 0.0),
        "exact_name": features["exact_name"],
        "link_hit": features.get("link_hit", []),
    }


def _group(row: Row) -> str:
    return row.decision.dup_of or row.record.input_id


def _source(row: Row, name: str, result: CombinedLink, url: str) -> str:
    if result.agreement == "research":
        return "research only"
    selection = row.selections[name]
    hit = selection.candidates[selection.choice - 1]
    if result.agreement == "disagree" and url != hit.link:
        return "research, search disagreed"
    suffix = {"both": "confirmed by research", "search": "search only", "disagree": "research disagreed"}[
        result.agreement
    ]
    return f"{hit.source}, {suffix}" if hit.source else suffix


def _type_rows(kind: Kind, rows: list[Row], names: dict[str, str]) -> list[dict[str, Any]]:
    input_names = {row.record.input_id: row.record.name for row in rows}
    group_roots = {row.decision.dup_of for row in rows if row.decision.dup_of}
    sources: dict[tuple[str, str, str, float], str] = {}
    for row in rows:
        for name, result in row.combined.items():
            if answer := result.emitted:
                sources.setdefault(
                    (_group(row), name, answer.url, answer.confidence), _source(row, name, result, answer.url)
                )
        for field_name, selection in row.selections.items():
            if field_name not in row.combined and selection.choice is not None and selection.confidence is not None:
                hit = selection.candidates[selection.choice - 1]
                sources.setdefault((_group(row), field_name, hit.link, selection.confidence), hit.source)

    result = []
    for row in rows:
        record, decision = row.record, row.decision
        flags = decision_flags(decision)
        if row.outcome == "link_failed":
            flags.append("link_error")
        disagreements = row.link_flags()
        flags.extend(disagreements)
        links, link_candidates = {}, {}
        fields = link_fields(kind)
        borderline = borderline_fields(decision, fields)
        flags.extend(f"borderline_link:{name}" for name in borderline)
        group_link = row.outcome in ("link_failed", "skipped") and (decision.dup_of or record.input_id in group_roots)
        for field_name in fields:
            if url := decision.links.get(field_name):
                confidence = decision.link_confidences.get(field_name, 0.0)
                source = (
                    "from a duplicate row"
                    if group_link
                    else sources.get((_group(row), field_name, url, confidence), "")
                )
                links[field_name] = {
                    "url": url,
                    "confidence": confidence,
                    "source": source,
                }
            selection = row.selections.get(field_name)
            hits = selection.candidates if selection else (row.options or {}).get(field_name, [])
            if hits:
                link_candidates[field_name] = [
                    {
                        "url": hit.link,
                        "title": hit.title,
                        "source": hit.source,
                        "chosen": selection is not None and index == selection.choice,
                    }
                    for index, hit in enumerate(hits, 1)
                ]
        result.append(
            {
                "type": kind,
                "input_id": record.input_id,
                "name": record.name,
                "text": record.text,
                "decision": decision.match_id,
                "decided_name": names.get(decision.match_id, "") if decision.match_id != NEW else "",
                "dup_of": decision.dup_of,
                "dup_of_name": input_names.get(decision.dup_of, ""),
                "confidence": decision.confidence,
                "model": decision.model,
                "reason": decision.reason,
                "flags": flags,
                "needs_review": bool(decision.needs_review or borderline or disagreements),
                "employer": decision.employer,
                "role": decision.role,
                "candidates": [_candidate(candidate, kind) for candidate in row.candidates[:5]],
                "links": links,
                "link_candidates": link_candidates,
            }
        )
    return result


def build_payload(stats: dict[str, Any], rows_by_type: dict[Kind, tuple[list[Row], dict[str, str]]]) -> dict[str, Any]:
    models = stats.get("models", {})
    active_stages = stats.get("active_stages", {})
    links_emitted = stats.get("links_emitted", {})
    types = stats.get("types", {})
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "stats": {
            "wall_seconds": stats.get("wall_seconds", 0.0),
            "deadline_hit": stats.get("deadline_hit", False),
            "active_stages": {
                stage: active_stages.get(stage, False)
                for stage in ("embeddings", "adjudication", "search", "links", "model_dedupe", "quoted_queries")
            },
            "models": models,
            "usd_total": sum((model.get("cost", 0.0) for model in models.values()), 0.0),
            "serper_queries": stats.get("serper_queries", 0),
            "escalation_count": stats.get("escalation_count", 0),
            "escalation_rate": stats.get("escalation_rate", 0.0),
            "disagreement_count": stats.get("disagreement_count", 0),
            "review_queue_size": stats.get("review_queue_size", 0),
            "links_emitted": {field: links_emitted.get(field, 0) for field in config.LINK_MIN_CONF},
            "types": {
                kind: {key: types.get(kind, {}).get(key, 0) for key in ("rows", "matched", "new", "dup_groups")}
                for kind in KINDS
            },
        },
        "rows": [row for kind in KINDS if kind in rows_by_type for row in _type_rows(kind, *rows_by_type[kind])],
    }


def render(payload: dict[str, Any], template_path: Path, out_path: Path) -> None:
    template = template_path.read_text(encoding="utf-8")
    if DATA_TOKEN not in template:
        raise ValueError("Report template is missing its data token")
    data = (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", r"\u003c")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )
    out_path.write_text(template.replace(DATA_TOKEN, data, 1), encoding="utf-8")
