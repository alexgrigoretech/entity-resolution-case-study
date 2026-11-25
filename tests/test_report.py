import csv
import json
import sys
from datetime import UTC, datetime

import pytest

from resolver import config, runner
from resolver.candidates import Candidate
from resolver.links import Selection
from resolver.records import NEW, DatabaseRecord, Decision, InputRecord, Kind
from resolver.report import DATA_TOKEN, Row, build_payload, render
from resolver.research import ResearchResult, ResearchSelection
from resolver.review import write_review
from resolver.search import Hit
from solve import main


def test_payload_shape_and_review_evidence(tmp_path):
    organization = InputRecord("a", "Ámber Grove", "Full description\nwith a second line.")
    duplicate = InputRecord("b", "Amber Grove Ltd", "Other input text")
    person = InputRecord("p", "Morgan Reed", "Full article", "people")
    candidates = [
        Candidate(
            DatabaseRecord(str(index), f"Candidate {index}", "Description"),
            {"name_score": 98.0 - index, "text_score": 0.4, "exact_name": False, "name_rivals": 0},
        )
        for index in range(6)
    ]
    domain = "amber.example"
    decision = Decision(
        "a",
        NEW,
        0.4,
        "fallback:fixture",
        links={"domain": domain},
        model="fixture",
        models_disagree=True,
        is_fallback=True,
    )
    decision.link_confidences["domain"] = 0.7
    dup_decision = Decision("b", NEW, 0.8, "fixture", dup_of="a", links={"domain": domain})
    dup_decision.link_confidences["domain"] = 0.7
    match = Decision("p", "db-p", 0.9, "fixture", employer="Amber Grove", role="Editor")
    selections = {"domain": Selection([Hit("Home", domain, "", "organic")], 1, 0.7)}
    options = {"x": [Hit("Feed", "https://x.com/amber", "", "organic")]}
    stats = {
        "models": {"first": {"cost": 0.25}, "second": {"cost": 0.125}},
        "network": {"other": {}},
        "active_stages": {"quoted_queries": True},
        "types": {"organizations": {"rows": 2, "matched": 0, "new": 2, "dup_groups": 1, "decisions": []}},
    }
    payload = build_payload(
        stats,
        {
            "organizations": (
                [
                    Row(organization, candidates, decision, decision, False, options, selections),
                    Row(duplicate, [], dup_decision, dup_decision, False),
                ],
                {},
            ),
            "people": (
                [Row(person, [], match, match, False, options={"domain": selections["domain"].candidates})],
                {"db-p": "Morgan L. Reed"},
            ),
        },
    )
    assert set(payload) == {"generated_at", "stats", "rows"}
    assert datetime.fromisoformat(payload["generated_at"]).tzinfo == UTC
    assert payload["stats"] == {
        "wall_seconds": 0.0,
        "deadline_hit": False,
        "active_stages": {
            **dict.fromkeys(("embeddings", "adjudication", "search", "links", "model_dedupe"), False),
            "quoted_queries": True,
        },
        "models": stats["models"],
        "usd_total": 0.375,
        "serper_queries": 0,
        "escalation_count": 0,
        "escalation_rate": 0.0,
        "disagreement_count": 0,
        "review_queue_size": 0,
        "links_emitted": dict.fromkeys(("domain", "linkedin", "crunchbase", "x"), 0),
        "types": {
            "organizations": {"rows": 2, "matched": 0, "new": 2, "dup_groups": 1},
            "people": {"rows": 0, "matched": 0, "new": 0, "dup_groups": 0},
        },
    }
    org, dup, person_row = payload["rows"]
    assert org == {
        "type": "organizations",
        "input_id": "a",
        "name": organization.name,
        "text": organization.text,
        "decision": NEW,
        "decided_name": "",
        "dup_of": "",
        "dup_of_name": "",
        "confidence": 0.4,
        "model": "fixture",
        "reason": "fallback:fixture",
        "flags": ["low_confidence", "models_disagree", "fallback", "borderline_link:domain"],
        "needs_review": True,
        "employer": "",
        "role": "",
        "candidates": [
            {
                "id": str(index),
                "name": f"Candidate {index}",
                "description": "Description",
                "links": {"domain": "", "linkedin": "", "crunchbase": "", "x": ""},
                "name_score": 98.0 - index,
                "text_score": 0.4,
                "embed_score": 0.0,
                "exact_name": False,
                "link_hit": [],
            }
            for index in range(5)
        ],
        "links": {"domain": {"url": domain, "confidence": 0.7, "source": "organic"}},
        "link_candidates": {
            "domain": [{"url": domain, "title": "Home", "source": "organic", "chosen": True}],
            "x": [{"url": "https://x.com/amber", "title": "Feed", "source": "organic", "chosen": False}],
        },
    }
    queue = tmp_path / "queue.csv"
    assert write_review(queue, [("organizations", organization, decision, "")]) == 1
    with queue.open(encoding="utf-8", newline="") as source:
        assert next(csv.DictReader(source))["flags"].split(";") == org["flags"]
    assert dup["dup_of_name"] == organization.name
    assert dup["links"] == org["links"]
    assert person_row["decided_name"] == "Morgan L. Reed"
    assert (person_row["employer"], person_row["role"]) == ("Amber Grove", "Editor")
    assert person_row["links"] == person_row["link_candidates"] == {}


@pytest.mark.parametrize(
    ("research_confidence", "source", "flags"),
    [
        (0.75, "organic, research disagreed", ["link_disagree:x", "borderline_link:x"]),
        (0.9, "research, search disagreed", ["link_disagree:x"]),
    ],
)
def test_payload_names_all_link_agreements(monkeypatch, research_confidence, source, flags):
    monkeypatch.setattr(config, "DISAGREE_POLICY", "higher")
    record = InputRecord("row", "Tin Orchid", "Fixture")
    decision = Decision("row", NEW, 0.9, "fixture")
    urls = {
        "domain": "orchid.example",
        "linkedin": "https://www.linkedin.com/company/tin-orchid",
        "crunchbase": "https://www.crunchbase.com/organization/tin-orchid",
        "x": "https://x.com/TinOrchid",
    }
    selections = {
        name: Selection([Hit("", urls[name], "", source)], 1, confidence)
        for name, source, confidence in (
            ("domain", "organic", 0.7),
            ("linkedin", "structured", 1.0),
            ("x", "organic", 0.8),
        )
    }
    decision.links = {name: selection.emitted(name) for name, selection in selections.items()}
    decision.link_confidences = {name: selection.confidence for name, selection in selections.items()}
    research = ResearchResult(
        {
            "domain": ResearchSelection(urls["domain"], "high", 0.75),
            "crunchbase": ResearchSelection(urls["crunchbase"], "high", 0.75),
            "x": ResearchSelection("https://x.com/Other", "high", research_confidence),
        },
        4,
    )
    row = Row(record, [], decision, decision, False, selections=selections, research=research)
    row.apply_research()

    payload = build_payload({}, {"organizations": ([row], {})})["rows"][0]
    assert {name: link["source"] for name, link in payload["links"].items()} == {
        "domain": "organic, confirmed by research",
        "linkedin": "structured, search only",
        "crunchbase": "research only",
        "x": source,
    }
    assert payload["flags"] == flags
    assert payload["needs_review"]


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (
            "organizations",
            {
                "domain": "Example.COM",
                "linkedin": "https://www.linkedin.com/company/fixture/",
                "crunchbase": "",
                "x": "https://twitter.com/Fixture",
            },
        ),
        (
            "people",
            {
                "linkedin": "https://www.linkedin.com/in/fixture/",
                "crunchbase": "https://www.crunchbase.com/person/fixture",
                "x": "",
            },
        ),
    ],
)
def test_payload_includes_database_candidate_links(kind: Kind, expected: dict[str, str]) -> None:
    record = InputRecord("input", "Fixture", "Input text", kind)
    database_links = {"domain": "Example.COM", **expected}
    candidate = Candidate(
        DatabaseRecord("db", "Fixture", "Database text", database_links),
        {"name_score": 100.0, "text_score": 0.5, "exact_name": True},
    )
    decision = Decision(record.input_id, NEW, 0.9, "fixture", links={"x": "https://x.com/other"})
    row = Row(record, [candidate], decision, decision, False)

    payload = build_payload({}, {kind: ([row], {})})

    assert payload["rows"][0]["candidates"][0]["links"] == expected


def test_render_escapes_script_tag_and_preserves_text(tmp_path):
    template = tmp_path / "template.html"
    template.write_text(f"<script>const REPORT = {DATA_TOKEN};</script>", encoding="utf-8")
    target = tmp_path / "report.html"
    payload = {"rows": [{"text": "Café </script><b>marker</b>"}]}
    render(payload, template, target)
    html = target.read_text(encoding="utf-8")
    assert html.count("</script>") == 1
    assert r"Café \u003c/script>" in html
    assert json.loads(html.removeprefix("<script>const REPORT = ").removesuffix(";</script>")) == payload


@pytest.mark.parametrize(
    "text",
    ["see <!-- <script> x", "</script><script>alert(1)</script>", "line\u2028separator", "paragraph\u2029separator"],
)
def test_render_round_trips_script_sensitive_text(tmp_path, text):
    opening, closing = "<script>const REPORT = ", ";</script>"
    template = tmp_path / "template.html"
    template.write_text(f"{opening}{DATA_TOKEN}{closing}", encoding="utf-8")
    target = tmp_path / "report.html"
    render({"rows": [{"text": text}]}, template, target)

    html = target.read_text(encoding="utf-8")
    assert html.startswith(opening) and html.endswith(closing)
    data = html[len(opening) : -len(closing)]
    assert all(char not in data for char in "<\u2028\u2029")
    assert json.loads(data)["rows"][0]["text"] == text


def test_render_rejects_missing_token(tmp_path):
    template = tmp_path / "template.html"
    template.write_text("<html></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="missing its data token"):
        render({}, template, tmp_path / "report.html")


@pytest.mark.parametrize(("template_exists", "disabled"), [(True, False), (False, False), (True, True)])
def test_cli_report_behavior(tmp_path, monkeypatch, caplog, template_exists, disabled):
    template = tmp_path / "template.html"
    if template_exists:
        template.write_text(f"<script>{DATA_TOKEN}</script>", encoding="utf-8")
    monkeypatch.setattr(runner, "REPORT_TEMPLATE", template)
    (tmp_path / "existing_people.csv").write_text("id,name,description\n", encoding="utf-8")
    (tmp_path / "input_people.csv").write_text(
        "input_id,name,source_article\np,Morgan Reed,Full article\n", encoding="utf-8"
    )
    out = tmp_path / "out"
    args = ["solve.py", "--db", str(tmp_path), "--input", str(tmp_path), "--out", str(out), "--no-network"]
    monkeypatch.setattr(sys, "argv", [*args, "--no-report"] if disabled else args)
    main()
    assert (out / "submission_people.csv").is_file()
    assert (out / "run_stats.json").is_file()
    warnings = [record for record in caplog.records if record.name == runner.LOGGER.name]
    assert len(warnings) == int(not template_exists and not disabled)
    target = out / "report.html"
    assert target.is_file() == (template_exists and not disabled)
    if warnings:
        assert "Report template is missing" in warnings[0].message
    if target.is_file():
        html = target.read_text(encoding="utf-8")
        payload = json.loads(html.removeprefix("<script>").removesuffix("</script>"))
        assert payload["rows"][0]["text"] == "Full article"
