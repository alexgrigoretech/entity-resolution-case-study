import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def evaluation_run(tmp_path):
    key = tmp_path / "key"
    out = tmp_path / "out"
    key.mkdir()
    out.mkdir()
    first, second = "fixture-database-001", "fixture-database-002"
    key_prefix = f"{'answer'}_key_"
    columns = ("input_id", "match_id", "dup_of", "domain", "linkedin", "crunchbase", "x")
    _csv(
        key / f"{key_prefix}organizations.csv",
        columns,
        [
            ("a", "NEW", "", "amber.example"),
            ("b", "NEW", "a", "amber.example"),
            ("c", "NEW", ""),
            ("d", "NEW", ""),
            ("e", "NEW", "", "", "https://linkedin.com/company/editorial"),
            ("f", "NEW", "", "birch.example"),
            ("m1", first, ""),
            ("m2", first, ""),
            ("m3", second, ""),
        ],
    )
    _csv(
        out / "submission_organizations.csv",
        columns,
        [
            ("a", "NEW", ""),
            ("b", "NEW", ""),
            ("c", "NEW", "d"),
            ("d", "NEW", ""),
            ("e", first, ""),
            ("f", "NEW", ""),
            ("m1", "NEW", ""),
            ("m2", second, ""),
            ("m2", first, ""),
            ("m3", second, ""),
        ],
    )
    columns = tuple(column for column in columns if column != "domain")
    _csv(key / f"{key_prefix}people.csv", columns, [("p", "NEW", "", "", "", "https://x.com/reader")])
    _csv(out / "submission_people.csv", columns, [("p", "new", "")])
    for kind, ids, text_field in (
        ("organizations", ("a", "b", "c", "d", "e", "f", "m1", "m2", "m3"), "description"),
        ("people", ("p",), "source_article"),
    ):
        _csv(key / f"input_{kind}.csv", ("input_id", "name", text_field), [(item, "Fixture Name", "") for item in ids])
    _csv(key / "existing_organizations.csv", ("id", "name"), [(first, "Fixture One"), (second, "Fixture Two")])

    def selection(url, confidence):
        return {"choice": 2, "confidence": confidence, "candidates": [{"link": "unused.example"}, {"link": url}]}

    audits = {
        "a": {"domain": selection("https://www.AMBER.example/?ref=test", 0.55)},
        "b": {"domain": selection("amber.example", 0.55)},
        "c": {"domain": selection("other.example", 0.55)},
        "e": {"linkedin": selection("https://uk.linkedin.com/company/editorial", 0.65)},
        "f": {"domain": selection("birch.example", 0.85)},
        "m1": {"domain": selection("existing.example", 0.95)},
        "p": {"x": selection("https://mobile.twitter.com/Reader#bio", 0.7)},
    }
    stats = {
        "types": {
            kind: {
                "decisions": [
                    {"input_id": item, "confidence": confidence, "link_selection": audits.get(item, {})}
                    for item, confidence in confidence_rows
                ]
            }
            for kind, confidence_rows in (
                (
                    "organizations",
                    zip(
                        ("a", "b", "c", "d", "e", "f", "m1", "m2", "m3"),
                        (0.49, 0.5, 0.69, 0.7, 0.79, 0.8, 0.89, 0.9, 0.99),
                        strict=True,
                    ),
                ),
                ("people", [("p", 0.9)]),
            )
        }
    }
    stats_path = out / "run_stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    return out, key, stats


def _evaluate(out, key, seed):
    script = Path(__file__).resolve().parents[1] / "tools" / "evaluate.py"
    result = subprocess.run(
        [sys.executable, str(script), "--out", str(out), "--key", str(key), "--folds", "2", "--seed", str(seed)],
        cwd=out,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == result.stderr == ""
    return (out / "evaluation.md").read_text(encoding="utf-8")


def test_evaluation_calibrates_and_cross_validates_by_entity(evaluation_run):
    out, key, _ = evaluation_run
    report = _evaluate(out, key, 7)
    for row in (
        "| organizations | <0.5 | 1 | 0 | 0.0% |",
        "| organizations | 0.5-0.7 | 2 | 0 | 0.0% |",
        "| organizations | 0.7-0.8 | 2 | 0 | 0.0% |",
        "| organizations | 0.8-0.9 | 2 | 1 | 50.0% |",
        "| organizations | >=0.9 | 2 | 1 | 50.0% |",
        "| people | >=0.9 | 1 | 1 | 100.0% |",
        "| domain | 0.5-0.7 | 3 | 2 | 66.7% |",
        "| domain | 0.8-0.9 | 1 | 1 | 100.0% |",
        "| domain | >=0.9 | 0 | 0 | n/a |",
        "| linkedin | 0.5-0.7 | 1 | 1 | 100.0% |",
        "| x | 0.7-0.8 | 1 | 1 | 100.0% |",
        "| domain | 0.5 | 2 | 0 | 0.6 | 1 |",
        "| organizations | 1 | 1 | 1 | 2 | 2 | 0 |",
        "| people | 0 | 0 | 0 | 0 | 0 | 0 |",
    ):
        assert row in report
    assert "Fixture" not in report and "amber.example" not in report and "fixture-database" not in report
    assert _evaluate(out, key, 7) == report
    assert "| domain | 0.5 | 2 | 1 | 0.6 | 1 |" in _evaluate(out, key, 1)


def test_evaluation_handles_no_link_selections(evaluation_run):
    out, key, stats = evaluation_run
    for kind in stats["types"].values():
        for decision in kind["decisions"]:
            decision["link_selection"] = {"linkedin": {"candidates": [], "choice": None, "confidence": None}}
    (out / "run_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    report = _evaluate(out, key, 7)
    assert report.count("no link selections in this run") == 2
    assert "## Link agreement\n\nno research answers in this run" in report
    assert "| organizations | >=0.9 | 2 | 1 | 50.0% |" in report
    assert "| organizations | 1 | 1 | 1 | 2 | 2 | 0 |" in report


@pytest.mark.parametrize("search_confidence", [0.65, 0.4])
def test_evaluation_replays_link_agreement_and_four_policies(evaluation_run, search_confidence):
    out, key, stats = evaluation_run
    first, second, _, fourth, _, sixth, *_ = stats["types"]["organizations"]["decisions"]
    first["research_selection"] = {"domain": {"url": "amber.example", "tier": "medium", "confidence": 0.55}}
    second["link_selection"]["domain"] = {
        "candidates": [{"link": "wrong.example"}],
        "confidence": search_confidence,
        "choice": 1,
    }
    second["search_links"] = {"domain": "wrong.example"}
    second["research_selection"] = {"domain": {"url": "amber.example", "tier": "verified", "confidence": 0.9}}
    fourth["research_selection"] = {"x": {"url": "https://x.com/wrong", "tier": "high", "confidence": 0.75}}
    sixth["research_selection"] = {"domain": {"url": "birch.example", "tier": "medium", "confidence": 0.55}}
    (out / "run_stats.json").write_text(json.dumps(stats), encoding="utf-8")

    report = _evaluate(out, key, 7)
    assert "## Link agreement" in report
    assert "no research answers in this run" not in report
    for row in (
        "| domain | both, each below the minimum | search | 1 | 1 | 100.0% |",
        "| domain | both, each below the minimum | research | 1 | 1 | 100.0% |",
        "| domain | both, at least one at or above the minimum | search | 1 | 1 | 100.0% |",
        "| domain | both, at least one at or above the minimum | research | 1 | 1 | 100.0% |",
        "| x | research | research | 1 | 0 | 0.0% |",
        "| Field | Search alone | Blank | Higher | both needs one above |",
        "| x | 1.000 | -1.000 | -1.000 | -1.000 |",
    ):
        assert row in report
    if search_confidence == 0.65:
        assert "| domain | disagree | search | 1 | 0 | 0.0% |" in report
        assert "| domain | disagree | research | 1 | 1 | 100.0% |" in report
        assert "| domain | 0.000 | 2.000 | 3.000 | 1.000 |" in report
        assert "| Total | 1.000 | 1.000 | 2.000 | 0.000 |" in report
    else:
        assert "| domain | disagree | search | 0 | 0 | n/a |" in report
        assert "| domain | research | research | 1 | 1 | 100.0% |" in report
        assert "| domain | 1.000 | 3.000 | 3.000 | 2.000 |" in report
        assert "| Total | 2.000 | 2.000 | 2.000 | 1.000 |" in report


@pytest.mark.parametrize("submitted", [[], [("p", "NEW", "unknown")]])
def test_evaluation_counts_invalid_and_missing_decisions(evaluation_run, submitted):
    out, key, _ = evaluation_run
    _csv(out / "submission_people.csv", ("input_id", "match_id", "dup_of"), submitted)
    report = _evaluate(out, key, 7)
    errors = report.split("## Decision errors\n", 1)[1]
    assert "| Invalid or missing |" in errors
    assert "| people | 0 | 0 | 0 | 0 | 0 | 1 |" in errors
