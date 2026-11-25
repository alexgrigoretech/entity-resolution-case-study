import csv
import json

import pytest

from resolver.runner import run
from solve import main


def test_pipeline_submits_one_decision_per_input(tmp_path):
    db = tmp_path / "db"
    inputs = tmp_path / "inputs"
    out = tmp_path / "out"
    db.mkdir()
    inputs.mkdir()
    (db / "existing_organizations.csv").write_text(
        "id,name,description,domain,linkedin,crunchbase,x\no1,Zephyr Sloth,Sells alpine wool blankets,,,,\n",
        encoding="utf-8-sig",
    )
    (db / "existing_people.csv").write_text(
        "id,name,description,linkedin,crunchbase,x\np1,Lira Fallow,Partner at Amber Lantern Ventures.,,,\n",
        encoding="utf-8",
    )
    (inputs / "input_organizations.csv").write_text(
        "input_id,name,description\nz,Maple Ibex,Models coastal erosion\na,Zephyr Sloth Ltd,Sells alpine wool blankets\n"
        "c,Maple Ibex LLC,Models coastal erosion\n",
        encoding="utf-8-sig",
    )
    (inputs / "input_people.csv").write_text(
        "input_id,name,source_article\nx,Lira Fallow,Lira Fallow said Amber Lantern Ventures will invest.\n",
        encoding="utf-8",
    )

    stats = run(db, inputs, out, network=False)
    for kind, ids, matches, duplicates, fields in (
        (
            "organizations",
            ["z", "a", "c"],
            ["NEW", "o1", "NEW"],
            ["", "", "z"],
            ["domain", "linkedin", "crunchbase", "x"],
        ),
        ("people", ["x"], ["p1"], [""], ["linkedin", "crunchbase", "x"]),
    ):
        with (out / f"submission_{kind}.csv").open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == ["input_id", "match_id", "dup_of", *fields]
            rows = list(reader)
        assert [row["input_id"] for row in rows] == ids
        assert [row["match_id"] for row in rows] == matches
        assert [row["dup_of"] for row in rows] == duplicates
        assert all(row[field] == "" for row in rows for field in fields)

    report = json.loads((out / "run_stats.json").read_text(encoding="utf-8"))
    assert report["types"] == stats
    assert report["types"]["organizations"]["dup_groups"] == 1
    assert [row["input_id"] for row in stats["organizations"]["decisions"]] == ["z", "a", "c"]
    assert report["network"] == {}
    assert report["wall_seconds"] >= 0


def test_cli_reports_missing_database_then_processes_only_present_type(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    (tmp_path / "input_people.csv").write_text("input_id,name,source_article\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["solve.py", "--db", str(tmp_path), "--input", str(tmp_path), "--out", str(out)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "Missing database file" in capsys.readouterr().err

    (tmp_path / "existing_people.csv").write_text("id,name,description,linkedin,crunchbase,x\n", encoding="utf-8")
    main()
    assert (out / "submission_people.csv").exists()
    assert not (out / "submission_organizations.csv").exists()
