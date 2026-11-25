import asyncio
import csv
import json
import sqlite3
from collections import Counter
from contextlib import closing

import httpx
import pytest

from resolver import config, pipeline
from resolver.links import Selection
from resolver.runner import run
from resolver.search import Hit


@pytest.fixture
def folders(tmp_path):
    db = tmp_path / "db"
    inputs = tmp_path / "inputs"
    out = tmp_path / "out"
    cache = tmp_path / "cache.sqlite"
    db.mkdir()
    inputs.mkdir()
    (db / "existing_organizations.csv").write_text(
        "id,name,description,domain,linkedin,crunchbase,x\n"
        "o1,Birch Foundry,Makes copper wind chimes,birch.example,,,\n",
        encoding="utf-8",
    )
    (db / "existing_people.csv").write_text(
        "id,name,description,linkedin,crunchbase,x\n"
        "p1,Cora Grove,Director at Silk Dove Works,https://uk.linkedin.com/in/CoraGrove,,\n"
        "p2,Etta Vale,Analyst at Twelve Finch Studio,,,\n",
        encoding="utf-8",
    )
    (inputs / "input_organizations.csv").write_text(
        "input_id,name,description\na,Birch Foundry,Makes copper wind chimes\n"
        "b,Maple Circuit,Builds portable radios\nc,Maple Circuit,Builds portable radios\n",
        encoding="utf-8",
    )
    (inputs / "input_people.csv").write_text(
        "input_id,name,source_article\np,Cora Grove,Cora Grove presented a new design.\n"
        "q,Etta Vale,Etta Vale discussed Twelve Finch Studio and other studios.\n",
        encoding="utf-8",
    )
    return db, inputs, out, cache


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-placeholder")
    monkeypatch.setenv("SERPER_API_KEY", "serper-placeholder")
    monkeypatch.setattr(config, "QUOTE_QUERIES", False)
    monkeypatch.setattr(config, "RESEARCH_ENABLED", False)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def run_mock(folders, handler, network=True, **options):
    db, inputs, out, cache = folders
    stats = run(db, inputs, out, network=network, cache_path=cache, transport=httpx.MockTransport(handler), **options)
    return stats, json.loads((out / "run_stats.json").read_text(encoding="utf-8"))


def report_row(folders, input_id):
    html = (folders[2] / "report.html").read_text(encoding="utf-8")
    payload, _ = json.JSONDecoder().raw_decode(html.split("const REPORT = ", 1)[1])
    return next(row for row in payload["rows"] if row["input_id"] == input_id)


def mock_service(requests, *, embedding_enabled=True):
    def handler(request):
        if request.method == "GET":
            assert request.url.host in ("birch.example", "maple.example")
            slug = request.url.host.split(".")[0]
            requests.append(("homepage", slug))
            html = f'<a href="https://linkedin.com/company/{slug}">LinkedIn</a>' + " " * 500_000
            return httpx.Response(200, html=html)
        body = json.loads(request.content)
        if request.url == "https://google.serper.dev/search":
            query = body["q"]
            requests.append(("search", query))
            if query.startswith("Cora Grove"):
                url = "https://www.linkedin.com/in/CoraGrove"
            elif query.startswith("Etta Vale Cedar Atelier"):
                url = "https://x.com/EttaV"
            elif query.startswith("Etta Vale"):
                url = "https://www.linkedin.com/in/etta-vale"
            elif query.startswith("Birch Foundry"):
                url = "https://birch.example"
            else:
                assert query.startswith("Maple Circuit")
                url = "https://maple.example"
            return httpx.Response(200, json={"organic": [{"title": "Synthetic result", "link": url}]})
        assert request.url.host == "openrouter.ai"
        if request.url.path.endswith("/embeddings"):
            requests.append(("probe" if body["input"] == ["ok"] else "embed", len(body["input"])))
            return httpx.Response(
                200, json={"data": [{"index": i, "embedding": [1, 2]} for i in range(len(body["input"]))]}
            )
        assert request.url.path.endswith("/chat/completions")
        schema = body["response_format"]["json_schema"]["name"]
        if schema == "preflight":
            requests.append(("probe", body["model"]))
            assert [message["content"] for message in body["messages"]] == ["Reply with JSON.", "ok"]
            return httpx.Response(
                200, json={"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {"cost": 0.01}}
            )
        prompt = json.loads(body["messages"][1]["content"])
        name = prompt["name"]
        if schema == "entity_match":
            requests.append(("match", name))
            strong = body["model"] == config.MODEL_STRONG
            decision, confidence, employer = "NEW", 0.91, ""
            if name == "Cora Grove":
                first = next(iter(prompt["candidates"].values()))
                if any(kind == "search" for kind, _ in requests):
                    assert first["link_hit"] == ["linkedin"]
                assert first["embed_score"] == pytest.approx(float(embedding_enabled))
                decision, confidence = ("p1", 0.92) if strong else ("NEW", 0.73)
            if name == "Etta Vale":
                employer = "Cedar Atelier"
            answer = {
                "decision": decision,
                "confidence": confidence,
                "reason": "Synthetic evidence.",
                "employer": employer,
                "role": "Analyst",
            }
        else:
            assert schema == "entity_links"
            requests.append(("choose", name))
            answer = {
                field: {"choice": 1 if options else None, "confidence": 0.64 if field == "domain" else 0.8}
                for field, options in prompt["candidates"].items()
            }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(answer)}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    return handler


def test_network_pipeline_and_review(folders, keys):
    _, _, out, cache = folders
    calls = []
    stats, report = run_mock(folders, mock_service(calls))
    assert report["active_stages"] == {
        **dict.fromkeys(("embeddings", "adjudication", "search", "links", "model_dedupe"), True),
        "quoted_queries": config.QUOTE_QUERIES,
        "research": False,
    }
    assert report["types"] == stats
    assert report["escalation_count"] == report["disagreement_count"] == 1
    assert report["escalation_rate"] == 0.2
    assert report["serper_queries"] == 14
    assert report["links_emitted"] == {"domain": 3, "linkedin": 3, "crunchbase": 0, "x": 1}
    assert report["models"][config.MODEL_FAST]["cost"] > 0
    assert report["models"][config.MODEL_STRONG]["calls"] == 2
    assert report["models"][config.MODEL_EMBED]["calls"] == 5
    assert not report["deadline_hit"]
    assert calls.index(("match", "Birch Foundry")) < calls.index(("search", "Birch Foundry Makes copper wind chimes"))
    assert calls.index(("match", "Etta Vale")) < calls.index(("search", "Etta Vale Cedar Atelier site:linkedin.com/in"))
    assert Counter(calls)[("choose", "Maple Circuit")] == 2
    assert Counter(calls)[("homepage", "maple")] == 1
    orgs = read_rows(out / "submission_organizations.csv")
    assert [row["match_id"] for row in orgs] == ["NEW", "NEW", "NEW"]
    assert [row["dup_of"] for row in orgs] == ["", "", "b"]
    assert [row["domain"] for row in orgs] == ["birch.example", "maple.example", "maple.example"]
    assert [row["linkedin"] for row in orgs] == [
        f"https://www.linkedin.com/company/{slug}" for slug in ("birch", "maple", "maple")
    ]
    people = read_rows(out / "submission_people.csv")
    assert [row["match_id"] for row in people] == ["p1", "NEW"]
    assert people[1]["x"] == "https://x.com/EttaV"
    queue = read_rows(out / "review_queue.csv")
    assert len(queue) == report["review_queue_size"] == 5
    assert [row["input_id"] for row in queue] == ["a", "b", "c", "q", "p"]
    assert [row["flags"] for row in queue] == ["borderline_link:domain"] * 3 + ["borderline_link:x", "models_disagree"]
    assert queue[-1]["decided_name"] == "Cora Grove"
    assert queue[-1]["model"] == config.MODEL_STRONG
    audit = stats["organizations"]["decisions"][0]["link_selection"]["linkedin"]
    assert audit["candidates"][0]["source"] == "homepage"
    assert audit["choice"] == 1 and audit["confidence"] == pytest.approx(0.95)
    with closing(sqlite3.connect(cache)) as connection:
        for (response,) in connection.execute("SELECT response FROM responses WHERE namespace = 'web'"):
            assert len(json.loads(response)) == 500_000
    assert b"router-placeholder" not in cache.read_bytes()
    assert b"serper-placeholder" not in cache.read_bytes()

    probes = []
    handle = mock_service(probes)

    def cached_service(request):
        body = json.loads(request.content)
        schema = body.get("response_format", {}).get("json_schema", {}).get("name")
        assert body.get("input") == ["ok"] or schema == "preflight"
        return handle(request)

    _, cached = run_mock(folders, cached_service)
    assert cached["types"] == report["types"]
    assert len(probes) == 3 and all(stage == "probe" for stage, _ in probes)
    assert cached["models"][config.MODEL_FAST]["cost"] == 0.01
    assert all(usage["calls"] - usage["cache_hits"] == 1 for usage in cached["models"].values())


def test_transport_failures_fall_back_and_finish(folders, keys, monkeypatch):
    monkeypatch.setattr(config, "REASONING_EFFORT", "low")
    _, _, out, _ = folders
    stats, report = run_mock(folders, lambda request: httpx.Response(400))
    assert report["models"][config.MODEL_STRONG]["failures"] == 2
    assert all(not check["ok"] and check["status"] == 400 for check in report["preflight"].values())
    assert not any(report["active_stages"][key] for key in ("embeddings", "adjudication", "links", "model_dedupe"))
    assert [row["match_id"] for row in stats["organizations"]["decisions"]] == ["o1", "NEW", "NEW"]
    assert [row["match_id"] for row in stats["people"]["decisions"]] == ["NEW", "p2"]
    for kind, data in stats.items():
        submission = read_rows(out / f"submission_{kind}.csv")
        assert len(submission) == data["rows"]
        assert all(not any(row.get(field) for field in config.LINK_MIN_CONF) for row in submission)


def test_deadline_prevents_requests_and_marks_fallback(folders, keys, monkeypatch):
    monkeypatch.setattr(config, "DEADLINE_SECONDS", 0)
    stats, report = run_mock(folders, lambda request: pytest.fail("Deadline permitted a request"), clock=lambda: 100.0)
    assert report["deadline_hit"]
    assert report["network"] == {}
    assert report["review_queue_size"] == 5
    assert all(row["reason"] == "fallback:deadline" for data in stats.values() for row in data["decisions"])


def test_deadline_cancels_in_flight_request(folders, keys, monkeypatch):
    monkeypatch.setattr(config, "DEADLINE_SECONDS", 10)
    now = 100.0
    calls = []
    canceled = []
    started = asyncio.Event()

    async def handler(request):
        calls.append(request.url.path)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            canceled.append(request.url.path)

    async def sleep(seconds):
        nonlocal now
        await started.wait()
        now = max(now, 100.0 + seconds)

    stats, report = run_mock(folders, handler, clock=lambda: now, sleep=sleep)
    assert calls and Counter(canceled) == Counter(calls)
    assert report["deadline_hit"]
    assert all(row["reason"] == "fallback:deadline" for data in stats.values() for row in data["decisions"])


def test_link_stage_exception_keeps_adjudication(folders, keys, monkeypatch, caplog):
    monkeypatch.setattr(config, "RESEARCH_ENABLED", True)
    real_select = pipeline.select

    async def fail_one(router, record, decision, kind, options):
        if record.input_id == "a":
            raise RuntimeError("synthetic row failure")
        return await real_select(router, record, decision, kind, options)

    monkeypatch.setattr(pipeline, "select", fail_one)
    stats, _ = run_mock(folders, research_service([]))
    first, second, *_ = stats["organizations"]["decisions"]
    assert first["links"] == {} and first["link_confidences"] == {}
    assert first["research_selection"] == {}
    assert first["match_id"] == "NEW" and first["confidence"] == 0.91
    assert first["reason"] == "Synthetic evidence." and first["flags"] == ["link_error"]
    assert second["research_selection"]["domain"]["url"] == "maple-alternative.example"
    assert "Row a failed (RuntimeError)" in caplog.text


def test_adjudication_exception_restores_offline_match(folders, keys, monkeypatch, caplog):
    original = pipeline.adjudicate

    async def fail_one(router, record, candidates, kind, offline):
        if record.input_id == "a":
            raise RuntimeError("synthetic adjudication failure")
        return await original(router, record, candidates, kind, offline)

    monkeypatch.setattr(pipeline, "adjudicate", fail_one)
    stats, _ = run_mock(folders, mock_service([]))
    first, second, *_ = stats["organizations"]["decisions"]
    assert first["match_id"] == "o1" and first["reason"] == "fallback:exact_name+text"
    assert second["links"]["domain"] == "maple.example"
    assert "Row a failed (RuntimeError)" in caplog.text


@pytest.mark.parametrize("pipeline_fallback", [False, True])
def test_only_pipeline_fallback_sets_review_marker(folders, keys, pipeline_fallback):
    handle = mock_service([])

    def handler(request):
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            schema = body["response_format"]["json_schema"]["name"]
            if schema == "entity_match" and json.loads(body["messages"][1]["content"])["name"] == "Birch Foundry":
                if pipeline_fallback:
                    return httpx.Response(400)
                answer = {
                    "decision": "o1",
                    "confidence": 0.95,
                    "reason": "fallback: model text",
                    "employer": "",
                    "role": "",
                }
                return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})
        return handle(request)

    stats, _ = run_mock(folders, handler)
    audit = stats["organizations"]["decisions"][0]
    assert audit["reason"].startswith("fallback:") and audit["confidence"] >= config.REVIEW_BELOW
    assert audit["is_fallback"] == audit["needs_review"] == pipeline_fallback
    row = report_row(folders, "a")
    assert row["needs_review"] == ("fallback" in row["flags"]) == pipeline_fallback
    queue = {entry["input_id"]: entry["flags"].split(";") for entry in read_rows(folders[2] / "review_queue.csv")}
    assert ("fallback" in queue.get("a", [])) == pipeline_fallback


@pytest.mark.parametrize("missing", ["OPENROUTER_API_KEY", "SERPER_API_KEY"])
def test_stages_degrade_independently(folders, keys, monkeypatch, missing):
    monkeypatch.delenv(missing)
    calls = []
    _, report = run_mock(folders, mock_service(calls))
    assert not any(report["links_emitted"].values())
    assert not report["active_stages"]["links"]
    assert not report["active_stages"]["research"]
    assert {stage for stage, _ in calls} == ({"embed", "match", "probe"} if missing == "SERPER_API_KEY" else {"search"})


def test_people_queries_use_and_merge_adjudicated_employer(folders, keys, monkeypatch):
    ambiguous = pipeline.is_ambiguous
    monkeypatch.setattr(
        pipeline, "is_ambiguous", lambda candidates, offline: offline.input_id == "q" or ambiguous(candidates, offline)
    )
    calls = []
    handle = mock_service(calls)

    def handler(request):
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            if body["response_format"]["json_schema"]["name"] == "entity_links":
                prompt = json.loads(body["messages"][1]["content"])
                if prompt["name"] == "Etta Vale":
                    assert prompt["candidates"]["linkedin"]["1"]["link"] == "https://www.linkedin.com/in/etta-vale"
                    assert prompt["candidates"]["x"]["1"]["link"] == "https://x.com/EttaV"
        return handle(request)

    run_mock(folders, handler)
    queries = [query for stage, query in calls if stage == "search" and query.startswith("Etta Vale")]
    suffixes = ("site:linkedin.com/in", "site:crunchbase.com/person", "(site:x.com OR site:twitter.com)")
    assert set(queries) == {
        f"Etta Vale {employer} {suffix}" for employer in ("Twelve Finch Studio", "Cedar Atelier") for suffix in suffixes
    }
    assert len(queries) == 6


def test_types_finish_while_one_type_has_a_slow_model_call(folders, keys):
    people_finished = asyncio.Event()
    calls = []
    handle = mock_service(calls)

    async def handler(request):
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            schema = body["response_format"]["json_schema"]["name"]
            if schema != "preflight":
                name = json.loads(body["messages"][1]["content"])["name"]
                if name == "Birch Foundry" and schema == "entity_match":
                    for _ in range(250):
                        if people_finished.is_set():
                            break
                        await asyncio.sleep(0)
                    assert people_finished.is_set()
                elif name == "Etta Vale" and schema == "entity_links":
                    people_finished.set()
        return handle(request)

    stats, report = run_mock(folders, handler, clock=lambda: 100.0)
    assert all(kind in stats for kind in ("organizations", "people"))
    assert report["links_emitted"]["x"] == 1 and report["links_emitted"]["domain"] == 3
    assert not report["deadline_hit"]


RESEARCH_URLS = {
    "Birch Foundry": ("domain", "https://birch.example/about"),
    "Maple Circuit": ("domain", "https://maple-alternative.example"),
    "Etta Vale": ("linkedin", "https://www.linkedin.com/in/etta-vale"),
}


def research_request(request):
    if request.method != "POST" or request.url.host != "openrouter.ai":
        return None
    body = json.loads(request.content)
    if body.get("response_format", {}).get("json_schema", {}).get("name") == "entity_research":
        return json.loads(body["messages"][1]["content"])["name"]
    return None


def research_service(calls, *, fail=False, urls=None, separate_twins=False):
    existing = mock_service(calls)
    urls = urls or RESEARCH_URLS

    def handler(request):
        if request.url.host == "google.serper.dev":
            body = json.loads(request.content)
            if body["num"] == config.RESEARCH_NUM:
                query = body["q"]
                calls.append(("research_query", query))
                name = query.split('"')[1]
                if name == "Etta Vale":
                    assert query.startswith('"Etta Vale" "Cedar Atelier" ')
                url = urls[name][1]
                if separate_twins and name == "Maple Circuit" and "workshops" in query:
                    url = "https://maple.example"
                return httpx.Response(200, json={"organic": [{"title": name, "link": url}]})
        if name := research_request(request):
            calls.append(("research_response", name))
            if fail:
                return httpx.Response(400)
            field, url = urls[name]
            if separate_twins and name == "Maple Circuit" and "workshops" in request.content.decode():
                url = "https://maple.example"
            answer = {field: {"url": url, "tier": "verified" if name == "Birch Foundry" else "high"}}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})
        return existing(request)

    return handler


def test_research_combines_and_audits_only_new_rows(folders, keys, monkeypatch):
    monkeypatch.setattr(config, "RESEARCH_ENABLED", True)
    monkeypatch.setattr(config, "DISAGREE_POLICY", "blank")
    _, _, out, _ = folders
    calls = []
    stats, report = run_mock(folders, research_service(calls))
    assert report["active_stages"]["research"]
    assert (report["research_calls"], report["research_queries"], report["research_rejected"]) == (3, 12, 0)
    assert report["link_agreement"] == {
        "domain": {"both": 1, "search": 0, "research": 0, "disagree": 2},
        "linkedin": {"both": 0, "search": 3, "research": 1, "disagree": 0},
        "crunchbase": {"both": 0, "search": 0, "research": 0, "disagree": 0},
        "x": {"both": 0, "search": 1, "research": 0, "disagree": 0},
    }
    assert Counter(name for stage, name in calls if stage == "research_response") == dict.fromkeys(RESEARCH_URLS, 1)
    first, second, duplicate = stats["organizations"]["decisions"]
    assert first["agreement"] == {"domain": "both", "linkedin": "search"}
    assert first["research_selection"]["domain"] == {"url": "birch.example", "tier": "verified", "confidence": 0.9}
    assert first["link_confidences"]["domain"] == 1.0
    assert second["flags"] == duplicate["flags"] == ["link_disagree:domain"]
    assert duplicate["dup_of"] == "b"
    assert all(
        "domain" not in row["links"] and row["search_links"]["domain"] == "maple.example" for row in (second, duplicate)
    )
    matched, new = stats["people"]["decisions"]
    assert matched["match_id"] == "p1" and matched["research_selection"] == matched["agreement"] == {}
    assert new["agreement"] == {"linkedin": "research", "x": "search"}
    assert new["links"]["linkedin"] == "https://www.linkedin.com/in/etta-vale"
    assert new["link_confidences"]["linkedin"] == 0.75
    queue = read_rows(out / "review_queue.csv")
    assert {row["input_id"] for row in queue if row["flags"] == "link_disagree:domain"} == {"b", "c"}


def test_research_call_failure_preserves_search(folders, keys, monkeypatch):
    monkeypatch.setattr(config, "RESEARCH_ENABLED", True)
    stats, report = run_mock(folders, research_service([], fail=True))
    assert (report["research_calls"], report["research_queries"], report["research_rejected"]) == (3, 12, 0)
    first = stats["organizations"]["decisions"][0]
    assert first["links"]["domain"] == "birch.example" and first["link_confidences"]["domain"] == 0.64
    assert first["agreement"]["domain"] == "search" and first["research_selection"] == {}


@pytest.mark.parametrize(
    ("host", "confirm", "expected", "rejected"),
    [
        ("gazette.example", False, "", 1),
        ("gazette.example", True, "gazette.example", 0),
        ("birch-copper.example", False, "birch-copper.example", 0),
        ("birc.example", False, "birc.example", 0),
    ],
)
def test_research_domain_requires_name_evidence_or_confirmation(
    folders, keys, monkeypatch, host, confirm, expected, rejected
):
    monkeypatch.setattr(config, "RESEARCH_ENABLED", True)
    original = pipeline.select

    async def select(router, record, decision, kind, options):
        selections = await original(router, record, decision, kind, options)
        if record.name == "Birch Foundry":
            selections["domain"] = Selection([Hit("", host, "", "organic")], 1, 0.64) if confirm else Selection([])
        return selections

    monkeypatch.setattr(pipeline, "select", select)
    urls = {**RESEARCH_URLS, "Birch Foundry": ("domain", f"https://{host}/article")}
    stats, report = run_mock(folders, research_service([], urls=urls))
    row = stats["organizations"]["decisions"][0]
    assert row["links"].get("domain", "") == expected
    assert row["agreement"].get("domain") == ("both" if confirm else "research" if expected else None)
    assert report["research_rejected"] == rejected


def distinct_twins(folders):
    (folders[1] / "input_organizations.csv").write_text(
        "input_id,name,description\nb,Maple Circuit,Builds portable radios\nc,Maple Circuit,Builds portable radios for workshops\n",
        encoding="utf-8",
    )


def test_duplicate_group_propagates_links_after_one_twin_disagrees(folders, keys, monkeypatch):
    monkeypatch.setattr(config, "RESEARCH_ENABLED", True)
    monkeypatch.setattr(config, "DISAGREE_POLICY", "blank")
    distinct_twins(folders)
    stats, _ = run_mock(folders, research_service([], separate_twins=True))
    first, twin = stats["organizations"]["decisions"]
    assert first["agreement"]["domain"] == "disagree"
    assert twin["agreement"]["domain"] == "both" and twin["dup_of"] == "b"
    assert first["links"] == twin["links"] and first["links"]["domain"] == "maple.example"
    assert first["link_confidences"] == twin["link_confidences"]


@pytest.mark.parametrize("failure", ["deadline", "exception"])
def test_skipped_or_failed_twin_receives_group_links(folders, keys, monkeypatch, failure):
    distinct_twins(folders)
    monkeypatch.setattr(config, "DEADLINE_SECONDS", 10)
    now = 100.0
    selected, blocked = asyncio.Event(), asyncio.Event()
    handle, original = mock_service([]), pipeline.select

    async def select(router, record, decision, kind, options):
        if record.input_id == "c":
            await selected.wait()
            if failure == "exception":
                raise RuntimeError("synthetic failure")
            blocked.set()
            await asyncio.sleep(0)
            return {}
        result = await original(router, record, decision, kind, options)
        if record.input_id == "b":
            selected.set()
            await asyncio.sleep(0)
        return result

    def clock():
        nonlocal now
        if blocked.is_set():
            now = 110.0
        return now

    monkeypatch.setattr(pipeline, "select", select)
    stats, report = run_mock(folders, handle, clock=clock)
    first, twin = stats["organizations"]["decisions"]
    assert twin["dup_of"] == "b" and first["links"] == twin["links"]
    assert twin["links"]["domain"] == "maple.example"
    assert report["deadline_hit"] == (failure == "deadline")
    assert ("link_error" in twin.get("flags", [])) == (failure == "exception")
    assert report_row(folders, "c")["links"]["domain"]["source"] == "from a duplicate row"


def test_deadline_during_research_preserves_completed_search(folders, keys, monkeypatch):
    monkeypatch.setattr(config, "RESEARCH_ENABLED", True)
    monkeypatch.setattr(config, "DEADLINE_SECONDS", 10)
    now = 100.0
    selected, researching, canceled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    handle = research_service([])
    select = pipeline.select

    async def signal_selection(router, record, decision, kind, options):
        result = await select(router, record, decision, kind, options)
        if record.name == "Etta Vale":
            selected.set()
        return result

    async def handler(request):
        if research_request(request) == "Etta Vale":
            researching.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()
        return handle(request)

    async def sleep(seconds):
        nonlocal now
        await researching.wait()
        await selected.wait()
        now = max(now, 100.0 + seconds)

    monkeypatch.setattr(pipeline, "select", signal_selection)
    stats, report = run_mock(folders, handler, clock=lambda: now, sleep=sleep)
    assert report["deadline_hit"] and canceled.is_set()
    row = stats["people"]["decisions"][1]
    assert row["links"] == {"x": "https://x.com/EttaV"}
    assert row["agreement"] == {"x": "search"} and row["research_selection"] == {}
