import asyncio
import json
import sqlite3
from contextlib import closing

import httpx
import pytest

from resolver import config
from resolver.httpcache import Cache, CachedClient, Ledger, RunClient
from resolver.llm import OpenRouter
from resolver.preflight import preflight
from resolver.records import read_database, read_inputs
from resolver.runner import run
from resolver.search import Serper, homepage_links
from solve import _print_preflight


@pytest.mark.parametrize(
    ("status", "attempts", "reasoning_disabled", "ok"),
    [
        (429, [True, True], False, True),
        (500, [True, True], False, True),
        (None, [True, True], False, True),
        (400, [True, False], True, True),
        (422, [True, False], True, True),
        (401, [True], False, False),
        (403, [True], False, False),
        (503, [True, True, True], False, False),
    ],
)
def test_preflight_retries_transients_and_uses_reasoning_fallback_only_for_bad_requests(
    tmp_path, monkeypatch, status, attempts, reasoning_disabled, ok
):
    model = "fixture/chat"
    monkeypatch.setattr(config, "MODEL_FAST", model)
    monkeypatch.setattr(config, "MODEL_STRONG", model)
    monkeypatch.setattr(config, "REASONING_EFFORT", "low")
    received, delays = [], []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}]})
        received.append("reasoning" in body)
        if status == 503 or len(received) == 1:
            if status is None:
                raise httpx.ReadTimeout("mock timeout", request=request)
            return httpx.Response(status, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}}]})

    async def sleep(delay):
        delays.append(delay)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            router = OpenRouter(CachedClient(Cache(tmp_path / "cache.sqlite"), http, sleep=sleep), "placeholder")
            result = (await preflight(router))[model]
            assert result["ok"] == router.chat_enabled == ok
            assert result.get("reasoning_disabled", False) == (model in router.without_reasoning) == reasoning_disabled
            if not ok:
                assert result["status"] == status

    asyncio.run(run())
    assert received == attempts
    if status in (429, 500, 503):
        assert delays == [2.0] * (len(attempts) - 1)
    else:
        assert len(delays) == (status is None)


@pytest.mark.parametrize("exhaust", [False, True])
def test_preflight_retries_unusable_200_before_disabling_reasoning(tmp_path, monkeypatch, exhaust):
    model = "fixture/chat"
    monkeypatch.setattr(config, "MODEL_FAST", model)
    monkeypatch.setattr(config, "MODEL_STRONG", model)
    monkeypatch.setattr(config, "REASONING_EFFORT", "low")
    received, delays = [], []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}]})
        received.append("reasoning" in body)
        if len(received) == 1:
            return httpx.Response(200, json={"choices": []})
        if exhaust and "reasoning" in body:
            return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": "{"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}}]})

    async def sleep(delay):
        delays.append(delay)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            router = OpenRouter(CachedClient(Cache(tmp_path / "cache.sqlite"), http, sleep=sleep), "placeholder")
            check = (await preflight(router))[model]
            assert check == ({"ok": True, "reasoning_disabled": True} if exhaust else {"ok": True})
            assert router.chat_enabled and (model in router.without_reasoning) == exhaust

    asyncio.run(run())
    assert received == ([True] * config.PROBE_ATTEMPTS + [False] if exhaust else [True, True])
    assert len(delays) == (config.PROBE_ATTEMPTS - 1 if exhaust else 1)


@pytest.mark.parametrize(
    ("namespace", "remaining", "read", "connect", "delay"),
    [
        ("llm", 100, 90, 10, 7),
        ("embed", 100, 60, 10, 0),
        ("serper", 100, 20, 10, 0),
        ("web", 100, 10, 10, 0),
        ("llm", 3, 3, 3, 0),
    ],
)
def test_request_timeout_policy(tmp_path, namespace, remaining, read, connect, delay):
    now = 100.0

    async def handler(request):
        nonlocal now
        timeout = request.extensions["timeout"]
        assert timeout == {"read": read, "write": read, "pool": read, "connect": connect}
        if delay > timeout["read"]:
            raise httpx.ReadTimeout("Synthetic response exceeded its read timeout")
        now += delay
        return httpx.Response(200, html="<p>ok</p>") if namespace == "web" else httpx.Response(200, json={"ok": True})

    async def run():
        client = RunClient(
            Cache(tmp_path / "cache.sqlite"), Ledger(), now + remaining, httpx.MockTransport(handler), clock=lambda: now
        )
        try:
            url = "https://birch.example"
            if namespace == "web":
                assert await client.get_text(namespace, url) == "<p>ok</p>"
            else:
                assert await client.post_json(namespace, url, {}) == {"ok": True}
        finally:
            await client.aclose()

    asyncio.run(run())
    assert now == 100.0 + delay


@pytest.mark.parametrize(
    ("namespace", "invalid", "valid"),
    [
        (
            "llm",
            {"error": {"message": "Unavailable"}, "choices": [{"message": {"content": "{}"}}]},
            {"choices": [{"message": {"content": "{}"}}]},
        ),
        ("llm", {"choices": [{"message": {"content": "plain text"}}]}, {"choices": [{"message": {"content": "{}"}}]}),
        ("embed", {"data": []}, {"data": [{"index": 0, "embedding": [1.0]}]}),
        ("serper", {"message": "Unavailable"}, {"organic": []}),
        ("serper", {"error": "Unavailable"}, {"organic": []}),
    ],
)
def test_unusable_success_is_retried_and_only_usable_body_is_cached(tmp_path, namespace, invalid, valid):
    replies = iter((invalid, valid))
    path = tmp_path / "cache.sqlite"

    def handler(_request):
        return httpx.Response(200, json=next(replies))

    async def sleep(_seconds):
        pass

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = CachedClient(Cache(path), http, sleep=sleep)
            router = OpenRouter(client, "router-placeholder")
            for _ in range(2):
                if namespace == "llm":
                    assert await router.chat_json(config.MODEL_FAST, "Reply with JSON.", "ok", "object", {}) == {}
                elif namespace == "embed":
                    assert await router.embed(config.MODEL_EMBED, ["ok"]) == [[1.0]]
                else:
                    assert await Serper(client, "serper-placeholder").search("ok") == []
            usage = client.ledger.to_dict()[namespace]
            assert (usage["retries"], usage["cache_hits"], usage["failures"]) == (1, 1, 0)

    asyncio.run(run())
    with closing(sqlite3.connect(path)) as connection:
        assert [json.loads(row[0]) for row in connection.execute("SELECT response FROM responses")] == [valid]


def test_web_redirects_www_fallback_and_html_only(tmp_path):
    responses = {
        "https://birch.example": httpx.Response(301, headers={"Location": "/one"}),
        "https://birch.example/one": httpx.Response(302, headers={"Location": "/two"}),
        "https://birch.example/two": httpx.Response(307, headers={"Location": "https://new-birch.example"}),
        "https://new-birch.example": httpx.Response(308, headers={"Location": "/four"}),
        "https://www.birch.example": httpx.Response(
            302, headers={"Location": "https://media.linkedin.com/company/birch"}
        ),
        "https://cedar.example": httpx.Response(200, text='<a href="https://x.com/unwanted">Feed</a>'),
        "https://www.cedar.example": httpx.Response(302, headers={"Location": "/home"}),
        "https://www.cedar.example/home": httpx.Response(200, html='<a href="https://x.com/cedar">Feed</a>'),
    }
    visits = []

    def handler(request):
        assert request.headers["User-Agent"] == "Mozilla/5.0 (compatible; entity-resolver/0.1; research)"
        assert request.headers["Accept"] == "text/html"
        url = str(request.url)
        visits.append(url)
        return responses[url]

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = CachedClient(Cache(tmp_path / "cache.sqlite"), http)
            assert await homepage_links(client, "birch.example") == {}
            assert await homepage_links(client, "cedar.example") == {"x": [("https://x.com/cedar", "homepage")]}

    asyncio.run(run())
    assert visits == list(responses)


def test_api_redirects_stay_off(tmp_path):
    def handler(request):
        assert str(request.url) == "https://api.example/search"
        return httpx.Response(302, headers={"Location": "/elsewhere"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as http:
            client = CachedClient(Cache(tmp_path / "cache.sqlite"), http)
            assert await client.post_json("serper", "https://api.example/search", {}) is None

    asyncio.run(run())


def test_csv_handles_extra_missing_and_blank_fields(tmp_path, caplog):
    db = tmp_path / "db.csv"
    db.write_text("id,name,description\n db1 , Birch , Tools ,ignored\n ,skip,bad\n db2 , Cedar \n", encoding="utf-8")
    records = tmp_path / "input.csv"
    records.write_text("input_id,name,source_article\n ,Skip,n/a\np1,Cora\n", encoding="utf-8")
    assert [(row.id, row.name, row.description) for row in read_database(db, "organizations")] == [
        ("db1", "Birch", "Tools"),
        ("db2", "Cedar", ""),
    ]
    assert [(row.input_id, row.name, row.text) for row in read_inputs(records, "people")] == [("p1", "Cora", "")]
    assert [(str(db) in line, str(records) in line, line.rsplit(" ", 1)[-1]) for line in caplog.messages] == [
        (True, False, "3"),
        (False, True, "2"),
    ]


@pytest.mark.parametrize("problem", ["fast", "strong", "both", "embed", "reasoning", "effort_disabled", "offline"])
def test_preflight_controls_models_and_records_failures(tmp_path, monkeypatch, capsys, problem):
    fast, strong, embed = config.MODEL_FAST, config.MODEL_STRONG, config.MODEL_EMBED
    failed_models = {
        "fast": {fast},
        "strong": {strong},
        "both": {fast, strong},
        "embed": {embed},
    }.get(problem, set())
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-placeholder")
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.setattr(config, "REASONING_EFFORT", "" if problem == "effort_disabled" else "low")
    monkeypatch.setattr(config, "QUOTE_QUERIES", True)
    (tmp_path / "existing_organizations.csv").write_text(
        "id,name,description\ndb1,Birch,Builds tools\n", encoding="utf-8"
    )
    (tmp_path / "input_organizations.csv").write_text(
        "input_id,name,description\nin1,Birch,Builds tools\n", encoding="utf-8"
    )
    probes = []
    live = []

    def handler(request):
        body = json.loads(request.content)
        model = body["model"]
        schema = body.get("response_format", {}).get("json_schema", {}).get("name")
        probe = schema == "preflight" or body.get("input") == ["ok"]
        (probes if probe else live).append((model, body.get("reasoning")))
        if probe and model in failed_models:
            return httpx.Response(400, json={"error": {"message": "Unknown model router-placeholder"}})
        if problem == "reasoning" and model == fast and "reasoning" in body:
            assert probe
            return httpx.Response(400, json={"error": {"message": "Unsupported reasoning"}})
        if model == embed:
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
        assert body["max_tokens"] == 800
        answer = (
            {"ok": True}
            if probe
            else {"decision": "NEW", "confidence": 0.71, "reason": "Synthetic evidence.", "employer": "", "role": ""}
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

    out = tmp_path / "out"
    stats = run(
        tmp_path,
        tmp_path,
        out,
        cache_path=tmp_path / "cache.sqlite",
        transport=httpx.MockTransport(handler),
        network=problem != "offline",
        on_preflight=_print_preflight,
    )
    report = json.loads((out / "run_stats.json").read_text(encoding="utf-8"))
    assert report["active_stages"]["quoted_queries"] is True
    if problem == "offline":
        assert report["preflight"] == {} and not probes and not live
        return
    assert {model for model, _ in probes} == {fast, strong, embed}
    assert {model for model, result in report["preflight"].items() if not result["ok"]} == failed_models
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(failed_models)
    for model in failed_models:
        assert report["preflight"][model] == {"ok": False, "status": 400, "error": "Unknown model [redacted]"}
        assert f"Model preflight failed for {model} (HTTP 400) Unknown model [redacted]" in lines
    assert {model for model, _ in live} == {fast, strong, embed} - failed_models
    assert report["active_stages"]["embeddings"] == (embed not in failed_models)
    assert report["active_stages"]["adjudication"] == report["active_stages"]["model_dedupe"] == (problem != "both")
    assert stats["organizations"]["decisions"][0]["model"] == (
        "" if problem == "both" else fast if problem == "strong" else strong
    )
    if problem == "reasoning":
        assert report["preflight"][fast] == {"ok": True, "reasoning_disabled": True}
        assert [reasoning for model, reasoning in probes if model == fast] == [{"effort": "low"}, None]
        assert all(reasoning is None for model, reasoning in live if model == fast)
        assert (strong, {"effort": "low"}) in live
    if problem == "effort_disabled":
        assert all(reasoning is None for _, reasoning in probes + live)
