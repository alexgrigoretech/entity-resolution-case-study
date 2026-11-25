import asyncio
import json
import sqlite3
from contextlib import closing

import httpx
import pytest

from resolver import config
from resolver.httpcache import Cache, CachedClient, Ledger

URL = "https://windlass.example/request"


def test_json_cache_uses_canonical_body_and_skips_headers(tmp_path):
    path = tmp_path / "cache.sqlite"
    calls = []
    fake_key = "synthetic-placeholder-token"
    payload = {"answer": "ok", "usage": {"prompt_tokens": 6, "completion_tokens": 2, "cost": 0.01}}

    def handler(request):
        calls.append(request.headers["Authorization"])
        return httpx.Response(200, json=payload)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            first = CachedClient(Cache(path), client)
            assert not path.exists()
            assert await first.post_json("model", URL, {"b": 2, "a": 1}, {"Authorization": fake_key}) == payload
            second = CachedClient(Cache(path), client, first.ledger)
            assert await second.post_json("model", URL, {"a": 1, "b": 2}, {"Authorization": "other"}) == payload
            assert second.ledger.to_dict()["model"] == {
                "calls": 2,
                "cache_hits": 1,
                "failures": 0,
                "retries": 0,
                "prompt_tokens": 6,
                "completion_tokens": 2,
                "cost": 0.01,
                "truncated": 0,
            }

    asyncio.run(run())
    assert calls == [fake_key]
    raw = path.read_bytes()
    assert fake_key.encode() not in raw
    assert b"authorization" not in raw.lower()


def test_retry_after_then_cache_text(tmp_path):
    delays = []
    calls = []

    def handler(request):
        calls.append(request.method)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2.5"})
        return httpx.Response(200, text="a test page")

    async def sleep(delay):
        delays.append(delay)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cached = CachedClient(Cache(tmp_path / "cache.sqlite"), client, sleep=sleep)
            assert await cached.get_text("site", URL) == "a test page"
            assert await cached.get_text("site", URL) == "a test page"
            assert cached.ledger.to_dict()["site"]["retries"] == 1

    asyncio.run(run())
    assert calls == ["GET", "GET"]
    assert delays == [2.5]


def test_text_limit_is_applied_before_caching(tmp_path):
    path = tmp_path / "cache.sqlite"
    calls = []

    def handler(request):
        calls.append(request.method)
        return httpx.Response(200, text="éclair recipe")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cached = CachedClient(Cache(path), client)
            assert await cached.get_text("site", URL, max_chars=3) == "écl"
            with closing(sqlite3.connect(path)) as connection:
                assert [json.loads(row[0]) for row in connection.execute("SELECT response FROM responses")] == ["écl"]
            assert await cached.get_text("site", URL, max_chars=3) == "écl"
            assert await cached.get_text("site", URL) == "éclair recipe"
            assert await cached.get_text("site", URL, max_chars=6) == "éclair"
            assert await cached.get_text("site", URL, max_chars=3) == "écl"

    asyncio.run(run())
    assert calls == ["GET", "GET", "GET"]


@pytest.mark.parametrize(("status", "attempts"), [(500, config.MAX_ATTEMPTS), (400, 1), (None, config.MAX_ATTEMPTS)])
def test_gives_up_on_failure_and_does_not_cache(tmp_path, status, attempts):
    delays = []
    calls = []

    def handler(request):
        calls.append(request.method)
        if status is None:
            raise httpx.ReadTimeout("fixture timeout", request=request)
        return httpx.Response(status)

    async def sleep(delay):
        delays.append(delay)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cached = CachedClient(Cache(tmp_path / "cache.sqlite"), client, sleep=sleep)
            for _ in range(2):
                assert await cached.post_json("model", URL, {}) is None
            usage = cached.ledger.to_dict()["model"]
            assert usage["failures"] == 2
            assert usage["cache_hits"] == 0
            assert usage["retries"] == 2 * (attempts - 1)

    asyncio.run(run())
    assert len(calls) == 2 * attempts
    assert len(delays) == 2 * (attempts - 1)
    bounds = [min(config.BACKOFF_CAP, config.BACKOFF_BASE * 2**i) for i in range(attempts - 1)]
    assert all(0 <= delay <= bound for delay, bound in zip(delays, bounds * 2, strict=True))


@pytest.mark.parametrize(
    ("cost_fields", "expected"),
    [
        (
            {
                "cost": 0,
                "is_byok": True,
                "cost_details": {
                    "upstream_inference_cost": 0.002918,
                    "upstream_inference_prompt_cost": 0.002658,
                    "upstream_inference_completions_cost": 0.00026,
                },
            },
            0.002918,
        ),
        ({"cost": 0.01, "cost_details": {"upstream_inference_cost": 0.02}}, 0.01),
        ({"cost_details": {"upstream_inference_cost": 0.02}}, 0.02),
        ({"cost": True, "cost_details": {"upstream_inference_cost": 0.02}}, 0.02),
        ({"cost": 0}, 0.0),
        ({"cost_details": None}, 0.0),
        ({"cost_details": "unavailable"}, 0.0),
        ({"cost_details": {}}, 0.0),
        ({"cost_details": {"upstream_inference_cost": None}}, 0.0),
        ({"cost_details": {"upstream_inference_cost": "0.02"}}, 0.0),
        ({"cost_details": {"upstream_inference_cost": True}}, 0.0),
    ],
)
def test_namespace_ledger_counts_reported_usage_cost(cost_fields, expected):
    ledger = Ledger()
    ledger.namespace("model").add_response({"usage": {"prompt_tokens": 2127, "completion_tokens": 52, **cost_fields}})
    usage = ledger.to_dict()["model"]
    assert usage["cost"] == expected
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (2127, 52)
