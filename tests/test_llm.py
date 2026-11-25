import asyncio
import json

import httpx
import pytest

from resolver import config
from resolver.httpcache import Cache, CachedClient
from resolver.llm import ModelLedger, ModelUsage, OpenRouter

KEY = "synthetic-router-placeholder"


@pytest.mark.parametrize(
    ("content", "key", "expected"),
    [
        (' {"ok": true}', KEY, {"ok": True}),
        ("{invalid", KEY, None),
        ('{"score": NaN}', KEY, None),
        ("[]", KEY, None),
        ("{}", None, None),
    ],
)
def test_chat_json(tmp_path, caplog, content, key, expected):
    requests = []
    schema = {"type": "object"}

    def handler(request):
        requests.append(request)
        assert request.url == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        assert json.loads(request.content) == {
            "model": config.MODEL_FAST,
            "messages": [{"role": "system", "content": "instruction"}, {"role": "user", "content": "evidence"}],
            "temperature": 0,
            "max_tokens": 800,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "test", "strict": True, "schema": schema},
            },
            "usage": {"include": True},
            **({"reasoning": {"effort": config.REASONING_EFFORT}} if config.REASONING_EFFORT else {}),
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def sleep(_delay):
        pass

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = OpenRouter(CachedClient(Cache(tmp_path / "requests.sqlite"), http, sleep=sleep), key)
            assert await client.chat_json(config.MODEL_FAST, "instruction", "evidence", "test", schema) == expected

    asyncio.run(run())
    assert len(requests) == (0 if not key else config.MAX_ATTEMPTS if expected is None else 1)
    if expected is None:
        assert "OpenRouter chat" in caplog.text
    assert KEY not in caplog.text


def test_embed_batch_order_and_model_usage(tmp_path):
    batches = []
    texts = [str(i) for i in range(130)]

    def handler(request):
        assert request.url == "https://openrouter.ai/api/v1/embeddings"
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        body = json.loads(request.content)
        assert body["model"] == config.MODEL_EMBED
        batch = body["input"]
        batches.append(batch)
        data = [{"index": index, "embedding": [float(value), 1.0]} for index, value in enumerate(batch)]
        usage = {"prompt_tokens": 100}
        if len(batches) == 3:
            usage["cost"] = 0.05
        return httpx.Response(200, json={"data": data[::-1], "usage": usage})

    async def run():
        ledger = ModelLedger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = OpenRouter(CachedClient(Cache(tmp_path / "requests.sqlite"), http, ledger), KEY)
            for _ in range(2):
                assert await client.embed(config.MODEL_EMBED, texts) == [[float(value), 1.0] for value in texts]
        totals = ledger.by_model()[config.MODEL_EMBED]
        assert totals["calls"] == 6
        assert totals["cache_hits"] == 3
        assert totals["prompt_tokens"] == 300
        assert totals["cost"] == pytest.approx(0.05 + 200 * 0.02 / 1_000_000)

    asyncio.run(run())
    assert batches == [texts[:64], texts[64:128], texts[128:]]


@pytest.mark.parametrize("truncated", ['{"fixture":', '{"fixture":false}'])
def test_chat_logs_and_counts_truncated_response_by_model(tmp_path, caplog, truncated):
    replies = iter([("length", truncated), ("stop", '{"fixture":true}')])
    requests, delays = [], []

    def handler(request):
        finish, content = next(replies)
        requests.append(request.content)
        assert json.loads(request.content)["max_tokens"] == 1600
        return httpx.Response(200, json={"choices": [{"finish_reason": finish, "message": {"content": content}}]})

    async def sleep(delay):
        delays.append(delay)

    async def run():
        ledger = ModelLedger()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            router = OpenRouter(CachedClient(Cache(tmp_path / "cache.sqlite"), http, ledger, sleep=sleep), KEY)
            arguments = ("fixture/research", "instruction", "evidence", "fixture", {})
            assert await router.chat_json(*arguments, max_tokens=1600) is None
            assert len(requests) == 1
            assert await router.chat_json(*arguments, max_tokens=1600) == {"fixture": True}
            assert await router.chat_json(*arguments, max_tokens=1600) == {"fixture": True}
        assert ledger.by_model()["fixture/research"]["truncated"] == 1
        assert ledger.to_dict()["llm"]["truncated"] == 1

    asyncio.run(run())
    assert len(requests) == 2 and requests[0] == requests[1] and not delays
    assert caplog.messages == ["OpenRouter model fixture/research output was cut"]


@pytest.mark.parametrize(
    ("model", "tokens"),
    [
        (config.MODEL_FAST, {"prompt_tokens": 2127, "completion_tokens": 52}),
        (config.MODEL_STRONG, {"prompt_tokens": 2127, "completion_tokens": 52}),
        (config.MODEL_EMBED, {"prompt_tokens": 2127}),
    ],
)
def test_model_usage_counts_byok_cost_without_price_estimate(model, tokens):
    usage = ModelUsage(model)
    byok = {"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.002918}}
    usage.add_response({"usage": {**tokens, **byok}})
    assert usage.cost == 0.002918


@pytest.mark.parametrize(("cost_fields", "expected"), [({"cost": 0}, 0.0), ({"cost": False}, 0.001), ({}, 0.001)])
def test_model_usage_estimates_only_absent_cost(monkeypatch, cost_fields, expected):
    monkeypatch.setitem(config.MODEL_PRICES, "fixture/model", (2, 5))
    usage = ModelUsage("fixture/model")
    usage.add_response({"usage": {"prompt_tokens": 250, "completion_tokens": 100, **cost_fields}})
    assert usage.cost == pytest.approx(expected)
