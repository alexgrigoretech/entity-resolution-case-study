"""Call OpenRouter for structured decisions and ordered embeddings."""

import json
import logging
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict
from typing import Any, Never

from . import config, embeddings
from .httpcache import CachedClient, Ledger, NamespaceUsage, Payload, reported_cost

LOGGER = logging.getLogger(__name__)
API_URL = "https://openrouter.ai/api/v1"


class ModelUsage(NamespaceUsage):
    def __init__(self, model: str) -> None:
        super().__init__()
        self.prices = config.MODEL_PRICES.get(model, (0, 0))

    def add_response(self, response: Payload) -> None:
        before = self.prompt_tokens, self.completion_tokens
        super().add_response(response)
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict) and reported_cost(usage) is None:
            self.cost += (
                (self.prompt_tokens - before[0]) * self.prices[0]
                + (self.completion_tokens - before[1]) * self.prices[1]
            ) / 1_000_000


class ModelLedger(Ledger):
    def __init__(self) -> None:
        super().__init__()
        self.models: dict[tuple[str, str], ModelUsage] = {}
        self.model: ContextVar[str | None] = ContextVar("resolver_model", default=None)

    def namespace(self, name: str) -> NamespaceUsage:
        model = self.model.get()
        if model is None or name not in ("llm", "embed"):
            return super().namespace(name)
        key = name, model
        if key not in self.models:
            self.models[key] = ModelUsage(model)
        return self.models[key]

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        return self._totals(by_model=False)

    def by_model(self) -> dict[str, dict[str, int | float]]:
        return self._totals(by_model=True)

    def _totals(self, by_model: bool) -> dict[str, dict[str, int | float]]:
        totals = {} if by_model else super().to_dict()
        for (namespace, model), usage in self.models.items():
            total = totals.setdefault(model if by_model else namespace, asdict(NamespaceUsage()))
            for name, value in asdict(usage).items():
                total[name] += value
        return totals


def confidence_of(value: object) -> float | None:
    if type(value) in (int, float) and 0 <= value <= 1:
        return float(value)
    return None


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"Invalid JSON constant {value}")


def _chat_content(body: dict[str, Any]) -> dict[str, Any] | None:
    if "error" in body:
        return None
    try:
        raw = body["choices"][0]["message"]["content"]
        if isinstance(raw, str):
            content = json.loads(raw, parse_constant=_reject_json_constant)
            if isinstance(content, dict):
                return content
    except (KeyError, IndexError, TypeError, ValueError, RecursionError):
        pass
    return None


def _truncated(payload: Payload) -> bool:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    return bool(
        isinstance(choices, list)
        and choices
        and isinstance(choices[0], dict)
        and choices[0].get("finish_reason") == "length"
    )


class OpenRouter:
    def __init__(self, client: CachedClient, api_key: str | None) -> None:
        self.client = client
        self.api_key = api_key
        self.chat_routes: dict[str, str] = {}
        self.chat_enabled = self.embed_enabled = bool(api_key)
        self.without_reasoning: set[str] = set()

    def chat_model(self, model: str) -> str:
        return self.chat_routes.get(model, model)

    async def _post(
        self,
        namespace: str,
        endpoint: str,
        body: dict[str, Any],
        usable: Callable[[dict], bool],
        *,
        probe: bool = False,
        cacheable: Callable[[Payload], bool] | None = None,
    ) -> dict[str, Any] | None:
        ledger = self.client.ledger
        token = ledger.model.set(body["model"]) if isinstance(ledger, ModelLedger) else None
        try:
            return await self.client.post_json(
                namespace,
                f"{API_URL}/{endpoint}",
                body,
                {"Authorization": f"Bearer {self.api_key}"},
                usable,
                probe=probe,
                cacheable=cacheable,
            )
        finally:
            if token is not None:
                ledger.model.reset(token)

    async def _chat(
        self,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
        *,
        probe: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": config.MAX_TOKENS if max_tokens is None else max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            "usage": {"include": True},
        }
        if config.REASONING_EFFORT and model not in self.without_reasoning:
            body["reasoning"] = {"effort": config.REASONING_EFFORT}

        def usable(payload: dict[str, Any]) -> bool:
            if _truncated(payload):
                usage = self.client.ledger.namespace("llm")
                usage.truncated += 1
                if not probe:
                    usage.failures += 1
                LOGGER.warning("OpenRouter model %s output was cut", model)
                return not probe
            return _chat_content(payload) is not None

        response = await self._post(
            "llm", "chat/completions", body, usable, probe=probe, cacheable=lambda payload: not _truncated(payload)
        )
        if response is None:
            LOGGER.warning("OpenRouter chat returned no JSON object")
            return None
        return None if _truncated(response) else _chat_content(response)

    async def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        if not self.chat_enabled:
            LOGGER.warning("OpenRouter chat is disabled")
            return None
        return await self._chat(self.chat_model(model), system, user, schema_name, schema, max_tokens=max_tokens)

    async def probe_chat(self, model: str) -> None:
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        await self._chat(model, "Reply with JSON.", "ok", "preflight", schema, probe=True)

    async def probe_embed(self, model: str) -> None:
        await embeddings.probe_embed(self, model)

    async def embed(self, model: str, texts: list[str]) -> list[list[float]] | None:
        return await embeddings.embed(self, model, texts)
