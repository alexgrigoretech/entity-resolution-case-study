"""Check model availability and configure the run's model routes."""

import asyncio
import os
from typing import NotRequired, TypedDict

from . import config
from .llm import OpenRouter
from .runclient import RequestFailure


class PreflightResult(TypedDict):
    ok: bool
    status: NotRequired[int | None]
    error: NotRequired[str]
    reasoning_disabled: NotRequired[bool]


def network_keys(network: bool) -> tuple[str | None, str | None]:
    return (os.getenv("OPENROUTER_API_KEY"), os.getenv("SERPER_API_KEY")) if network else (None, None)


def active_stages(network: bool, router: OpenRouter | None = None) -> dict[str, bool]:
    router_key, serper_key = network_keys(network)
    chat = router.chat_enabled if router is not None else bool(router_key)
    return {
        "embeddings": router.embed_enabled if router is not None else bool(router_key),
        "adjudication": chat,
        "search": bool(serper_key),
        "links": chat and bool(serper_key),
        "model_dedupe": chat,
        "quoted_queries": config.QUOTE_QUERIES,
    }


async def preflight(router: OpenRouter) -> dict[str, PreflightResult]:
    if not router.api_key:
        return {}

    def failed(error: RequestFailure) -> PreflightResult:
        message = " ".join(str(error).replace(router.api_key or "", "[redacted]").split())
        return {"ok": False, "status": error.status, "error": message[:300]}

    async def chat(model: str) -> PreflightResult:
        try:
            await router.probe_chat(model)
        except RequestFailure as error:
            if error.status not in (200, 400, 422) or not config.REASONING_EFFORT or router.client.expired():
                return failed(error)
            router.without_reasoning.add(model)
            try:
                await router.probe_chat(model)
            except RequestFailure as retry_error:
                router.without_reasoning.discard(model)
                return failed(retry_error)
            return {"ok": True, "reasoning_disabled": True}
        return {"ok": True}

    async def embed() -> PreflightResult:
        try:
            await router.probe_embed(config.MODEL_EMBED)
        except RequestFailure as error:
            return failed(error)
        return {"ok": True}

    chat_models = list(dict.fromkeys((config.MODEL_FAST, config.MODEL_STRONG)))
    *checks, embedding = await asyncio.gather(*(chat(model) for model in chat_models), embed())
    results = dict(zip(chat_models, checks, strict=True))
    available = [model for model in chat_models if results[model]["ok"]]
    router.chat_enabled = bool(available)
    if available:
        router.chat_routes = {model: available[0] for model in chat_models if model not in available}
    router.embed_enabled = embedding["ok"]
    key = config.MODEL_EMBED
    results[f"embed:{key}" if key in results else key] = embedding
    return results
