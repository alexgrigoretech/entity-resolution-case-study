"""Cache successful HTTP responses and retry transient failures."""

import asyncio
import hashlib
import json
import logging
import math
import sqlite3
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from . import config
from .runclient import RequestFailure, RequestSender, before_deadline

Payload = dict[str, Any] | str


class Cache:
    def __init__(self, path: str | Path = Path("cache/cache.sqlite")) -> None:
        self.path = Path(path)
        # The event-loop thread opens, uses, and closes this connection.
        # Transactions contain no awaits.
        self._db: sqlite3.Connection | None = None

    def _connection(self) -> sqlite3.Connection:
        if self._db is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
            try:
                with connection:
                    connection.execute("PRAGMA journal_mode=WAL").fetchone()
                    connection.execute("PRAGMA synchronous=NORMAL")
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, namespace TEXT, "
                        "request TEXT, response TEXT, created_at TEXT)"
                    )
            except BaseException:
                connection.close()
                raise
            self._db = connection
        return self._db

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def get(self, key: str) -> Payload | None:
        row = self._connection().execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, namespace: str, request: str, response: Payload) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO responses (key, namespace, request, response, created_at) "
                "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (key, namespace, request, json.dumps(response)),
            )


def reported_cost(usage: dict) -> float | None:
    cost = usage.get("cost")
    if type(cost) in (int, float) and cost > 0:
        return float(cost)
    details = usage.get("cost_details")
    upstream = details.get("upstream_inference_cost") if isinstance(details, dict) else None
    if type(upstream) in (int, float) and upstream > 0:
        return float(upstream)
    return 0.0 if type(cost) in (int, float) else None


@dataclass
class NamespaceUsage:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    truncated: int = 0

    def add_response(self, response: Payload) -> None:
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            if isinstance(prompt, int):
                self.prompt_tokens += prompt
            if isinstance(completion, int):
                self.completion_tokens += completion
            if (cost := reported_cost(usage)) is not None:
                self.cost += cost


class Ledger:
    def __init__(self) -> None:
        self._namespaces: dict[str, NamespaceUsage] = {}

    def namespace(self, name: str) -> NamespaceUsage:
        return self._namespaces.setdefault(name, NamespaceUsage())

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        return {name: asdict(usage) for name, usage in self._namespaces.items()}


class CachedClient:
    def __init__(
        self,
        cache: Cache | None = None,
        client: httpx.AsyncClient | None = None,
        ledger: Ledger | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.cache = cache if cache is not None else Cache()
        self.client = client if client is not None else httpx.AsyncClient()
        self.ledger = ledger if ledger is not None else Ledger()
        self.sleep = sleep
        self.sender = RequestSender(self.client, sleep)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def post_json(
        self,
        namespace: str,
        url: str,
        body: dict[str, Any],
        headers: Mapping[str, str] | None = None,
        usable: Callable[[dict], bool] | None = None,
        *,
        probe: bool = False,
        cacheable: Callable[[Payload], bool] | None = None,
    ) -> dict[str, Any] | None:
        def valid(payload: Payload) -> bool:
            return isinstance(payload, dict) and (usable is None or usable(payload))

        def parse(response: httpx.Response) -> dict[str, Any]:
            payload = response.json()
            if not valid(payload):
                raise ValueError("Unusable JSON response")
            return payload

        response = await self._request(
            namespace, "POST", url, parse, body, headers, valid=valid, probe=probe, cacheable=cacheable
        )
        return response if isinstance(response, dict) else None

    async def get_text(self, namespace: str, url: str, max_chars: int | None = None) -> str | None:
        response = await self._request(
            namespace, "GET", url, lambda result: result.text[:max_chars], max_chars=max_chars
        )
        return response if isinstance(response, str) else None

    def remaining(self) -> float:
        return math.inf

    def expired(self) -> bool:
        return self.remaining() <= 0

    async def _request(
        self,
        namespace: str,
        method: str,
        url: str,
        parse: Callable[[httpx.Response], Payload],
        body: dict[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        max_chars: int | None = None,
        valid: Callable[[Payload], bool] | None = None,
        probe: bool = False,
        cacheable: Callable[[Payload], bool] | None = None,
    ) -> Payload | None:
        if self.expired():
            if probe:
                raise RequestFailure(None, "Resolver deadline reached")
            return None
        usage = self.ledger.namespace(namespace)
        usage.calls += 1
        try:
            if not probe:
                request_data = {"method": method, "url": url, "body": body}
                if max_chars is not None:
                    request_data["max_chars"] = max_chars
                if namespace == "web":
                    request_data["html"] = True
                request = json.dumps(request_data, sort_keys=True)
                key = hashlib.sha256(f"{namespace}\n{request}".encode()).hexdigest()
                cached = self.cache.get(key)
                if cached is not None and (cacheable is None or cacheable(cached)) and (valid is None or valid(cached)):
                    usage.cache_hits += 1
                    return cached
            work = self.sender.send(
                namespace,
                method,
                url,
                parse,
                body,
                headers,
                usage,
                self.remaining,
                config.PROBE_ATTEMPTS if probe else config.MAX_ATTEMPTS,
            )
            payload = (
                await work if math.isinf(self.remaining()) else await before_deadline(self.remaining, self.sleep, work)
            )
        except RequestFailure as error:
            usage.failures += 1
            if probe:
                raise
            detail = f"HTTP {error.status}" if error.status is not None else str(error)
            logging.getLogger(__name__).warning("Request failed in %s (%s)", namespace, detail)
            return None
        except Exception as error:
            usage.failures += 1
            if probe:
                raise RequestFailure(None, type(error).__name__) from error
            logging.getLogger(__name__).warning("Request failed in %s (%s)", namespace, type(error).__name__)
            return None
        if not probe and (cacheable is None or cacheable(payload)):
            try:
                self.cache.put(key, namespace, request, payload)
            except (OSError, sqlite3.Error):
                logging.getLogger(__name__).warning("Could not cache a response in %s", namespace)
        return payload


class RunClient(CachedClient):
    def __init__(
        self,
        cache: Cache,
        ledger: Ledger,
        deadline: float,
        transport: httpx.AsyncBaseTransport | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.deadline = deadline
        self.clock = clock
        super().__init__(cache, httpx.AsyncClient(transport=transport, follow_redirects=False), ledger, sleep)

    def remaining(self) -> float:
        return self.deadline - self.clock()
