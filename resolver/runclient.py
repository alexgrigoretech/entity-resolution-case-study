"""Send rate-limited HTTP requests within the run deadline."""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import TYPE_CHECKING, Any

import httpx

from . import config

if TYPE_CHECKING:
    from .httpcache import NamespaceUsage, Payload

WEB_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; entity-resolver/0.1; research)", "Accept": "text/html"}
MAX_REDIRECTS = 3


class RequestFailure(Exception):
    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(message)
        self.status = status


def blocked_host(host: str) -> bool:
    host = host.rstrip(".")
    return any(host == blocked or host.endswith(f".{blocked}") for blocked in config.BLOCKED_HOSTS)


def _retry_delay(attempt: int, response: httpx.Response | None) -> float:
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after is not None:
        try:
            seconds = float(retry_after)
        except ValueError:
            pass
        else:
            if math.isfinite(seconds) and seconds >= 0:
                return seconds
    return random.uniform(0, min(config.BACKOFF_CAP, config.BACKOFF_BASE * 2**attempt))


def _error_message(response: httpx.Response, default: str) -> str:
    try:
        body = response.json()
    except (ValueError, RecursionError):
        return default
    if isinstance(body, dict):
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        if isinstance(message, str) and message.strip():
            return message
        if isinstance(message := body.get("message"), str) and message.strip():
            return message
    return default


async def before_deadline[T](
    remaining: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    work: Coroutine[Any, Any, T],
) -> T:
    async def wait() -> None:
        if (seconds := remaining()) > 0:
            await sleep(seconds)
        if (seconds := remaining()) > 0:
            await asyncio.sleep(seconds)

    request = asyncio.create_task(work)
    timer = asyncio.create_task(wait())
    try:
        await asyncio.wait((request, timer), return_when=asyncio.FIRST_COMPLETED)
        if request.done() and remaining() > 0:
            return request.result()
        raise RequestFailure(None, "Resolver deadline reached")
    finally:
        for task in (request, timer):
            if not task.done():
                task.cancel()
        await asyncio.gather(request, timer, return_exceptions=True)


class RequestSender:
    def __init__(self, client: httpx.AsyncClient, sleep: Callable[[float], Awaitable[None]]) -> None:
        self.client = client
        self.sleep = sleep
        self.limits: dict[str, asyncio.Semaphore] = {}

    async def _fetch(
        self,
        namespace: str,
        method: str,
        url: str,
        body: dict[str, Any] | None,
        headers: Mapping[str, str] | None,
        remaining: Callable[[], float],
    ) -> httpx.Response:
        web = namespace == "web"
        target = httpx.URL(url)
        for hop in range(MAX_REDIRECTS + 1):
            seconds = remaining()
            if seconds <= 0:
                raise RequestFailure(None, "Resolver deadline reached")
            timeout = httpx.Timeout(min(config.TIMEOUTS.get(namespace, 5.0), seconds), connect=min(10.0, seconds))
            response = await self.client.request(
                method,
                target,
                json=body,
                headers=WEB_HEADERS if web else headers,
                follow_redirects=False,
                timeout=timeout,
            )
            if not web or response.next_request is None or hop == MAX_REDIRECTS:
                return response
            target = response.next_request.url
            if target.scheme not in ("http", "https") or target.userinfo or blocked_host(target.host):
                return response
        raise AssertionError("Redirect loop exceeded its limit")

    async def send(
        self,
        namespace: str,
        method: str,
        url: str,
        parse: Callable[[httpx.Response], Payload],
        body: dict[str, Any] | None,
        headers: Mapping[str, str] | None,
        usage: NamespaceUsage,
        remaining: Callable[[], float],
        attempts: int,
    ) -> Payload:
        if namespace not in self.limits:
            self.limits[namespace] = asyncio.Semaphore(config.CONCURRENCY.get(namespace, config.DEFAULT_CONCURRENCY))
        failure = RequestFailure(None, "No request attempted")
        for attempt in range(attempts):
            response = None
            try:
                async with self.limits[namespace]:
                    response = await self._fetch(namespace, method, url, body, headers, remaining)
            except httpx.TransportError as error:
                failure = RequestFailure(None, type(error).__name__)
            else:
                status = response.status_code
                if response.is_success:
                    content_type = response.headers.get("Content-Type", "").partition(";")[0].strip().lower()
                    if namespace == "web" and content_type != "text/html":
                        raise RequestFailure(status, "Expected text/html")
                    try:
                        payload = parse(response)
                    except (TypeError, ValueError, RecursionError):
                        try:
                            usage.add_response(response.json())
                        except (ValueError, RecursionError):
                            pass
                        failure = RequestFailure(status, _error_message(response, "Unusable response"))
                    else:
                        usage.add_response(payload)
                        return payload
                else:
                    failure = RequestFailure(status, _error_message(response, response.reason_phrase))
                    if status != 429 and not 500 <= status < 600:
                        raise failure
            if attempt + 1 < attempts:
                usage.retries += 1
                await self.sleep(_retry_delay(attempt, response))
        raise failure
