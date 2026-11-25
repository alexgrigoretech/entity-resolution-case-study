"""Search the web and classify candidate links."""

import json
import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlsplit

from . import config
from .httpcache import CachedClient
from .httpcache import RunClient as RunClient  # noqa: PLC0414
from .normalize import link_key
from .records import Kind
from .runclient import blocked_host

LOGGER = logging.getLogger(__name__)
SERPER_URL = "https://google.serper.dev/search"
HOMEPAGE_LIMIT = 500_000
X_RESERVED = frozenset(["home", "share", "intent", "search", "hashtag", "i", "explore", "login", "signup"])
X_HANDLE = re.compile(r"[A-Za-z0-9_]+")
ORG_TYPES = frozenset(
    (
        "Organization",
        "Corporation",
        "LocalBusiness",
        "NGO",
        "EducationalOrganization",
        "FinancialService",
        "ProfessionalService",
    )
)


@dataclass(frozen=True, slots=True)
class Hit:
    title: str
    link: str
    snippet: str
    source: str


class Serper:
    def __init__(self, client: CachedClient, api_key: str | None) -> None:
        self.client = client
        self.api_key = api_key

    async def search(self, query: str, num: int | None = None) -> list[Hit]:
        if not self.api_key:
            return []
        try:
            response = await self.client.post_json(
                "serper",
                SERPER_URL,
                {"q": query, "num": config.SERPER_NUM if num is None else num},
                {"X-API-KEY": self.api_key},
                usable=lambda body: "error" not in body and "message" not in body,
            )
            if response is None:
                return []
            hits = []
            for source, items, link_field in (
                ("organic", response.get("organic"), "link"),
                ("knowledge_graph", [response.get("knowledgeGraph")], "website"),
            ):
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict) and isinstance(link := item.get(link_field), str) and link:
                        title = item.get("title")
                        snippet = item.get("snippet") or item.get("description")
                        hits.append(
                            Hit(
                                title if isinstance(title, str) else "",
                                link,
                                snippet if isinstance(snippet, str) else "",
                                source,
                            )
                        )
            return hits
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Serper search failed (%s)", type(error).__name__)
            return []


def classify(url: str, kind: Kind) -> tuple[str, str] | None:
    try:
        parsed = urlsplit(url)
        if not parsed.scheme and not parsed.netloc:
            parsed = urlsplit(f"//{url}")
    except ValueError:
        return None
    if parsed.scheme not in ("https", "http", "") or not parsed.hostname or parsed.username is not None:
        return None
    host = parsed.hostname.rstrip(".").removeprefix("www.")
    if not re.fullmatch(r"[\w.-]+\.[\w-]+", host):
        return None
    parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
    if host == "linkedin.com" or host.endswith(".linkedin.com"):
        category = "company" if kind == "organizations" else "in"
        if len(parts) == 2 and parts[0] == category and parts[1]:
            return "linkedin", f"https://www.linkedin.com/{category}/{parts[1]}"
    elif host == "crunchbase.com":
        category = "organization" if kind == "organizations" else "person"
        if len(parts) == 2 and parts[0] == category and parts[1]:
            return "crunchbase", f"https://www.crunchbase.com/{category}/{parts[1]}"
    elif host in ("x.com", "twitter.com", "mobile.twitter.com"):
        if len(parts) == 1 and X_HANDLE.fullmatch(parts[0]) and parts[0].lower() not in X_RESERVED:
            return "x", f"https://x.com/{parts[0]}"
    elif kind == "organizations" and not blocked_host(host):
        return "domain", host
    return None


class HomepageParser(HTMLParser):
    def __init__(self, homepage: str) -> None:
        super().__init__()
        self.homepage = homepage
        self.links: dict[str, dict[str, tuple[str, str]]] = {}
        self.script: list[str] | None = None

    def _add(self, url: str, source: str) -> None:
        match = classify(url, "organizations")
        if match is None or match[0] == "domain":
            return
        field, canonical = match
        links = self.links.setdefault(field, {})
        key = link_key(field, canonical)
        if source == "structured" and links.get(key, ("", ""))[1] == "homepage":
            links[key] = (canonical, source)
        else:
            links.setdefault(key, (canonical, source))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script" and (attributes.get("type") or "").strip().lower() == "application/ld+json":
            self.script = []
        elif tag == "meta" and any(
            (attributes.get(key) or "").lower() == "twitter:site" for key in ("name", "property")
        ):
            handle = (attributes.get("content") or "").strip().removeprefix("@")
            if X_HANDLE.fullmatch(handle):
                self._add(f"https://x.com/{handle}", "structured")
        elif tag == "a" and (href := attributes.get("href")):
            try:
                url = urljoin(self.homepage, href)
            except ValueError:
                return
            self._add(url, "homepage")

    def handle_data(self, data: str) -> None:
        if self.script is not None:
            self.script.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "script" or self.script is None:
            return
        script, self.script = "".join(self.script), None
        try:
            pending = [json.loads(script)]
        except (ValueError, RecursionError):
            return
        while pending:
            value = pending.pop()
            if isinstance(value, list):
                pending.extend(reversed(value))
            elif isinstance(value, dict):
                pending.extend(reversed(value.values()))
                types = value.get("@type")
                types = types if isinstance(types, list) else [types]
                if any(isinstance(item, str) and item in ORG_TYPES for item in types):
                    urls = value.get("sameAs")
                    for url in urls if isinstance(urls, list) else [urls]:
                        if isinstance(url, str):
                            self._add(url, "structured")


async def homepage_links(client: CachedClient, domain: str) -> dict[str, list[tuple[str, str]]]:
    homepage = f"https://{domain}"
    if classify(homepage, "organizations") != ("domain", domain):
        return {}
    try:
        for url in (homepage, f"https://www.{domain}"):
            body = await client.get_text("web", url, max_chars=HOMEPAGE_LIMIT)
            if body is not None:
                parser = HomepageParser(url)
                parser.feed(body)
                return {field: list(links.values()) for field, links in parser.links.items()}
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Homepage extraction failed (%s)", type(error).__name__)
    return {}
