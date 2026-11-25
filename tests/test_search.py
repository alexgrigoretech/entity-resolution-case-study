import asyncio
import json

import httpx
import pytest

from resolver import config
from resolver.httpcache import Cache, CachedClient
from resolver.search import HOMEPAGE_LIMIT, Hit, Serper, classify, homepage_links


@pytest.mark.parametrize(
    ("kind", "url", "expected"),
    [
        (
            "organizations",
            "https://uk.linkedin.com/company/Amber%2DCove/?ref=x#people",
            ("linkedin", "https://www.linkedin.com/company/Amber-Cove"),
        ),
        ("organizations", "https://linkedin.com/school/amber", None),
        ("organizations", "https://www.linkedin.com/showcase/amber", None),
        ("organizations", "https://linkedin.com/company/amber/jobs", None),
        (
            "organizations",
            "https://www.crunchbase.com/organization/birch",
            ("crunchbase", "https://www.crunchbase.com/organization/birch"),
        ),
        ("organizations", "https://crunchbase.com/person/brin", None),
        ("organizations", "https://twitter.com/Birch?origin=foo", ("x", "https://x.com/Birch")),
        ("organizations", "https://x.com/intent", None),
        ("organizations", "https://x.com/Birch/status/1", None),
        ("organizations", "https://www.birch.example/contact/", ("domain", "birch.example")),
        ("organizations", "https://en.wikipedia.org/wiki/Birch", None),
        ("organizations", "https://signal.nfx.com/company/birch", None),
        ("people", "https://de.linkedin.com/in/En%C3%A9e#about", ("linkedin", "https://www.linkedin.com/in/Enée")),
        ("people", "https://linkedin.com/company/amber", None),
        ("people", "https://crunchbase.com/person/brin/", ("crunchbase", "https://www.crunchbase.com/person/brin")),
        ("people", "https://x.com/brin", ("x", "https://x.com/brin")),
        ("people", "https://x.com/%68ome", None),
        ("people", "https://brin.example", None),
    ],
)
def test_classify(kind, url, expected):
    assert classify(url, kind) == expected


def test_homepage_extracts_only_supported_social_links(tmp_path):
    html = (
        '<a href="//uk.linkedin.com/company/Amber">About</a><a href="https://twitter.com/Ambers">Feed</a>'
        '<a href="https://crunchbase.com/organization/amber/">Profile</a><a href="https://linkedin.com/in/amber">Team</a>'
        '<a href="/contact">Contact</a><a href="https://x.com/home">Home</a>'
    )

    def handler(request):
        assert request.method == "GET"
        assert request.url == "https://amber.example"
        return httpx.Response(200, html=html + " " * HOMEPAGE_LIMIT + '<a href="https://x.com/outside">Outside</a>')

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = CachedClient(Cache(tmp_path / "web.sqlite"), http)
            assert await homepage_links(client, "linkedin.com") == {}
            return await homepage_links(client, "amber.example")

    assert asyncio.run(run()) == {
        "linkedin": [("https://www.linkedin.com/company/Amber", "homepage")],
        "crunchbase": [("https://www.crunchbase.com/organization/amber", "homepage")],
        "x": [("https://x.com/Ambers", "homepage")],
    }


def test_homepage_uses_public_text_fetch():
    class Client:
        async def get_text(self, namespace, url, *, max_chars):
            assert (namespace, url, max_chars) == ("web", "https://amber.example", HOMEPAGE_LIMIT)
            return '<a href="https://x.com/amber">Feed</a>'

    assert asyncio.run(homepage_links(Client(), "amber.example")) == {"x": [("https://x.com/amber", "homepage")]}


def test_homepage_extracts_structured_profiles_and_upgrades_anchors():
    class Client:
        async def get_text(self, namespace, url, *, max_chars):
            return """
                <a href="https://linkedin.com/company/amber">LinkedIn</a>
                <script type="application/ld+json">
                {"@type": "Organization", "sameAs": "https://uk.linkedin.com/company/Amber/"}
                </script>
                <script type="application/ld+json">{broken}</script>
                <script type="application/ld+json">
                [{"nested": {"@graph": [
                    {"@type": ["Thing", "NGO"], "sameAs": [
                        "https://crunchbase.com/organization/amber/", "https://twitter.com/AmberFeed", null
                    ]},
                    {"@type": "Person", "sameAs": "https://x.com/unrelated"}
                ]}}]
                </script>
                <meta name="twitter:site" content="@AmberNews">
                <meta property="twitter:site" content="AmberUpdates">
                <a href="https://x.com/AmberNews">News</a>
            """

    assert asyncio.run(homepage_links(Client(), "amber.example")) == {
        "linkedin": [("https://www.linkedin.com/company/Amber", "structured")],
        "crunchbase": [("https://www.crunchbase.com/organization/amber", "structured")],
        "x": [
            ("https://x.com/AmberFeed", "structured"),
            ("https://x.com/AmberNews", "structured"),
            ("https://x.com/AmberUpdates", "structured"),
        ],
    }


def test_serper_reads_organic_and_knowledge_graph(tmp_path):
    def handler(request):
        assert request.url == "https://google.serper.dev/search"
        assert request.headers["X-API-KEY"] == "serper-placeholder"
        assert json.loads(request.content) == {"q": "Amber Grove", "num": config.SERPER_NUM}
        return httpx.Response(
            200,
            json={
                "organic": [
                    {"title": "Amber", "link": "https://amber.example", "snippet": "Organic result"},
                    {"title": "No link"},
                ],
                "knowledgeGraph": {"title": "Grove", "website": "https://grove.example", "description": "Graph result"},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = CachedClient(Cache(tmp_path / "search.sqlite"), http)
            assert await Serper(client, None).search("unused") == []
            return await Serper(client, "serper-placeholder").search("Amber Grove")

    assert asyncio.run(run()) == [
        Hit("Amber", "https://amber.example", "Organic result", "organic"),
        Hit("Grove", "https://grove.example", "Graph result", "knowledge_graph"),
    ]
