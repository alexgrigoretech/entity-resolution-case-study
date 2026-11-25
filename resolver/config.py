"""Thresholds for matching, deduplication and request retries."""

import os

TOP_K = 8
MIN_NAME_SCORE = 70
MIN_TEXT_SCORE = 0.35
ORG_NAME_FUZZY = 90
ORG_NAME_WITH_TEXT = 60
ORG_TEXT_EXACT = 0.06
ORG_TEXT_FUZZY = 0.12
ORG_TEXT_ALONE = 0.45
PERSON_NAME_SCORE = 92
PERSON_ORG_OVERLAP = 1
AMBIGUOUS_NAME_SCORE = 80
AMBIGUOUS_BELOW = 0.6

DEDUPE_NAME_SCORE = 95
DEDUPE_ORG_TEXT = 0.20
DEDUPE_PERSON_OVERLAP = 3
DEDUPE_PERSON_WINDOW = 24
DEDUPE_MODEL_CONF = 0.7

MODEL_FAST = os.getenv("RESOLVER_MODEL_FAST", "openai/gpt-6.1-terra")
MODEL_STRONG = os.getenv("RESOLVER_MODEL_STRONG", "openai/gpt-6.2-sol")
MODEL_RESEARCH = os.getenv("RESOLVER_MODEL_RESEARCH", MODEL_STRONG)
MODEL_EMBED = os.getenv("RESOLVER_MODEL_EMBED", "openai/text-embedding-3-small")
MODEL_PRICES = {MODEL_FAST: (1, 5), MODEL_STRONG: (2, 10), MODEL_EMBED: (0.02, 0)}
MAX_TOKENS = 800
REASONING_EFFORT = os.getenv("RESOLVER_REASONING_EFFORT", "low")
ESCALATE_BELOW = 0.8
REVIEW_BELOW = 0.7
LINK_MIN_CONF = {"domain": 0.6, "linkedin": 0.6, "crunchbase": 0.6, "x": 0.7}
LINK_BONUS_MIN_CONF = 0.5
LINK_SOURCE_BONUS = {"structured": 0.25, "homepage": 0.15}
EMBED_TOP_N = 3
SERPER_NUM = 10
RESEARCH_ENABLED = os.getenv("RESOLVER_RESEARCH", "").lower() not in {"0", "false", "no"}
RESEARCH_NUM = 5
RESEARCH_MAX_CHARS = 20000
RESEARCH_MAX_TOKENS = 1600
RESEARCH_TIERS = {"verified": 0.9, "high": 0.75, "medium": 0.55}
MULTILABEL_DOMAIN_SUFFIXES = frozenset("co.uk org.uk ac.uk com.au com.br co.in co.jp co.kr co.za com.sg com.mx co.nz com.tr com.cn".split())  # fmt: skip  # noqa: SIM905
GENERIC_DOMAIN_TOKENS = frozenset("group capital ventures venture partners holdings global labs tech data systems "  # noqa: SIM905
                                  "solutions digital international company fund management bank health energy media software".split())  # fmt: skip
AGREE_BONUS = 0.15
DISAGREE_POLICY = "blank"
QUOTE_QUERIES: bool = os.getenv("RESOLVER_QUOTE_QUERIES", "").lower() in {"1", "true", "yes"}
DEADLINE_SECONDS = 540
TIMEOUTS = {"llm": 90.0, "embed": 60.0, "serper": 20.0, "web": 10.0}
BLOCKED_HOSTS = frozenset(
    [
        "linkedin.com",
        "crunchbase.com",
        "x.com",
        "twitter.com",
        "facebook.com",
        "instagram.com",
        "youtube.com",
        "wikipedia.org",
        "bloomberg.com",
        "pitchbook.com",
        "zoominfo.com",
        "techcrunch.com",
        "github.com",
        "medium.com",
        "tracxn.com",
        "cbinsights.com",
        "dealroom.co",
        "signal.nfx.com",
        "rocketreach.co",
        "apollo.io",
        "glassdoor.com",
        "g2.com",
    ]
)

CONCURRENCY = {"llm": 8, "embed": 4, "serper": 8, "web": 8}
DEFAULT_CONCURRENCY = 8
MAX_ATTEMPTS = 5
PROBE_ATTEMPTS = 3
BACKOFF_BASE = 1.0
BACKOFF_CAP = 30.0
