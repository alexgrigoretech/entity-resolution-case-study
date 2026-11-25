"""Resolve two link answers using their agreement and confidence."""

from dataclasses import dataclass
from typing import Literal

from . import config
from .normalize import link_key

type Agreement = Literal["both", "search", "research", "disagree"]
AGREEMENTS: tuple[Agreement, ...] = ("both", "search", "research", "disagree")


@dataclass(frozen=True, slots=True)
class LinkAnswer:
    url: str
    confidence: float
    confirmation_only: bool = False


@dataclass(frozen=True, slots=True)
class CombinedLink:
    agreement: Agreement
    emitted: LinkAnswer | None


def combine(field: str, search: LinkAnswer | None, research: LinkAnswer | None, policy: str) -> CombinedLink | None:
    minimum = config.LINK_MIN_CONF[field]
    if search and research and link_key(field, search.url) == link_key(field, research.url):
        answer = LinkAnswer(search.url, min(1.0, max(search.confidence, research.confidence) + config.AGREE_BONUS))
        return CombinedLink("both", answer if answer.confidence >= minimum else None)
    if search and search.confidence < minimum:
        search = None
    if research and (research.confidence < minimum or research.confirmation_only):
        research = None
    if search and research:
        return CombinedLink(
            "disagree", max(search, research, key=lambda link: link.confidence) if policy == "higher" else None
        )
    if search:
        return CombinedLink("search", search)
    return CombinedLink("research", research) if research else None
