"""Rank database records using names, text and organization context."""

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NotRequired, TypedDict

import numpy as np
from rapidfuzz import fuzz

from . import config
from .embeddings import top_n
from .normalize import name_key, org_key, person_name_variants, slug_aliases
from .records import DatabaseRecord, InputRecord, Kind

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "s",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    ]
)
GENERIC_ORG_WORDS = frozenset(["capital", "ventures", "partners", "group", "fund", "management"])


def tokens(text: str) -> list[str]:
    return [token for token in name_key(text).split() if token not in STOPWORDS]


def passes_lexical(name_score: float, text_score: float) -> bool:
    return name_score >= config.MIN_NAME_SCORE or text_score >= config.MIN_TEXT_SCORE


class Tfidf:
    def __init__(self, documents: Iterable[str]) -> None:
        frequencies: Counter[str] = Counter()
        count = 0
        for document in documents:
            frequencies.update(set(tokens(document)))
            count += 1
        self.idf = {token: math.log((1 + count) / (1 + freq)) + 1 for token, freq in frequencies.items()}
        self.unseen_idf = math.log(1 + count) + 1

    def vector(self, text: str) -> dict[str, float]:
        weights = {
            token: count * self.idf.get(token, self.unseen_idf) for token, count in Counter(tokens(text)).items()
        }
        length = math.sqrt(sum(weight * weight for weight in weights.values()))
        return {token: weight / length for token, weight in weights.items()} if length else {}

    @staticmethod
    def cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return min(1.0, sum(weight * right.get(token, 0.0) for token, weight in left.items()))


class Features(TypedDict):
    name_score: float
    exact_name: bool
    text_score: float
    name_rivals: int
    org_overlap: NotRequired[int]
    name_in_text: NotRequired[bool]
    embed_score: NotRequired[float]
    link_hit: NotRequired[list[str]]


@dataclass(frozen=True, slots=True)
class Candidate:
    record: DatabaseRecord
    features: Features


@dataclass(frozen=True, slots=True)
class IndexedRecord:
    record: DatabaseRecord
    names: set[str]
    text_vector: dict[str, float]
    org_tokens: set[str]
    name_rivals: int


def _organization_tokens(description: str) -> set[str]:
    phrases = re.findall(r"\s+at\s+([^;.]+)", description, re.IGNORECASE)
    return set(tokens(" ".join(phrases))) - GENERIC_ORG_WORDS


class CandidateIndex:
    def __init__(self, records: list[DatabaseRecord], kind: Kind) -> None:
        self.kind = kind
        self.tfidf = Tfidf(record.description for record in records)
        key = org_key if kind == "organizations" else name_key
        rivals = Counter(key(record.name) for record in records)
        self.records = [
            IndexedRecord(
                record,
                {key(record.name)} | slug_aliases(record)
                if kind == "organizations"
                else person_name_variants(record.name),
                self.tfidf.vector(f"{record.name} {record.description}"),
                _organization_tokens(record.description) if kind == "people" else set(),
                rivals[key(record.name)],
            )
            for record in records
        ]

    def candidates(
        self,
        record: InputRecord,
        embedding: np.ndarray | None = None,
        matrix: np.ndarray | None = None,
        link_hits: dict[str, list[str]] | None = None,
    ) -> list[Candidate]:
        names = {org_key(record.name)} if self.kind == "organizations" else person_name_variants(record.name)
        names.discard("")
        text_vector = self.tfidf.vector(record.text)
        article = name_key(record.text) if self.kind == "people" else ""
        article_tokens = set(article.split())
        input_name = name_key(record.name)
        name_in_text = bool(input_name) and f" {input_name} " in f" {article} "
        found = []
        extra: dict[int, Candidate] = {}
        link_hits = link_hits or {}
        has_embeddings = (
            embedding is not None and matrix is not None and matrix.shape == (len(self.records), *embedding.shape)
        )
        embedded = dict(top_n(embedding, matrix, config.EMBED_TOP_N)) if has_embeddings else {}
        for position, indexed in enumerate(self.records):
            name_score = max(
                (
                    max(fuzz.token_set_ratio(left, right), fuzz.ratio(left, right))
                    for left in names
                    for right in indexed.names
                    if right
                ),
                default=0.0,
            )
            text_score = self.tfidf.cosine(text_vector, indexed.text_vector)
            passes = passes_lexical(name_score, text_score)
            link_hit = link_hits.get(indexed.record.id)
            if not passes and position not in embedded and not link_hit:
                continue
            features: Features = {
                "name_score": name_score,
                "exact_name": not names.isdisjoint(indexed.names),
                "text_score": text_score,
                "name_rivals": indexed.name_rivals,
                "embed_score": float(matrix[position] @ embedding) if has_embeddings else 0.0,
            }
            if link_hit:
                features["link_hit"] = link_hit
            if self.kind == "people":
                features["org_overlap"] = len(article_tokens & indexed.org_tokens)
                features["name_in_text"] = name_in_text
            candidate = Candidate(indexed.record, features)
            if passes:
                found.append(candidate)
            if position in embedded or link_hit:
                extra[position] = candidate
        found.sort(key=lambda item: (-item.features["name_score"], -item.features["text_score"], item.record.id))
        found = found[: config.TOP_K]
        included = {candidate.record.id for candidate in found}
        for position in dict.fromkeys((*embedded, *extra)):
            candidate = extra[position]
            if candidate.record.id not in included:
                found.append(candidate)
        return found
