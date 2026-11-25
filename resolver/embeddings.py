"""Request embeddings and build normalized matrices for candidate retrieval."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from . import config
from .records import DatabaseRecord, InputRecord

if TYPE_CHECKING:
    from .llm import OpenRouter

LOGGER = logging.getLogger(f"{__package__}.llm")
EMBED_BATCH_SIZE = 64


def _has_embeddings(body: dict[str, Any]) -> bool:
    return isinstance(body.get("data"), list) and bool(body["data"])


def _vectors(response: dict[str, Any], count: int) -> list[list[float]]:
    entries = sorted(response["data"], key=lambda entry: entry["index"])
    indices = [entry["index"] for entry in entries]
    if any(type(index) is not int for index in indices) or indices != list(range(count)):
        raise ValueError("Unexpected embedding indices")
    vectors = []
    for entry in entries:
        vector = entry["embedding"]
        if (
            not isinstance(vector, list)
            or not vector
            or (vectors and len(vector) != len(vectors[0]))
            or any(type(value) not in (float, int) or not math.isfinite(value) for value in vector)
        ):
            raise ValueError("Invalid embedding vector")
        vectors.append(vector)
    return vectors


async def probe_embed(openrouter: OpenRouter, model: str) -> None:
    def usable(payload: dict[str, Any]) -> bool:
        try:
            _vectors(payload, 1)
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            return False
        return True

    await openrouter._post("embed", "embeddings", {"model": model, "input": ["ok"]}, usable, probe=True)


async def embed(openrouter: OpenRouter, model: str, texts: list[str]) -> list[list[float]] | None:
    if not openrouter.embed_enabled:
        return None
    batches = [texts[start : start + EMBED_BATCH_SIZE] for start in range(0, len(texts), EMBED_BATCH_SIZE)]
    vectors = []
    try:
        responses = await asyncio.gather(
            *(
                openrouter._post("embed", "embeddings", {"model": model, "input": batch}, _has_embeddings)
                for batch in batches
            ),
            return_exceptions=True,
        )
        for batch, response in zip(batches, responses, strict=True):
            if isinstance(response, BaseException):
                raise response
            if response is None:
                return None
            batch_vectors = _vectors(response, len(batch))
            if vectors and len(batch_vectors[0]) != len(vectors[0]):
                raise ValueError("Invalid embedding vector")
            vectors.extend(batch_vectors)
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("OpenRouter embeddings failed (%s)", type(error).__name__)
        return None
    return vectors


def text_for(record: DatabaseRecord | InputRecord) -> str:
    if isinstance(record, DatabaseRecord):
        return f"{record.name}. {record.description}"
    text = record.text
    if record.kind == "people" and record.name:
        match = re.search(re.escape(record.name), text, re.IGNORECASE)
        if match:
            center = (match.start() + match.end()) // 2
            start = max(0, min(center - 200, len(text) - 400))
            text = text[start : start + 400]
    return f"{record.name}. {text}"


async def embed_matrix(openrouter: OpenRouter, texts: list[str]) -> np.ndarray | None:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors = await openrouter.embed(config.MODEL_EMBED, texts)
    if vectors is None:
        return None
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)


def top_n(input_vector: np.ndarray, matrix: np.ndarray, n: int) -> list[tuple[int, float]]:
    """Rank rows by cosine similarity using normalized embeddings."""
    if not len(matrix) or n <= 0:
        return []
    scores = matrix @ input_vector
    return [(int(index), float(scores[index])) for index in np.argsort(-scores, kind="stable")[:n]]
