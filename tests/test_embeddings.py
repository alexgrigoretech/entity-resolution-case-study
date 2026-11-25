import asyncio

import httpx
import numpy as np
import pytest

from resolver.candidates import CandidateIndex
from resolver.embeddings import embed_matrix, text_for, top_n
from resolver.httpcache import Cache, CachedClient
from resolver.llm import OpenRouter
from resolver.records import DatabaseRecord, InputRecord


def test_embedding_text_uses_first_mention_and_full_article_without_one():
    name = "Cora Gale"
    article = "a" * 230 + name.lower() + "b" * 212 + name
    center = 230 + len(name) // 2
    assert text_for(InputRecord("a", name, article, "people")) == f"{name}. {article[center - 200 : center + 200]}"
    assert text_for(InputRecord("b", "Enid Firth", article, "people")) == f"Enid Firth. {article}"
    assert text_for(InputRecord("c", "Fir Grove", "Grows seedlings.")) == "Fir Grove. Grows seedlings."
    assert text_for(DatabaseRecord("d", "Fir Grove", "Grows seedlings.")) == "Fir Grove. Grows seedlings."


def test_embeddings_are_normalized_and_ranked(tmp_path):
    def handler(request):
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": vec} for i, vec in enumerate([[3, 4], [0, 0], [1, 0]])]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            router = OpenRouter(CachedClient(Cache(tmp_path / "embed.sqlite"), http), "placeholder")
            return await embed_matrix(router, ["first", "second", "third"])

    matrix = asyncio.run(run())
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix, [[0.6, 0.8], [0, 0], [1, 0]])
    ranked = top_n(matrix[2], matrix, 2)
    assert [index for index, _ in ranked] == [2, 0]
    assert [score for _, score in ranked] == pytest.approx([1, 0.6])


def test_embedding_candidates_follow_lexical_candidates():
    records = [
        DatabaseRecord(str(i), name, "")
        for i, name in enumerate(["Boreal Birch", "Amber Canopy", "Lynx Works", "Poppy Harbor", "Cidery"])
    ]
    index = CandidateIndex(records, "organizations")
    record = InputRecord("a", "Boreal Birch", "")
    matrix = np.array([[0, 1], [1, 0], [0.8, 0.6], [0.6, 0.8], [-1, 0]], dtype=np.float32)
    assert [(row.record.id, row.features["embed_score"]) for row in index.candidates(record)] == [("0", 0.0)]
    ranked = index.candidates(record, matrix[1], matrix)
    assert [row.record.id for row in ranked] == ["0", "1", "2", "3"]
    assert [row.features["embed_score"] for row in ranked] == pytest.approx([0, 1, 0.8, 0.6])
