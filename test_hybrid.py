"""Tests for semantic/hybrid retrieval using a fake embedder (no torch needed)."""

import numpy as np
from retrieval import (
    chunk_document, EmbeddingsRetriever, HybridRetriever,
    reciprocal_rank_fusion, build_retriever, FullContextRetriever, BM25Retriever,
)


class FakeEmbedder:
    """
    Deterministic 'embeddings' for tests: each text is represented by the
    normalised count of a few marker words. No model download required.
    """
    VOCAB = ["кофе", "капучино", "цена", "стоит", "улун", "чай"]

    def encode(self, texts, is_query=False):
        vecs = []
        for t in texts:
            low = t.lower()
            v = np.array([low.count(w) for w in self.VOCAB], dtype=float)
            # semantic trick: "стоит"/"цена" and "кофе"/"капучино" reinforce each other
            if "стоит" in low or "цена" in low:
                v[2] += 1; v[3] += 1
            n = np.linalg.norm(v)
            vecs.append(v / n if n else v)
        return np.array(vecs)


def _corpus():
    a = "Цена капучино составляет 1100 тенге."
    b = "Улун заваривают водой 85 градусов."
    return chunk_document(a, "menu.txt", 200, 20) + chunk_document(b, "tea.txt", 200, 20)


def test_reciprocal_rank_fusion_merges():
    # item 2 is top of one list and near-top of the other -> should win
    fused = reciprocal_rank_fusion([2, 0, 1], [2, 1, 0])
    assert fused[0] == 2


def test_embeddings_retriever_finds_by_meaning():
    chunks = _corpus()
    r = EmbeddingsRetriever(chunks, FakeEmbedder())
    # "сколько стоит кофе" shares no words with "Цена капучино" but is closer semantically
    picked = r.select("сколько стоит кофе", char_budget=500)
    assert picked[0]["doc"] == "menu.txt"


def test_hybrid_retriever_runs_and_ranks():
    chunks = _corpus()
    r = HybridRetriever(chunks, FakeEmbedder())
    picked = r.select("цена кофе", char_budget=500)
    assert picked[0]["doc"] == "menu.txt"


def test_build_retriever_semantic_and_hybrid():
    chunks = _corpus()
    emb = FakeEmbedder()
    assert isinstance(build_retriever(chunks, False, "semantic", emb), EmbeddingsRetriever)
    assert isinstance(build_retriever(chunks, False, "hybrid", emb), HybridRetriever)
    # no embedder -> falls back to BM25
    assert isinstance(build_retriever(chunks, False, "hybrid", None), BM25Retriever)
    # full mode always wins
    assert isinstance(build_retriever(chunks, True, "hybrid", emb), FullContextRetriever)