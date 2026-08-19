"""
retrieval.py - pluggable retrieval of relevant chunks.

Pipeline:
- documents are split into overlapping chunks;
- a Retriever picks the chunks most relevant to a question within a budget;
- the picked chunks are formatted into the model context, with source labels.

Two retrievers ship here:
- FullContextRetriever - returns everything (used when documents are small);
- BM25Retriever - lexical ranking (used when documents are large).

The Retriever interface is deliberately small so a semantic (embeddings-based)
retriever can be dropped in later without touching the rest of the app.
"""

import re
import math
from abc import ABC, abstractmethod

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def chunk_document(text: str, doc_name: str,
                   chunk_chars: int | None = None, overlap: int | None = None) -> list[dict]:
    """
    Split a document into overlapping chunks by paragraphs.

    Each chunk is a dict: {"id": "<doc>#<n>", "doc": <doc>, "text": <str>}.
    The id makes exact citation possible (we can point at a specific chunk).
    """
    from config import settings
    chunk_chars = settings.chunk_chars if chunk_chars is None else chunk_chars
    overlap = settings.chunk_overlap if overlap is None else overlap

    paras = [p.strip() for p in text.split("\n") if p.strip()]
    raw_chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 1 > chunk_chars:
            raw_chunks.append(buf.strip())
            buf = buf[-overlap:] + "\n" + p          # keep a tail of the previous chunk for context
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf.strip():
        raw_chunks.append(buf.strip())

    return [
        {"id": f"{doc_name}#{i}", "doc": doc_name, "text": c}
        for i, c in enumerate(raw_chunks)
    ]


# BM25 core
class BM25:
    """Compact BM25 implementation for ranking chunks."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.corpus = [tokenize(c["text"]) for c in chunks]
        self.N = len(self.corpus)
        self.avgdl = (sum(len(d) for d in self.corpus) / self.N) if self.N else 0.0
        self.k1, self.b = k1, b
        self.df: dict[str, int] = {}
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query_terms: list[str], idx: int) -> float:
        doc = self.corpus[idx]
        if not doc or self.avgdl == 0:
            return 0.0
        freq: dict[str, int] = {}
        for t in doc:
            freq[t] = freq.get(t, 0) + 1
        dl = len(doc)
        s = 0.0
        for t in query_terms:
            if t not in freq:
                continue
            f = freq[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self._idf(t) * (f * (self.k1 + 1)) / denom
        return s

    def rank(self, query: str) -> list[tuple[float, int]]:
        """Return (score, chunk_index) pairs sorted by descending score."""
        q = tokenize(query)
        scored = [(self.score(q, i), i) for i in range(self.N)]
        scored.sort(reverse=True)
        return scored


# Retrievers 
class Retriever(ABC):
    """Selects the chunks most relevant to a question, within a char budget."""

    @abstractmethod
    def select(self, question: str, char_budget: int) -> list[dict]:
        ...


class FullContextRetriever(Retriever):
    """Returns all chunks — used when the whole corpus fits the context window."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

    def select(self, question: str, char_budget: int) -> list[dict]:
        return self.chunks


class BM25Retriever(Retriever):
    """Lexical retriever: ranks chunks with BM25 and fills the char budget."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25(chunks)

    def select(self, question: str, char_budget: int) -> list[dict]:
        ordered = [self.chunks[i] for s, i in self.bm25.rank(question) if s > 0]
        if not ordered:                      # no lexical matches — fall back to the beginning
            ordered = self.chunks
        return _fill_budget(ordered, char_budget)


def build_retriever(chunks: list[dict], full_mode: bool,
                    mode: str = "bm25", embedder=None) -> Retriever:
    """
    Factory: choose a retriever.

    - full_mode: everything fits the context window -> send it all.
    - otherwise pick by `mode`: "semantic" / "hybrid" need an embedder;
      anything else (or a missing embedder) falls back to BM25.
    """
    if full_mode:
        return FullContextRetriever(chunks)
    if embedder is not None and mode == "semantic":
        return EmbeddingsRetriever(chunks, embedder)
    if embedder is not None and mode == "hybrid":
        return HybridRetriever(chunks, embedder)
    return BM25Retriever(chunks)


#  Semantic / hybrid 
class EmbeddingsRetriever(Retriever):
    """
    Semantic retriever. Embeds every chunk once, then ranks chunks by cosine
    similarity to the question's embedding.

    `embedder` is injected (see embeddings.py) so this module never imports
    torch directly and stays easy to test with a fake embedder.
    """

    def __init__(self, chunks: list[dict], embedder):
        import numpy as np
        self.chunks = chunks
        self.embedder = embedder
        # precompute chunk vectors once (passages)
        texts = [c["text"] for c in chunks]
        self.matrix = embedder.encode(texts, is_query=False) if texts else np.zeros((0, 1))

    def rank(self, question: str) -> list[int]:
        """Return chunk indices ordered by descending semantic similarity."""
        import numpy as np
        if len(self.chunks) == 0:
            return []
        qvec = self.embedder.encode([question], is_query=True)[0]
        sims = self.matrix @ qvec            # cosine sim (vectors are normalised)
        return list(np.argsort(-sims))

    def select(self, question: str, char_budget: int) -> list[dict]:
        order = self.rank(question)
        return _fill_budget([self.chunks[i] for i in order], char_budget)


def reciprocal_rank_fusion(*rankings: list[int], k: int = 60) -> list[int]:
    """
    Combine several ranked lists of chunk indices into one.

    RRF gives each item a score of sum(1 / (k + rank)) across the lists it
    appears in, then sorts by that score. It's the standard way to merge
    lexical (BM25) and semantic rankings without normalising their raw scores.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class HybridRetriever(Retriever):
    """Fuses BM25 (lexical) and embeddings (semantic) rankings via RRF."""

    def __init__(self, chunks: list[dict], embedder):
        self.chunks = chunks
        self.bm25 = BM25(chunks)
        self.semantic = EmbeddingsRetriever(chunks, embedder)

    def select(self, question: str, char_budget: int) -> list[dict]:
        lexical = [i for s, i in self.bm25.rank(question) if s > 0]
        semantic = self.semantic.rank(question)
        fused = reciprocal_rank_fusion(lexical, semantic)
        if not fused:                        # nothing matched lexically or semantically
            fused = list(range(len(self.chunks)))
        return _fill_budget([self.chunks[i] for i in fused], char_budget)


def _fill_budget(ordered_chunks: list[dict], char_budget: int) -> list[dict]:
    """Take chunks in order until the character budget is filled."""
    selected, used = [], 0
    for chunk in ordered_chunks:
        if used + len(chunk["text"]) > char_budget and selected:
            break
        selected.append(chunk)
        used += len(chunk["text"])
    return selected


# ----------------------------- Formatting -----------------------------
def format_context(picked: list[dict]) -> str:
    """Turn picked chunks into labelled context text for the system prompt."""
    parts = [f"[Источник: {c['doc']}]\n{c['text']}" for c in picked]
    return "\n\n---\n\n".join(parts)


def unique_sources(picked: list[dict]) -> list[str]:
    """Distinct document names among the picked chunks, order-preserving."""
    seen: list[str] = []
    for c in picked:
        if c["doc"] not in seen:
            seen.append(c["doc"])
    return seen


def snippet(text: str, limit: int = 220) -> str:
    """Short preview of a chunk for citation display."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"