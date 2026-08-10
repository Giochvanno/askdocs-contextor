"""
retrieval.py — lightweight retrieval of relevant chunks without a vector DB.

Logic:
- documents are split into chunks;
- if everything fits the budget, we send all of it;
- otherwise BM25 selects the chunks most relevant to the question.

This is a "mini-RAG": it scales to large / multiple documents,
but without embeddings or external dependencies.
"""

import re
import math

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def chunk_document(text: str, doc_name: str,
                   chunk_chars: int | None = None, overlap: int | None = None) -> list[dict]:
    """Split a document into overlapping chunks by paragraphs."""
    from config import settings
    chunk_chars = settings.chunk_chars if chunk_chars is None else chunk_chars
    overlap = settings.chunk_overlap if overlap is None else overlap

    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 1 > chunk_chars:
            chunks.append(buf.strip())
            buf = buf[-overlap:] + "\n" + p          # keep a tail of the previous chunk for context
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf.strip():
        chunks.append(buf.strip())
    return [{"doc": doc_name, "text": c} for c in chunks]


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

    def _score(self, query_terms: list[str], idx: int) -> float:
        doc = self.corpus[idx]
        if not doc or self.avgdl == 0:
            return 0.0
        freq: dict[str, int] = {}
        for t in doc:
            freq[t] = freq.get(t, 0) + 1
        dl = len(doc)
        score = 0.0
        for t in query_terms:
            if t not in freq:
                continue
            f = freq[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self._idf(t) * (f * (self.k1 + 1)) / denom
        return score

    def top(self, query: str, char_budget: int) -> list[dict]:
        """Return the best chunks until the character budget is filled."""
        q = tokenize(query)
        scored = [(self._score(q, i), i) for i in range(self.N)]
        scored.sort(reverse=True)
        selected, used = [], 0
        for s, i in scored:
            if s <= 0:
                continue
            chunk = self.chunks[i]
            if used + len(chunk["text"]) > char_budget and selected:
                break
            selected.append(chunk)
            used += len(chunk["text"])
        return selected


def select_context(chunks: list[dict], bm25: "BM25 | None",
                   question: str, char_budget: int) -> tuple[str, list[str]]:
    """
    Return (context_text, list_of_sources).
    If bm25 is None — full-context mode (everything fits).
    """
    if bm25 is None:
        picked = chunks
    else:
        picked = bm25.top(question, char_budget)
        if not picked:                       # no matches for the question — take the beginning
            picked = chunks[: max(1, char_budget // 2500)]

    parts, sources = [], []
    for c in picked:
        parts.append(f"[Источник: {c['doc']}]\n{c['text']}")
        if c["doc"] not in sources:
            sources.append(c["doc"])
    return "\n\n---\n\n".join(parts), sources