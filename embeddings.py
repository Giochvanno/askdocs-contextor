"""
embeddings.py — local sentence embeddings via sentence-transformers.

Runs a multilingual model (good for Russian/Kazakh) on the GPU when available,
falling back to CPU automatically. The model is loaded lazily and cached, so it
is downloaded/initialised only once per process.

This module is optional: it's only imported when semantic/hybrid retrieval is
enabled, so the app still runs without torch installed (BM25-only mode).
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """Wraps a sentence-transformers model and returns L2-normalised vectors."""

    def __init__(self, model_name: str, device: str | None = None):
        # Imported here (not at top) so the app works without torch when
        # semantic retrieval is turned off.
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """
        Encode a list of texts into a matrix of normalised vectors.

        The e5 family expects a task prefix: "query: ..." for questions and
        "passage: ..." for documents. We add it automatically.
        """
        prefix = "query: " if is_query else "passage: "
        prepared = [prefix + t for t in texts]
        vecs = self.model.encode(
            prepared,
            convert_to_numpy=True,
            normalize_embeddings=True,   # so dot product == cosine similarity
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs


# process-wide cache so we don't reload the model on every rerun
_EMBEDDER: Embedder | None = None


def get_embedder(model_name: str, device: str | None = None) -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None or _EMBEDDER.model_name != model_name:
        _EMBEDDER = Embedder(model_name, device)
    return _EMBEDDER