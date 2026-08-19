# ask-docs

Chat with your documents. Load one or more files (PDF, DOCX, TXT, MD) and ask
questions in plain language. Answers are grounded strictly in the document text,
and each answer shows the exact fragments it was built from. If the answer isn't
in the documents, the app says so instead of guessing.

Built with Python, Streamlit and the Claude API. Retrieval runs locally: a BM25
ranker by default, with optional semantic and hybrid search via local embeddings.

## Features

Multiple documents at once, answered as a single set.

Grounded answers with a strict prompt that stops the model from inventing facts.

Exact citations: every answer lists the specific chunks used, not just filenames.

Three retrieval modes: lexical (BM25), semantic (embeddings), and hybrid (both
combined with Reciprocal Rank Fusion). BM25 needs no extra setup. Semantic and
hybrid use a local multilingual model that works for Russian and Kazakh; if the
embedding model can't load, the app falls back to BM25 automatically.

Small documents are sent to the model in full; larger ones are chunked and
retrieved, which keeps requests inside the context window and cheaper.

Streaming responses, chat export, a per-request cost estimate, and prompt
caching so follow-up questions on the same documents cost less.

## Project layout

    config.py       Settings loaded from .env; models and parameters.
    extract.py      Text extraction from PDF, DOCX, TXT, MD.
    retrieval.py    Chunking and pluggable retrievers (BM25 / semantic / hybrid).
    embeddings.py   Local sentence embeddings, used by semantic and hybrid modes.
    app.py          Streamlit interface and Claude API calls.
    tests/          Pytest suite.

Retrieval sits behind a small `Retriever` interface, so a new retriever can be
added without touching the interface or the prompt code. The embedder is passed
in, so `retrieval.py` never imports torch and stays quick to test.

## Setup

Install the core dependencies:

    pip install -r requirements.txt

If the `pip` command isn't found on Windows, use `python -m pip` instead.

Get an API key from the Anthropic console and add credit to the account. Copy
the settings template and add your key:

    copy .env.example .env      # Windows
    cp .env.example .env        # macOS / Linux

Open `.env` and set `ANTHROPIC_API_KEY`. The file stays on your machine and is
ignored by git.

Run the app:

    streamlit run app.py

It opens in the browser at http://localhost:8501.

## Semantic and hybrid search (optional)

These modes use local embeddings and need extra packages plus a fair amount of
RAM. Install PyTorch matched to your hardware first:

    pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
    pip install torch                                                      # CPU only

Then the embedding library:

    pip install -r requirements-embeddings.txt

Set `RETRIEVAL_MODE` in `.env` to `bm25`, `semantic`, or `hybrid`. The first run
downloads the embedding model once. On machines with limited memory, keep
`RETRIEVAL_MODE=bm25` — the app runs the same, just without semantic search.

## Configuration

All values live in `.env`:

    ANTHROPIC_API_KEY        API key (required)
    DEFAULT_MODEL            haiku or sonnet
    MAX_TOKENS               max answer length
    FULL_MODE_TOKEN_LIMIT    send full text below this size, retrieve above it
    RETRIEVAL_CHAR_BUDGET    context budget when retrieving
    CHUNK_CHARS              chunk size
    CHUNK_OVERLAP            overlap between chunks
    RETRIEVAL_MODE           bm25 | semantic | hybrid
    EMBEDDING_MODEL          embedding model name
    EMBEDDING_DEVICE         empty for auto, or cpu / cuda

## Tests

    pip install -r requirements-dev.txt
    pytest

The suite covers text extraction, chunking, BM25 ranking, semantic and hybrid
retrieval, and configuration loading.

## Tech stack

Python, Streamlit, Anthropic Claude API, pypdf and python-docx for parsing,
sentence-transformers for optional local embeddings, and a BM25 ranker written
from scratch.

## Limitations

Scanned or image-only PDFs aren't read; that needs OCR.

BM25 matches on words, so it can miss questions phrased with different wording.
Semantic and hybrid modes handle that, but need more memory to run locally.

## License

MIT. See LICENSE.