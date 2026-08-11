# Chat with Documents (mini-RAG)

Ask questions about your documents (PDF / DOCX / TXT / MD) in plain language.
The model answers **strictly from the document content** and cites which file the
answer came from. If the answer isn't in the documents, it says so instead of
making something up.

Built with Python, Streamlit and the Claude API, with a custom BM25 retrieval
layer so it scales to large or multiple documents without a vector database.

---

##  Features

- **Multiple documents at once** — questions are answered across the whole set.
- **Smart context selection** — small documents are sent in full; large ones go
  through a BM25 retriever that picks only the chunks relevant to the question.
  This keeps requests cheap and within the model's context window.
- **Source attribution** — every answer shows which file(s) it was built from.
- **Grounded answers** — a strict system prompt prevents the model from inventing
  facts that aren't in the documents.
- **Streaming responses**, **chat export**, and a **per-request cost estimate**.
- **Prompt caching** (`cache_control`) — follow-up questions on the same documents
  are cheaper.

---

##  Architecture

```
config.py       Single source of truth. Reads .env, defines models and parameters.
extract.py      Text extraction from PDF / DOCX / TXT / MD.
retrieval.py    Chunking + BM25 ranking (relevant-chunk selection).
app.py          Streamlit UI + Claude API calls.
.env            Your secrets (created from .env.example, never committed).
.env.example    Settings template.
.gitignore      Keeps .env and build artifacts out of git.
```

**Design principle:** secrets live only in `.env`; the code reads them through
`config.py`. The API key is never hardcoded.

**Data flow:** `document → extract text → split into chunks → (BM25 select /
send all) → build prompt → Claude API → grounded answer with sources`.

---

##  Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
# If 'pip' is not found (common on Windows):
python -m pip install -r requirements.txt
```

### 2. Configure your API key

Get a key at https://console.anthropic.com/ (API Keys section) and add credit
to your account.

```bash
# macOS / Linux
cp .env.example .env
# Windows
copy .env.example .env
```

Open `.env` and set your key:

```
ANTHROPIC_API_KEY=sk-ant-your_key_here
```

`.env` stays on your machine and is excluded from git via `.gitignore`.

### 3. Run

```bash
streamlit run app.py
# or, if streamlit is not on PATH:
python -m streamlit run app.py
```

The app opens in your browser (usually http://localhost:8501).

---

##  How It Works

- If all documents together are small (under ~30k tokens), the full text is sent
  to the model.
- For larger sets, a **BM25 retriever** ranks chunks by relevance to the question
  and sends only the top ones — the same idea behind search engines, implemented
  from scratch in `retrieval.py` with no external search dependency.
- The selected context is placed in the system prompt and cached, so repeated
  questions about the same documents cost less.

---

##  Configuration

All settings can be overridden in `.env`:

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | API key (required) | — |
| `DEFAULT_MODEL` | `haiku` or `sonnet` | `haiku` |
| `MAX_TOKENS` | max answer length | `1024` |
| `FULL_MODE_TOKEN_LIMIT` | threshold to send the full text | `30000` |
| `RETRIEVAL_CHAR_BUDGET` | context budget when retrieving | `48000` |
| `CHUNK_CHARS` / `CHUNK_OVERLAP` | chunk size and overlap | `2500` / `250` |

---

##  Tech Stack

- **Python 3.12**
- **Streamlit** — web UI
- **Anthropic Claude API** — answer generation
- **pypdf**, **python-docx** — document parsing
- **python-dotenv** — environment configuration
- Custom **BM25** retriever (no external vector DB)

---

##  Limitations & Roadmap

- **Scanned / image PDFs** aren't read — OCR (`pytesseract` + `tesseract`) is a
  planned addition.
- **Lexical vs. semantic search** — BM25 matches on words. The next step is
  embeddings + a vector store for semantic retrieval (matching by meaning, not
  just wording). The architecture is ready for this: only the retriever in
  `retrieval.py` needs to be swapped, the rest stays the same.
- Other planned items: exact-sentence citations next to answers, and persisting
  the index between runs.

---

##  License

MIT — free to use, modify and learn from.