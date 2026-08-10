"""
app.py — "Chat with documents" using the Claude API + Streamlit.

Settings come from config.py (which reads .env). No secrets in this file.

Run:
    streamlit run app.py
"""

import datetime

import streamlit as st
import anthropic

from config import settings, MODELS
from extract import extract_text, estimate_tokens
from retrieval import chunk_document, BM25, select_context


# System prompt: force the model to answer strictly from the provided document text.
INSTRUCTIONS = (
    "Ты помощник, который отвечает на вопросы СТРОГО по фрагментам документов ниже.\n"
    "Правила:\n"
    "- Отвечай только на основе предоставленного текста, ничего не выдумывай.\n"
    "- Если ответа в тексте нет — честно скажи: «В документах нет ответа на этот вопрос».\n"
    "- Отвечай на языке вопроса.\n"
    "- Указывай источник ответа — имя файла из пометки [Источник: ...]."
)

st.set_page_config(page_title="Чат с документами", page_icon="📄", layout="centered")


@st.cache_resource
def get_client():
    return anthropic.Anthropic(api_key=settings.api_key) if settings.has_api_key() else None


client = get_client()


# state
def init_state():
    st.session_state.setdefault("documents", {})   # {filename: text}
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("bm25", None)
    st.session_state.setdefault("full_mode", True)
    st.session_state.setdefault("messages", [])


init_state()


def rebuild_index():
    """Rebuild chunks and the BM25 index for the current set of documents."""
    chunks = []
    for name, text in st.session_state.documents.items():
        chunks += chunk_document(text, name)
    st.session_state.chunks = chunks

    total_tokens = sum(estimate_tokens(t) for t in st.session_state.documents.values())
    st.session_state.full_mode = total_tokens <= settings.full_mode_token_limit
    st.session_state.bm25 = None if st.session_state.full_mode else BM25(chunks)


# sidebar 
with st.sidebar:
    st.header(" Настройки")

    model_labels = {cfg.label: key for key, cfg in MODELS.items()}
    default_label = MODELS[settings.default_model].label
    chosen_label = st.selectbox(
        "Модель", list(model_labels.keys()),
        index=list(model_labels.keys()).index(default_label),
    )
    model_cfg = MODELS[model_labels[chosen_label]]

    st.divider()
    uploaded = st.file_uploader(
        "Документы", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )

    if uploaded:
        current = {f.name for f in uploaded}
        changed = False
        # add newly uploaded files
        for f in uploaded:
            if f.name not in st.session_state.documents:
                try:
                    text = extract_text(f, f.name)
                    if text.strip():
                        st.session_state.documents[f.name] = text
                        changed = True
                    else:
                        st.error(f"«{f.name}»: текст не извлёкся (возможно, скан — нужен OCR).")
                except Exception as e:  # noqa: BLE001
                    st.error(f"«{f.name}»: {e}")
        # remove files the user dropped from the uploader
        for name in list(st.session_state.documents.keys()):
            if name not in current:
                del st.session_state.documents[name]
                changed = True
        if changed:
            rebuild_index()
            st.session_state.messages = []

    if st.session_state.documents:
        st.divider()
        total_tokens = sum(estimate_tokens(t) for t in st.session_state.documents.values())
        st.caption(f" Документов: {len(st.session_state.documents)}")
        st.caption(f"~{total_tokens:,} токенов · чанков: {len(st.session_state.chunks)}")
        mode = "весь текст в контексте" if st.session_state.full_mode else "поиск релевантных кусков (BM25)"
        st.caption(f"Режим: {mode}")

        if st.session_state.messages:
            transcript = "\n\n".join(
                f"**{m['role']}:** {m['content']}" for m in st.session_state.messages
            )
            st.download_button(
                "⬇ Экспорт диалога", data=transcript,
                file_name=f"chat_{datetime.date.today()}.md", mime="text/markdown",
            )
        if st.button("🗑 Очистить диалог"):
            st.session_state.messages = []
            st.rerun()


# main screen 
st.title("📄 Чат с документами")

if client is None:
    st.error(
        "Не найден ANTHROPIC_API_KEY. Создайте файл .env из .env.example и впишите ключ, "
        "затем перезапустите.\n\nКлюч: https://console.anthropic.com/"
    )
    st.stop()

if not st.session_state.documents:
    st.info(" Загрузите один или несколько документов в панели слева, чтобы начать.")
    st.stop()

# render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📎 Источники: " + ", ".join(msg["sources"]))


def build_system(question: str):
    """Build the system prompt: instructions + selected document context (cached)."""
    context, sources = select_context(
        st.session_state.chunks, st.session_state.bm25,
        question, settings.retrieval_char_budget,
    )
    system = [
        {"type": "text", "text": INSTRUCTIONS},
        {
            "type": "text",
            "text": f"=== ФРАГМЕНТЫ ДОКУМЕНТОВ ===\n{context}\n=== КОНЕЦ ===",
            "cache_control": {"type": "ephemeral"},  # cache the context -> cheaper follow-up questions
        },
    ]
    return system, sources, context


# user input
if prompt := st.chat_input("Спросите что-нибудь по документам…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            system, sources, context = build_system(prompt)
            # API history without our internal "sources" field
            api_messages = [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ]

            def stream_text():
                with client.messages.stream(
                    model=model_cfg.id,
                    max_tokens=settings.max_tokens,
                    system=system,
                    messages=api_messages,
                ) as stream:
                    for chunk in stream.text_stream:
                        yield chunk

            answer = st.write_stream(stream_text)
            if sources:
                st.caption("📎 Источники: " + ", ".join(sources))

            # rough cost estimate for this request
            in_tok = estimate_tokens(context) + estimate_tokens(prompt) + 200
            out_tok = estimate_tokens(answer)
            cost = in_tok / 1e6 * model_cfg.price_in + out_tok / 1e6 * model_cfg.price_out
            st.caption(f"≈ {in_tok:,} вх. / {out_tok:,} исх. токенов · ~${cost:.4f}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except anthropic.APIError as e:
            st.error(f"Ошибка API: {e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Ошибка: {e}")