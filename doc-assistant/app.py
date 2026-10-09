"""Streamlit UI for the document assistant.

Visual system matches the Fastshot hero: full-bleed video, glass panels,
Inter, and an orange send control. Retrieval and the model call live in rag.py.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

import rag

VIDEO_URL = (
    "https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/"
    "hf_20260826_124724_bc041163-d651-425f-aea3-2acc1efc2c96.mp4"
)

PAGE_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,100..900&display=block">
<style>
  html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #0a0d12 !important;
    color: #f4f4f5;
    font-family: Inter, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-synthesis: none;
  }
  header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer {
    display: none !important;
  }
  .fs-video, .fs-scrim {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }
  .fs-video { object-fit: cover; z-index: 0; }
  .fs-scrim {
    z-index: 0;
    background:
      linear-gradient(180deg, rgba(10,13,18,.55) 0%, rgba(10,13,18,.12) 38%, rgba(10,13,18,.62) 100%);
  }
  [data-testid="stAppViewContainer"] > .main,
  [data-testid="stBottom"] { position: relative; z-index: 1; }
  [data-testid="stBottom"] { background: transparent; }
  section[data-testid="stSidebar"] {
    z-index: 2;
    background: rgba(12, 12, 16, 0.82) !important;
    border-right: 1px solid rgba(255,255,255,.08);
    backdrop-filter: blur(22px) saturate(120%);
  }
  section[data-testid="stSidebar"] > div { padding-top: 1.4rem; }
  [data-testid="stSidebar"] * { color: #f3f3f4; }
  .block-container {
    padding-top: 1.6rem;
    max-width: 860px;
  }
  .fs-brand { display: flex; align-items: center; gap: 12px; }
  .fs-word {
    color: #fff;
    font-weight: 500;
    font-size: 1.15rem;
    letter-spacing: -0.015em;
    text-shadow: 0 1px 10px rgba(0,0,0,.35);
  }
  h1.fs-title {
    color: #fff;
    font-weight: 410;
    letter-spacing: 0.0018em;
    font-size: clamp(2rem, 4vw, 2.7rem);
    line-height: 1.08;
    text-shadow: 0 2px 22px rgba(0,0,0,.35);
    margin: 1.1rem 0 0.45rem;
  }
  .fs-sub {
    color: rgba(255,255,255,.9);
    font-size: 1.02rem;
    line-height: 1.45;
    max-width: 38rem;
    margin: 0 0 1.1rem;
    text-shadow: 0 1px 12px rgba(0,0,0,.35);
  }
  .fs-empty {
    background: rgba(41, 41, 43, 0.78);
    border: 1px solid rgba(214, 228, 255, 0.16);
    border-radius: 22px;
    box-shadow: 0 22px 60px rgba(0,0,0,.28);
    backdrop-filter: blur(18px) saturate(120%);
    padding: 1.15rem 1.25rem 1.2rem;
    margin: 0.4rem 0 1rem;
  }
  .fs-empty strong { color: #fff; font-weight: 500; }
  .fs-empty p { color: #c8c8cb; margin: 0.35rem 0 0.8rem; }
  .fs-steps { display: flex; gap: 0.7rem; flex-wrap: wrap; }
  .fs-step {
    flex: 1 1 140px;
    background: rgba(255,255,255,.05);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px;
    padding: 0.75rem 0.8rem;
    color: #d7d7da;
    font-size: 0.88rem;
    line-height: 1.35;
  }
  .fs-step b { display: block; color: #fff; font-weight: 500; margin-bottom: 0.2rem; }
  .doc-card {
    background: rgba(255,255,255,.05);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 12px;
    padding: 0.55rem 0.7rem 0.15rem;
    margin-top: 0.55rem;
  }
  .doc-name { display: block; font-weight: 500; color: #fff; word-break: break-word; }
  .doc-meta { display: block; color: #a3a3a8; font-size: 0.78rem; margin-top: 0.1rem; }
  .side-label {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9a9aa0;
    margin: 0.2rem 0 0.35rem;
  }
  .snip {
    background: rgba(255,255,255,.04);
    border-left: 2px solid #E88654;
    border-radius: 8px;
    padding: 0.55rem 0.7rem;
    margin-bottom: 0.55rem;
  }
  .snip-meta { color: #F4B08A; font-size: 0.78rem; margin-bottom: 0.25rem; }
  .snip p { margin: 0; color: #ececee; line-height: 1.45; }
  .banner {
    background: rgba(41, 41, 43, 0.9);
    border: 1px solid rgba(248, 178, 133, 0.45);
    color: #fff;
    border-radius: 14px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.8rem;
  }
  [data-testid="stChatMessage"] {
    background: rgba(28, 28, 32, 0.82);
    border: 1px solid rgba(214, 228, 255, 0.12);
    border-radius: 18px;
    backdrop-filter: blur(16px);
    padding: 0.15rem 0.2rem;
  }
  [data-testid="stChatMessageAvatar"] { display: none; }
  [data-testid="stChatInput"] {
    background: rgba(41, 41, 43, 0.96);
    border-radius: 26px;
    border: 1px solid rgba(214, 228, 255, 0.16) !important;
    box-shadow: 0 18px 50px rgba(0,0,0,.35);
  }
  [data-testid="stChatInput"] textarea {
    color: #f5f5f5;
    caret-color: #F49D70;
    font-size: 1rem;
  }
  [data-testid="stChatInput"] button {
    background: linear-gradient(163deg, #FBBC94 0%, #F49D70 46%, #E88654 100%) !important;
    color: #fff !important;
    border: 0 !important;
    border-radius: 50% !important;
    width: 36px !important;
    height: 36px !important;
    box-shadow: 0 3px 12px rgba(210,110,60,.34);
  }
  [data-testid="stChatInput"] button:hover { filter: brightness(1.07); }
  .stButton > button, [data-testid="stBaseButton-secondary"] {
    border-radius: 10px;
    background: linear-gradient(180deg, rgba(255,255,255,.1), rgba(255,255,255,.04));
    color: #d0d0d3;
    border: 1px solid rgba(255,255,255,.1);
    font-weight: 500;
  }
  .stButton > button:hover { color: #fff; border-color: rgba(255,255,255,.22); }
  [data-testid="stBaseButton-primary"] {
    background: linear-gradient(163deg, #FBBC94 0%, #E88654 100%) !important;
    color: #1a120e !important;
    border: 0 !important;
  }
  [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.04);
    border: 1px dashed rgba(255,255,255,.2);
    border-radius: 14px;
  }
  [data-testid="stExpander"] {
    background: rgba(16, 16, 20, 0.55);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 12px;
  }
  [data-testid="stSlider"] [role="slider"] { background: #E88654; }
  hr { border-color: rgba(255,255,255,.08) !important; }
</style>
<video class="fs-video" autoplay muted loop playsinline src="__VIDEO__"></video>
<div class="fs-scrim"></div>
""".replace("__VIDEO__", VIDEO_URL)

BRAND = """
<div class="fs-brand">
  <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true">
    <circle cx="17" cy="17" r="17" fill="#9C86CE"/>
    <circle cx="17" cy="17" r="8.6" fill="#FFFFFF"/>
    <circle cx="17" cy="17" r="3.7" fill="#151519"/>
  </svg>
  <span class="fs-word">Fastshot</span>
</div>
<h1 class="fs-title">Ask your notes. Check the source.</h1>
<p class="fs-sub">Answers come only from the lecture files you upload. Every reply shows the snippets it used.</p>
"""

EMPTY = """
<div class="fs-empty">
  <strong>Start with a lecture file</strong>
  <p>Drop a PDF or a text note in the sidebar. Questions stay inside those pages.</p>
  <div class="fs-steps">
    <div class="fs-step"><b>1. Upload</b>PDF or TXT. Image-only scans are skipped.</div>
    <div class="fs-step"><b>2. Ask</b>Your own words. Meaning, not exact phrases.</div>
    <div class="fs-step"><b>3. Check</b>Read the snippets under each answer.</div>
  </div>
</div>
"""


def get_model() -> Any:
    """Load the embedding model once per server process."""

    return rag.load_embedding_model()


@st.cache_resource(show_spinner="Loading AI model...")
def cached_model() -> Any:
    """Streamlit cache around the local embedding model."""

    return get_model()


def _init_state() -> None:
    """Create the in-memory index and chat. Nothing is written to disk."""

    if "index" not in st.session_state:
        st.session_state.index = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "seen" not in st.session_state:
        st.session_state.seen = set()
    if "banner" not in st.session_state:
        st.session_state.banner = ""
    if "job" not in st.session_state:
        st.session_state.job = None


def _render_snippets(snippets: list[dict[str, Any]]) -> None:
    """Show the chunks that grounded one answer."""

    if not snippets:
        return
    with st.expander("Referenced snippets"):
        for snippet in snippets:
            st.markdown(
                (
                    '<div class="snip">'
                    f'<div class="snip-meta">{html.escape(str(snippet["source"]))} · '
                    f'page {snippet["page"]} · similarity {snippet["score"]:.2f}</div>'
                    f'<p>{html.escape(str(snippet["text"]))}</p>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def _render_history() -> None:
    """Draw the chat stored in the session."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_snippets(message.get("snippets") or [])


def _ingest(uploaded: Any, model: Any) -> None:
    """Index new uploads and skip files already in the session."""

    for item in uploaded:
        key = (item.name, item.size)
        if item.name in st.session_state.index or key in st.session_state.seen:
            if item.name in st.session_state.index and key not in st.session_state.seen:
                st.sidebar.caption(f"Skipped duplicate: {item.name}")
            st.session_state.seen.add(key)
            continue
        data = item.getvalue()
        with st.spinner(f"Reading {item.name}..."):
            record, warning = rag.index_bytes(item.name, data, model)
        st.session_state.seen.add(key)
        if warning:
            st.sidebar.warning(warning)
        if record is None:
            continue
        st.session_state.index[item.name] = record
        count = len(record["chunks"])
        st.sidebar.success(f"{count} chunks indexed")


def _delete(name: str) -> None:
    """Drop one file and its chunks from the index."""

    st.session_state.index.pop(name, None)
    st.session_state.seen = {key for key in st.session_state.seen if key[0] != name}


def _stream_into_chat(messages: list[dict[str, str]], snippets: list[dict[str, Any]]) -> None:
    """Stream a reply, then store it with the snippets used."""

    with st.chat_message("assistant"):
        try:
            tokens = rag.generate_text(messages, stream=True)
            text = st.write_stream(tokens)
        except rag.AssistantError as exc:
            st.session_state.banner = str(exc)
            return
    st.session_state.messages.append(
        {"role": "assistant", "content": text or "", "snippets": snippets}
    )


def _run_job(model: Any, k: int) -> None:
    """Run the queued question, summary, or quiz, then clear the queue."""

    job = st.session_state.job
    st.session_state.job = None
    if not job:
        return
    kind = job["kind"]
    try:
        if kind == "ask":
            question = job["payload"]
            history = st.session_state.messages[:-1]
            search_query = rag.rewrite_query(question, history)
            hits = rag.search_index(model, search_query, st.session_state.index, k=k)
            snippets = rag.snippet_payload(hits)
            _stream_into_chat(rag.answer_messages(question, hits, history), snippets)
        elif kind == "summarize":
            name = job["payload"]
            record = st.session_state.index.get(name)
            if not record:
                st.session_state.banner = "That file is no longer in the index."
                return
            st.session_state.messages.append(
                {"role": "user", "content": f"Summarize {name}.", "snippets": []}
            )
            _stream_into_chat(rag.summary_messages(name, record["chunks"]), [])
        elif kind == "quiz":
            if not st.session_state.index:
                st.session_state.banner = (
                    "Upload a lecture PDF or a text note first. "
                    "Answers come only from your files."
                )
                return
            st.session_state.messages.append(
                {"role": "user", "content": "Generate 5 quiz questions from my notes.", "snippets": []}
            )
            _stream_into_chat(rag.quiz_messages(st.session_state.index), [])
    except rag.AssistantError as exc:
        st.session_state.banner = str(exc)


def _sidebar(model: Any) -> int:
    """Document list, upload control, and the retrieval slider."""

    st.sidebar.markdown('<p class="side-label">Library</p>', unsafe_allow_html=True)
    uploaded = st.sidebar.file_uploader(
        "Upload lecture notes",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        _ingest(uploaded, model)

    index = st.session_state.index
    total = sum(len(record["chunks"]) for record in index.values())
    st.sidebar.caption(f"{len(index)} files · {total} chunks in memory")

    for name in list(index):
        count = len(index[name]["chunks"])
        st.sidebar.markdown(
            (
                '<div class="doc-card">'
                f'<span class="doc-name">{html.escape(name)}</span>'
                f'<span class="doc-meta">{count} chunks</span>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        left, right = st.sidebar.columns(2)
        if left.button("Summarize", key=f"sum-{name}"):
            st.session_state.job = {"kind": "summarize", "payload": name}
            st.rerun()
        if right.button("Remove", key=f"del-{name}"):
            _delete(name)
            st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown('<p class="side-label">Search</p>', unsafe_allow_html=True)
    return st.sidebar.slider("Snippets per answer", min_value=3, max_value=5, value=4)


def main() -> None:
    """Draw the page and handle one chat turn."""

    st.set_page_config(
        page_title="Fastshot — Ask your notes",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(BRAND, unsafe_allow_html=True)

    model = cached_model()
    k = _sidebar(model)

    if st.session_state.banner:
        st.markdown(
            f'<div class="banner">{html.escape(st.session_state.banner)}</div>',
            unsafe_allow_html=True,
        )
        st.session_state.banner = ""

    if not st.session_state.index and not st.session_state.messages:
        st.markdown(EMPTY, unsafe_allow_html=True)

    _render_history()

    chip_quiz, chip_clear, _spacer = st.columns([1.15, 1, 2.4])
    if chip_quiz.button("5 quiz questions", type="primary", use_container_width=True):
        st.session_state.job = {"kind": "quiz", "payload": None}
        st.rerun()
    if chip_clear.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    prompt = st.chat_input("Ask about your uploaded notes")
    if prompt:
        if not st.session_state.index:
            st.session_state.banner = (
                "Upload a lecture PDF or a text note first. "
                "Answers come only from your files."
            )
        else:
            st.session_state.messages.append({"role": "user", "content": prompt, "snippets": []})
            st.session_state.job = {"kind": "ask", "payload": prompt}
        st.rerun()

    if st.session_state.job:
        _run_job(model, k)
        st.rerun()


if __name__ == "__main__":
    main()
