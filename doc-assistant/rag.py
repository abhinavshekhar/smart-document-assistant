"""Retrieval and generation for the document assistant.

Embeddings stay on this machine. The only network call is ``generate_text``,
which talks to Groq. Swap providers by editing that function.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterator

import numpy as np

# llama-3.3-70b-versatile was removed from Groq's free tier on 2026-08-16.
# openai/gpt-oss-120b is the documented replacement that still has a free rate limit.
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_DEFAULT = 4
MAX_CONTEXT_CHARS = 3200
NOT_FOUND = "I couldn't find this in the uploaded documents."
MISSING_KEY_MESSAGE = (
    "No Groq API key found. Create a free key at https://console.groq.com "
    "(no credit card). Locally, set the GROQ_API_KEY environment variable. "
    "On Streamlit Community Cloud, open the app settings, then Advanced settings, "
    "and add GROQ_API_KEY under Secrets."
)
RATE_LIMIT_MESSAGE = "Free API limit reached, wait a minute and try again."


class AssistantError(Exception):
    """A failure the UI can show without a stack trace."""


@dataclass
class Chunk:
    """One slice of an uploaded file."""

    source: str
    page: int
    text: str


@dataclass
class ScoredChunk:
    """A retrieved chunk plus its cosine similarity to the query."""

    chunk: Chunk
    score: float


def decode_text(data: bytes) -> str:
    """Decode note bytes. Latin-1 is the fallback so odd encodings do not crash."""

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def extract_document(filename: str, data: bytes) -> tuple[list[tuple[int, str]], str | None]:
    """Pull ``(page_number, text)`` pairs out of a PDF or text file.

    Returns a warning when the file is empty or a scan with no text layer.
    Page numbers start at 1. Text files are treated as a single page.
    """

    if not data:
        return [], f"{filename} is empty, so nothing was indexed."

    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix == "pdf":
        return _extract_pdf(filename, data)
    pages = [(1, decode_text(data).strip())]
    if not pages[0][1]:
        return [], f"{filename} is empty, so nothing was indexed."
    return pages, None


def _extract_pdf(filename: str, data: bytes) -> tuple[list[tuple[int, str]], str | None]:
    """Read a PDF with pypdf, keeping each page's number."""

    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(BytesIO(data))
    except (PdfReadError, Exception):
        return [], f"{filename} could not be read. Try exporting it again."

    pages: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = raw.strip()
        if text:
            pages.append((number, text))
    if not pages:
        return [], (
            f"{filename} has no extractable text. Scanned pages need OCR, "
            "which this app does not support."
        )
    return pages, None


def chunk_pages(
    pages: list[tuple[int, str]],
    source: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split pages into overlapping character windows of about ``size``."""

    chunks: list[Chunk] = []
    for page_number, text in pages:
        chunks.extend(_chunk_page(text, page_number, source, size, overlap))
    return chunks


def _chunk_page(text: str, page: int, source: str, size: int, overlap: int) -> list[Chunk]:
    """Slide a window across one page. The step is ``size - overlap``."""

    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if overlap >= size:
        overlap = size // 5
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    start = 0
    while start < len(cleaned):
        piece = cleaned[start : start + size].strip()
        if piece:
            chunks.append(Chunk(source=source, page=page, text=piece))
        if start + size >= len(cleaned):
            break
        start += step
    return chunks


def load_embedding_model() -> Any:
    """Load all-MiniLM-L6-v2. Cache this once in the Streamlit app."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(model: Any, texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings, shape ``(n, dim)``."""

    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def index_bytes(
    filename: str,
    data: bytes,
    model: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Chunk and embed one upload. ``None`` means nothing was added."""

    pages, warning = extract_document(filename, data)
    if not pages:
        return None, warning
    chunks = chunk_pages(pages, filename)
    if not chunks:
        return None, warning or f"{filename} is empty, so nothing was indexed."
    embeddings = embed_texts(model, [chunk.text for chunk in chunks])
    return {"chunks": chunks, "embeddings": embeddings}, warning


def search_index(
    model: Any,
    query: str,
    index: dict[str, dict[str, Any]],
    k: int = TOP_K_DEFAULT,
) -> list[ScoredChunk]:
    """Return the ``k`` chunks closest to ``query`` by cosine similarity."""

    blocks: list[np.ndarray] = []
    chunks: list[Chunk] = []
    for record in index.values():
        matrix = record["embeddings"]
        if len(matrix) == 0:
            continue
        blocks.append(matrix)
        chunks.extend(record["chunks"])
    if not chunks:
        return []

    matrix = np.vstack(blocks)
    query_vec = embed_texts(model, [query])[0]
    # Vectors are normalized, so the dot product is cosine similarity.
    scores = matrix @ query_vec
    k = max(1, min(k, len(chunks)))
    # argpartition is enough; the shortlist is sorted afterwards.
    picked = np.argpartition(-scores, k - 1)[:k]
    ordered = picked[np.argsort(-scores[picked])]
    return [ScoredChunk(chunk=chunks[int(i)], score=float(scores[int(i)])) for i in ordered]


def recent_messages(history: list[dict[str, Any]], limit: int = 6) -> list[dict[str, str]]:
    """Keep the last few chat turns, dropping snippet metadata."""

    trimmed = history[-limit:]
    return [{"role": item["role"], "content": item["content"]} for item in trimmed if item.get("content")]


def get_api_key() -> str:
    """Read the key from Streamlit secrets, then from the environment."""

    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            value = str(st.secrets["GROQ_API_KEY"]).strip()
            if value:
                return value
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


def generate_text(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 700,
    temperature: float = 0.2,
    stream: bool = False,
) -> str | Iterator[str]:
    """Call Groq. This is the only function that knows which provider we use."""

    key = get_api_key()
    if not key:
        raise AssistantError(MISSING_KEY_MESSAGE)

    from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq, RateLimitError

    client = Groq(api_key=key)
    try:
        if stream:
            return _stream_tokens(client, messages, max_tokens, temperature, RateLimitError, APIConnectionError, APITimeoutError, APIStatusError)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (completion.choices[0].message.content or "").strip()
    except RateLimitError as exc:
        raise AssistantError(RATE_LIMIT_MESSAGE) from exc
    except (APIConnectionError, APITimeoutError) as exc:
        raise AssistantError("Could not reach the free AI service. Check your connection and try again.") from exc
    except APIStatusError as exc:
        if getattr(exc, "status_code", None) == 429:
            raise AssistantError(RATE_LIMIT_MESSAGE) from exc
        raise AssistantError("The free AI service could not answer just now. Please try again.") from exc


def _stream_tokens(
    client: Any,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    rate_limit_error: type[Exception],
    connection_error: type[Exception],
    timeout_error: type[Exception],
    status_error: type[Exception],
) -> Iterator[str]:
    """Yield answer text as Groq sends it. Errors stay user-facing."""

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in completion:
            piece = chunk.choices[0].delta.content or ""
            if piece:
                yield piece
    except rate_limit_error as exc:
        raise AssistantError(RATE_LIMIT_MESSAGE) from exc
    except (connection_error, timeout_error) as exc:
        raise AssistantError("Could not reach the free AI service. Check your connection and try again.") from exc
    except status_error as exc:
        if getattr(exc, "status_code", None) == 429:
            raise AssistantError(RATE_LIMIT_MESSAGE) from exc
        raise AssistantError("The free AI service could not answer just now. Please try again.") from exc


def rewrite_query(question: str, history: list[dict[str, Any]]) -> str:
    """Turn a follow-up into a standalone search query. Falls back to the original."""

    prior = recent_messages(history, limit=6)
    if not prior:
        return question
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the latest question as one standalone search query. "
                "Resolve words like 'that' or 'it' using the chat. "
                "Return only the query."
            ),
        },
        *prior,
        {"role": "user", "content": question},
    ]
    try:
        rewritten = str(generate_text(messages, max_tokens=80, temperature=0))
    except AssistantError:
        return question
    cleaned = rewritten.strip().strip('"')
    return cleaned or question


def build_context(hits: list[ScoredChunk]) -> str:
    """Format retrieved chunks as a short labeled context block."""

    parts: list[str] = []
    used = 0
    for number, hit in enumerate(hits, start=1):
        block = f"[Source {number}] ({hit.chunk.source}, page {hit.chunk.page})\n{hit.chunk.text}"
        if used + len(block) > MAX_CONTEXT_CHARS and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def answer_messages(question: str, hits: list[ScoredChunk], history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build the chat payload for a grounded answer."""

    system = (
        "You answer a first-year student using ONLY the context below. "
        "Rules: use no outside knowledge. Explain simply, in a few sentences. "
        "Cite the context labels as [Source 1] or [Source 2]. "
        f"If the answer is not in the context, reply exactly: {NOT_FOUND}"
    )
    context = build_context(hits) or "(no context)"
    messages: list[dict[str, str]] = [{"role": "system", "content": f"{system}\n\nContext:\n{context}"}]
    messages.extend(recent_messages(history, limit=6))
    messages.append({"role": "user", "content": question})
    return messages


def summary_messages(filename: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    """Prompt a short summary of one uploaded file."""

    body = _clip("\n\n".join(chunk.text for chunk in chunks))
    return [
        {
            "role": "system",
            "content": (
                "Summarize the notes for a first-year student. Use only the text provided. "
                "Use short paragraphs and plain words. Do not add facts that are not in the text."
            ),
        },
        {"role": "user", "content": f"Summarize {filename}.\n\n{body}"},
    ]


def quiz_messages(index: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Prompt five quiz questions drawn only from the uploaded notes."""

    pieces: list[str] = []
    for record in index.values():
        for chunk in record["chunks"]:
            pieces.append(f"{chunk.source} page {chunk.page}: {chunk.text}")
    body = _clip("\n".join(pieces))
    return [
        {
            "role": "system",
            "content": (
                "Write exactly 5 quiz questions for a first-year student, using only the notes. "
                "After each question, give the answer in one sentence. "
                "Do not test anything that is not in the notes."
            ),
        },
        {"role": "user", "content": body or "No notes were provided."},
    ]


def _clip(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    """Keep prompts inside the free tier's context budget."""

    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0]


def snippet_payload(hits: list[ScoredChunk]) -> list[dict[str, Any]]:
    """Plain dicts the chat history can store."""

    return [
        {
            "source": hit.chunk.source,
            "page": hit.chunk.page,
            "score": round(hit.score, 3),
            "text": hit.chunk.text,
        }
        for hit in hits
    ]


def demo_retrieval() -> None:
    """Index the sample notes and print the top hit for a paraphrased question."""

    from pathlib import Path

    model = load_embedding_model()
    index: dict[str, dict[str, Any]] = {}
    folder = Path(__file__).resolve().parent / "sample_docs"
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in {".txt", ".pdf"}:
            continue
        record, warning = index_bytes(path.name, path.read_bytes(), model)
        print(f"{path.name}: {0 if record is None else len(record['chunks'])} chunks {warning or ''}")
        if record:
            index[path.name] = record
    query = "Which organelle creates the energy currency?"
    hits = search_index(model, query, index, k=4)
    print(f"query: {query}")
    for hit in hits:
        print(f"  {hit.score:.3f} {hit.chunk.source} p{hit.chunk.page}: {hit.chunk.text[:80]}")


if __name__ == "__main__":
    demo_retrieval()
