"""Chunking and retrieval tests. These never call the LLM."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rag  # noqa: E402

SAMPLES = ROOT / "sample_docs"


class ChunkTests(unittest.TestCase):
    """Windows of about 800 characters with a 150-character overlap."""

    def test_short_text_is_one_chunk(self) -> None:
        chunks = rag.chunk_pages([(2, "Mitochondria produce ATP.")], "notes.txt")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page, 2)
        self.assertEqual(chunks[0].source, "notes.txt")
        self.assertIn("ATP", chunks[0].text)

    def test_overlap_and_length(self) -> None:
        text = "word " * 400
        chunks = rag.chunk_pages([(1, text)], "long.txt", size=800, overlap=150)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks[0].text), 800)
        self.assertGreater(len(chunks[0].text), 700)
        # strip() can drop a boundary space, so compare the overlap after trimming it.
        tail = chunks[0].text[-150:].lstrip()
        self.assertTrue(chunks[1].text.startswith(tail[:40]))

    def test_empty_text_file(self) -> None:
        pages, warning = rag.extract_document("blank.txt", b"   ")
        self.assertEqual(pages, [])
        self.assertIsNotNone(warning)
        self.assertIn("empty", warning or "")

    def test_non_utf8_does_not_crash(self) -> None:
        pages, warning = rag.extract_document("latin.txt", "café notes".encode("latin-1"))
        self.assertIsNone(warning)
        self.assertIn("caf", pages[0][1])

    def test_empty_pdf_warns_about_ocr(self) -> None:
        data = _tiny_pdf("")
        pages, warning = rag.extract_document("scan.pdf", data)
        self.assertEqual(pages, [])
        self.assertIn("OCR", warning or "")

    def test_pdf_keeps_page_text(self) -> None:
        data = _tiny_pdf("Consuls served one-year terms in Rome.")
        pages, warning = rag.extract_document("rome.pdf", data)
        self.assertIsNone(warning)
        self.assertEqual(pages[0][0], 1)
        self.assertIn("Consuls", pages[0][1])

    def test_not_found_phrase_is_exact(self) -> None:
        self.assertEqual(
            rag.NOT_FOUND,
            "I couldn't find this in the uploaded documents.",
        )

    def test_recent_messages_keeps_six(self) -> None:
        history = [{"role": "user", "content": str(i), "snippets": []} for i in range(10)]
        kept = rag.recent_messages(history, limit=6)
        self.assertEqual([item["content"] for item in kept], ["4", "5", "6", "7", "8", "9"])


class RetrievalTests(unittest.TestCase):
    """Semantic search over the sample notes. No Groq calls."""

    model = None
    index: dict = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = rag.load_embedding_model()
        cls.index = {}
        for path in sorted(SAMPLES.glob("*.txt")):
            record, warning = rag.index_bytes(path.name, path.read_bytes(), cls.model)
            assert warning is None
            assert record is not None
            cls.index[path.name] = record

    def test_paraphrase_without_shared_keywords(self) -> None:
        query = "Which organelle creates the energy currency?"
        chunk = self.index["cell_energy.txt"]["chunks"][0].text.lower()
        stopwords = {"which", "the", "a", "an", "of", "to"}
        for word in query.lower().replace("?", "").split():
            if word in stopwords:
                continue
            self.assertNotIn(word, chunk)
        hits = rag.search_index(self.model, query, self.index, k=4)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.source, "cell_energy.txt")
        self.assertGreater(hits[0].score, 0.2)

    def test_direct_fact_ranks_cell_notes(self) -> None:
        hits = rag.search_index(self.model, "What do mitochondria produce?", self.index, k=4)
        self.assertEqual(hits[0].chunk.source, "cell_energy.txt")
        self.assertIn("ATP", hits[0].chunk.text)

    def test_delete_removes_chunks_from_search(self) -> None:
        remaining = {name: record for name, record in self.index.items() if name != "cell_energy.txt"}
        hits = rag.search_index(
            self.model,
            "Which organelle creates the energy currency?",
            remaining,
            k=4,
        )
        sources = {hit.chunk.source for hit in hits}
        self.assertNotIn("cell_energy.txt", sources)


def _tiny_pdf(text: str) -> bytes:
    """Build a one-page PDF that pypdf can read. No extra dependency."""

    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 18 Tf 72 720 Td ({safe}) Tj ET\n".encode("latin-1", errors="replace")
    objects = [
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n",
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n",
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj\n",
        b"4 0 obj<< /Length %d >>stream\n" % len(stream) + stream + b"endstream\nendobj\n",
        b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n",
    ]
    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)
    xref_at = len(body)
    body.extend(f"xref\n0 {len(offsets)}\n".encode())
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode())
    body.extend(
        f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    )
    return bytes(body)


if __name__ == "__main__":
    unittest.main()
