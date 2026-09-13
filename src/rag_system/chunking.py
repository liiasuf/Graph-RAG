from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def _words(text):
    return [w for w in text.strip().split() if w]


def chunk_document_words(*, doc_id, title, text, max_words, overlap_words):
    """
    A fuction that splits a document into chunks based on a specified maximum number of words and an overlap between chunks.
    Each chunk is represented as a `Chunk` dataclass instance, containing the chunk ID, document ID, title, and text of the chunk.
    """

    if max_words <= 0:
        raise ValueError("max_words must be > 0")

    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")

    if overlap_words >= max_words:
        raise ValueError("overlap_words must be < max_words")

    ws = _words(text)
    if not ws:
        return []

    chunks: list[Chunk] = []
    start = 0
    i = 0
    step = max_words - overlap_words

    while start < len(ws):
        end = min(start + max_words, len(ws))

        chunk_text = " ".join(ws[start:end])
        chunk_id = f"{doc_id}:{i}"
        chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, title=title, text=chunk_text))

        if end >= len(ws):
            break

        start += step
        i += 1

    return chunks
