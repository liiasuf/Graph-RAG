from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """
    Generic text document used for indexing and retrieval.

    - doc_id - an identifier of the document (hash, filename, etc.).
    - title - short title for logging and prompts
    - text - full textual content of the document
    """

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A text chunk derived from a Document."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
