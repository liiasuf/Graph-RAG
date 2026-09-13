from __future__ import annotations

import hashlib
from pathlib import Path

from rag_system.schema import Document

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx"}


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    if suffix == ".docx":
        import docx2txt

        return docx2txt.process(str(path)) or ""

    return ""


def load_documents_from_dir(corpus_dir: str, max_docs: int | None = None) -> list[Document]:
    root = Path(corpus_dir)
    if not root.exists():
        return []

    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )

    docs: list[Document] = []
    for path in paths:
        text = _extract_text(path).strip()
        if not text:
            continue
        doc_id = hashlib.sha1(str(path.relative_to(root)).encode("utf-8")).hexdigest()[:16]
        docs.append(Document(doc_id=doc_id, title=path.stem, text=text))
        if max_docs is not None and len(docs) >= max_docs:
            break

    return docs
