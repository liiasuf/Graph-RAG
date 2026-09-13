from __future__ import annotations

import json
from pathlib import Path


class ChunkMapping:
    """
    Persistent map from vector-index row position -> original chunk metadata.

    This is the only place where chunk text lives after indexing.
    NumpyCosineIndex stores only float32 embeddings and integer row positions;
    every component that needs the original text resolves row -> chunk via this
    mapping, loaded from mapping.json on disk.

    Layout of mapping.json:
        [{"index": 0, "chunk_id": "...", "doc_id": "...", "title": "...", "text": "..."}, ...]
    """

    def __init__(self, entries: list[dict]) -> None:
        self._entries = entries
        self._chunk_id_to_row: dict[str, int] = {
            e["chunk_id"]: e["index"] for e in entries
        }

    @classmethod
    def build(cls, chunks) -> "ChunkMapping":
        entries = [
            {
                "index": i,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "text": c.text,
            }
            for i, c in enumerate(chunks)
        ]
        return cls(entries)

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "mapping.json").write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "ChunkMapping":
        data = json.loads((Path(path) / "mapping.json").read_text(encoding="utf-8"))
        return cls(data)

    def lookup(self, row_idx: int) -> dict:
        """
        Return chunk metadata for a given vector-index row position.
        Keys: chunk_id, doc_id, title, text  (no 'index' field).
        """
        e = self._entries[row_idx]
        return {k: e[k] for k in ("chunk_id", "doc_id", "title", "text")}

    def chunk_id_to_row(self, chunk_id: str) -> int | None:
        """Resolve a chunk_id (e.g. from Neo4j) to its vector-index row."""
        return self._chunk_id_to_row.get(chunk_id)

    def __len__(self) -> int:
        return len(self._entries)
