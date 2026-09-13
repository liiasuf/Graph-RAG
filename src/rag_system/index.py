from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


class NumpyCosineIndex:
    """
    Pure vector index: stores only L2-normalised float32 embeddings.

    search() returns (row_idx, score) pairs - no text, no metadata.
    Callers resolve rows to original chunks via ChunkMapping.lookup().

    This separation means the index is reusable across modalities and
    retrieval strategies without carrying redundant text in memory.
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D [n, d]")
        self.embeddings = _l2_normalize(embeddings)

    @classmethod
    def build(cls, embeddings: np.ndarray) -> "NumpyCosineIndex":
        return cls(embeddings)

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings.astype(np.float32))

    @classmethod
    def load(cls, path) -> "NumpyCosineIndex":
        embeddings = np.load(Path(path) / "embeddings.npy")
        return cls(embeddings)

    def search(self, query_vec, top_k: int) -> list[tuple[int, float]]:
        """
        Return [(row_idx, cosine_score), ...] sorted descending by score.
        Row indices are resolved to chunk text via ChunkMapping.lookup().
        """
        if top_k <= 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

        scores = self.embeddings @ q
        k = min(top_k, scores.shape[0])

        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        return [(int(i), float(scores[i])) for i in idx.tolist()]

    def __len__(self) -> int:
        return self.embeddings.shape[0]
