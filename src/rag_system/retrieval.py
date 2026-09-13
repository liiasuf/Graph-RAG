from __future__ import annotations

from dataclasses import dataclass

from rag_system.embeddings import RemoteEmbedder
from rag_system.index import NumpyCosineIndex, RetrievedChunk
from rag_system.mapping import ChunkMapping


@dataclass
class Retriever:

    embedder: RemoteEmbedder
    index: NumpyCosineIndex
    mapping: ChunkMapping

    def retrieve(self, query: str, top_k: int, qvec=None) -> list[RetrievedChunk]:
        if qvec is None:
            qvec = self.embedder.embed_query(query)
        hits = self.index.search(qvec, top_k=top_k)
        return [
            RetrievedChunk(score=score, **self.mapping.lookup(row))
            for row, score in hits
        ]
