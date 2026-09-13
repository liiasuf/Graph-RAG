from __future__ import annotations

from dataclasses import dataclass, field

from rag_system.embeddings import RemoteEmbedder
from rag_system.graph.entities import extract_entities
from rag_system.graph_neo4j.neo4j_graph_index import Neo4jGraphIndex
from rag_system.index import NumpyCosineIndex, RetrievedChunk
from rag_system.mapping import ChunkMapping


@dataclass
class Neo4jGraphReranker:
    """
    "Graph as a re-ranker" retrieval mode (retrieval.mode ==
    "graph_neo4j_rerank"): the graph never contributes new candidate
    documents, only re-scores a vector-search pool. This isolates the
    question of whether graph proximity is a useful RANKING signal from
    whether graph traversal finds documents vector search misses (that's
    what "graph_neo4j"/"graph_neo4j_only" test).

    Flow:
        query -> vector search -> top `vector_pool_k` candidates
        query -> seed entities (extract_entities + Neo4jGraphIndex.match_entities)
        candidates -> Neo4jGraphIndex.hop_distances() -> one batched Cypher call
        combined = beta * hop_decay**hops + (1 - beta) * cosine  -> sort -> top_k

    Exposes the same `.retrieve(query, top_k, qvec=None)` contract as the
    other retrievers so it plugs directly into `pipeline.run_eval_loop`.
    """

    embedder: RemoteEmbedder
    index: NumpyCosineIndex
    mapping: ChunkMapping
    graph: Neo4jGraphIndex
    vector_pool_k: int = 30
    beta: float = 0.5
    hop_decay: float = 0.5
    max_hops: int = 3

    last_trace: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def _seed_entities(self, query: str) -> list[str]:
        seeds: set[str] = set()
        for mention in extract_entities(query):
            seeds.update(self.graph.match_entities(mention))
        return sorted(seeds)

    def retrieve(self, query: str, top_k: int, qvec=None) -> list[RetrievedChunk]:
        if qvec is None:
            qvec = self.embedder.embed_query(query)

        pool_k = max(self.vector_pool_k, top_k)
        hits = self.index.search(qvec, top_k=pool_k)
        pool = [RetrievedChunk(score=score, **self.mapping.lookup(row)) for row, score in hits]

        if not pool:
            self.last_trace = {"n_seed_entities": 0, "graph_signal_available": False}
            return pool

        seeds = self._seed_entities(query)
        if not seeds:
            # no graph signal available - combined score degenerates to a
            # monotonic function of cosine, so the vector order is preserved
            self.last_trace = {"n_seed_entities": 0, "graph_signal_available": False}
            return pool[:top_k]

        chunk_ids = [c.chunk_id for c in pool]
        hops_by_chunk = self.graph.hop_distances(seeds, chunk_ids, max_hops=self.max_hops)

        scored: list[tuple[float, RetrievedChunk]] = []
        n_with_signal = 0
        for c in pool:
            hops = hops_by_chunk.get(c.chunk_id)
            if hops is not None:
                graph_signal = self.hop_decay**hops
                n_with_signal += 1
            else:
                graph_signal = 0.0
            cosine_norm = max(0.0, min(1.0, float(c.score)))
            combined = self.beta * graph_signal + (1 - self.beta) * cosine_norm
            scored.append((combined, RetrievedChunk(
                chunk_id=c.chunk_id, doc_id=c.doc_id, title=c.title, text=c.text, score=combined,
            )))

        scored.sort(key=lambda t: t[0], reverse=True)

        self.last_trace = {
            "n_seed_entities": len(seeds),
            "graph_signal_available": True,
            "n_candidates_with_graph_signal": n_with_signal,
        }
        return [c for _, c in scored[:top_k]]
