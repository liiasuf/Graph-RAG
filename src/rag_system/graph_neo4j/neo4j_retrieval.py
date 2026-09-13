from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from rag_system.embeddings import RemoteEmbedder
from rag_system.graph.entities import extract_entities
from rag_system.graph_neo4j.neo4j_graph_index import Neo4jGraphIndex
from rag_system.index import NumpyCosineIndex, RetrievedChunk
from rag_system.mapping import ChunkMapping


@dataclass
class Neo4jGraphRetriever:
    """
    GraphRAG retriever backed by Neo4j for graph traversal.

    Neo4j knows only about (Entity)-[:CO_OCCURS]->(Entity) and
    (Entity)-[:MENTIONS]->(Chunk {chunk_id}).
    It never stores embeddings or chunk text - those live in:
      - NumpyCosineIndex  (vectors)
      - ChunkMapping      (text / metadata, loaded from mapping.json)

    Flow:
        query -> entities -> Neo4j graph traversal -> chunk_ids
        chunk_ids -> ChunkMapping.chunk_id_to_row() -> row indices
        row indices -> NumpyCosineIndex sub-matrix -> cosine re-rank
        row indices -> ChunkMapping.lookup() -> RetrievedChunk

    `strategy`:
      - "overlay"    - mirrors `graph.retrieval.GraphRetriever`: vector top-k
                       is always included (quality floor), graph adds new
                       candidates not already in the vector top-k, merged and
                       re-sorted by cosine, cut to top_k.
      - "graph_only" - candidates come only from the graph; no partial
                       vector top-up if the graph yields fewer than top_k
                       candidates. If the graph yields NO candidates at all
                       (no seed entities / no graph matches), falls back to
                       pure vector search - `last_trace["fallback_used"]`
                       records whether that happened.
    """

    embedder: RemoteEmbedder
    index: NumpyCosineIndex
    mapping: ChunkMapping
    graph: Neo4jGraphIndex
    max_hops: int = 1
    max_neighbors_per_hop: int = 20
    max_candidate_chunks: int = 300
    strategy: Literal["overlay", "graph_only"] = "overlay"

    last_trace: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def _seed_entities(self, query: str) -> set[str]:
        seeds: set[str] = set()
        for mention in extract_entities(query):
            seeds.update(self.graph.match_entities(mention))
        return seeds

    def _expand(self, seeds: set[str]) -> set[str]:
        visited = set(seeds)
        frontier = set(seeds)
        for _ in range(self.max_hops):
            next_frontier: set[str] = set()
            for e in frontier:
                for n in self.graph.neighbors(e, max_neighbors=self.max_neighbors_per_hop):
                    if n not in visited:
                        next_frontier.add(n)
            if not next_frontier:
                break
            visited |= next_frontier
            frontier = next_frontier
        return visited

    def _graph_candidate_rows(self, entities: set[str]) -> set[int]:
        candidate_rows: set[int] = set()
        for e in entities:
            for chunk_id in self.graph.chunks_for(e):
                row = self.mapping.chunk_id_to_row(chunk_id)
                if row is not None:
                    candidate_rows.add(row)
            if len(candidate_rows) >= self.max_candidate_chunks:
                break
        return candidate_rows

    def retrieve(self, query: str, top_k: int, qvec=None) -> list[RetrievedChunk]:
        if qvec is None:
            qvec = self.embedder.embed_query(query)

        seeds = self._seed_entities(query)
        entities = self._expand(seeds) if seeds else set()

        if self.strategy == "overlay":
            return self._retrieve_overlay(qvec, top_k, entities)
        return self._retrieve_graph_only(qvec, top_k, entities)

    def _retrieve_overlay(self, qvec, top_k: int, entities: set[str]) -> list[RetrievedChunk]:
        vector_hits = self.index.search(qvec, top_k=top_k)
        vector_results = [
            RetrievedChunk(score=score, **self.mapping.lookup(row)) for row, score in vector_hits
        ]
        seen_chunk_ids = {r.chunk_id for r in vector_results}

        graph_rows: set[int] = set()
        for e in entities:
            for chunk_id in self.graph.chunks_for(e):
                if chunk_id in seen_chunk_ids:
                    continue
                row = self.mapping.chunk_id_to_row(chunk_id)
                if row is not None:
                    graph_rows.add(row)
            if len(graph_rows) >= self.max_candidate_chunks:
                break

        self.last_trace = {"strategy": "overlay", "n_graph_candidates": len(graph_rows)}

        if not graph_rows:
            return vector_results

        q = np.asarray(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

        graph_row_list = sorted(graph_rows)[: self.max_candidate_chunks]
        sub_emb = self.index.embeddings[graph_row_list]
        graph_scores = sub_emb @ q

        n_graph = min(top_k, len(graph_row_list))
        best_order = np.argsort(-graph_scores)[:n_graph]

        graph_chunks = [
            RetrievedChunk(
                score=float(graph_scores[oi]),
                **self.mapping.lookup(graph_row_list[oi]),
            )
            for oi in best_order.tolist()
        ]

        merged = list(vector_results)
        for gc in graph_chunks:
            if len(merged) < top_k:
                merged.append(gc)
            else:
                worst_idx = min(range(len(merged)), key=lambda i: merged[i].score)
                if gc.score > merged[worst_idx].score:
                    merged[worst_idx] = gc
                else:
                    break

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:top_k]

    def _retrieve_graph_only(self, qvec, top_k: int, entities: set[str]) -> list[RetrievedChunk]:
        candidate_rows = self._graph_candidate_rows(entities)

        if not candidate_rows:
            self.last_trace = {
                "strategy": "graph_only",
                "fallback_used": True,
                "n_graph_candidates": 0,
            }
            hits = self.index.search(qvec, top_k=top_k)
            return [RetrievedChunk(score=score, **self.mapping.lookup(row)) for row, score in hits]

        self.last_trace = {
            "strategy": "graph_only",
            "fallback_used": False,
            "n_graph_candidates": len(candidate_rows),
        }

        rows = sorted(candidate_rows)[: self.max_candidate_chunks]
        q = np.asarray(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

        sub_emb = self.index.embeddings[rows]
        scores = sub_emb @ q
        order = np.argsort(-scores)[:top_k]

        return [
            RetrievedChunk(score=float(scores[oi]), **self.mapping.lookup(rows[oi]))
            for oi in order.tolist()
        ]
