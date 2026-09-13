from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from rag_system.config import RunConfig
from rag_system.embeddings import RemoteEmbedder
from rag_system.graph.entities import extract_entities
from rag_system.graph_neo4j.neo4j_graph_index import Neo4jGraphIndex
from rag_system.graph_neo4j.neo4j_rerank import Neo4jGraphReranker
from rag_system.graph_neo4j.neo4j_retrieval import Neo4jGraphRetriever
from rag_system.index import NumpyCosineIndex, RetrievedChunk
from rag_system.llm import RemoteLLM
from rag_system.mapping import ChunkMapping
from rag_system.metrics import hallucination_rate
from rag_system.retrieval import Retriever
from rag_system.router import heuristic_route, llm_route
from rag_system.subquestion import decompose_question, followup_subquestion


@dataclass
class AdaptiveRetriever:
    """
    retrieval.mode == "adaptive": router (skip the graph for low-signal
    questions) + iterative subquestion decomposition over the Neo4j graph,
    each independently toggleable via `cfg.adaptive` (see AdaptiveMethodConfig
    docstring in config.py). Reuses the Neo4j retrieval strategies from
    steps 1-2 (`Neo4jGraphRetriever` / `Neo4jGraphReranker`) rather than a
    separate graph-traversal implementation.

    The entire router decision + subquestion loop happens inside
    `retrieve()`, so this object still satisfies the plain
    `.retrieve(query, top_k, qvec=None) -> list[RetrievedChunk]` contract
    and plugs into `pipeline.run_eval_loop` like every other retriever -
    only the FINAL answer generation (over the combined chunk pool this
    method returns) happens outside, in the shared loop.
    """

    embedder: RemoteEmbedder
    index: NumpyCosineIndex
    mapping: ChunkMapping
    graph: Neo4jGraphIndex
    llm: RemoteLLM
    cfg: RunConfig

    last_trace: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._vector_retriever = Retriever(embedder=self.embedder, index=self.index, mapping=self.mapping)

        mode = self.cfg.adaptive.graph_retrieval_mode
        if mode == "graph_neo4j_rerank":
            gr = self.cfg.graph_rerank
            self._graph_retriever = Neo4jGraphReranker(
                embedder=self.embedder, index=self.index, mapping=self.mapping, graph=self.graph,
                vector_pool_k=gr.vector_pool_k, beta=gr.beta, hop_decay=gr.hop_decay, max_hops=gr.max_hops,
            )
        else:
            g = self.cfg.graph
            strategy = "overlay" if mode == "graph_neo4j" else "graph_only"
            self._graph_retriever = Neo4jGraphRetriever(
                embedder=self.embedder, index=self.index, mapping=self.mapping, graph=self.graph,
                max_hops=g.max_hops, max_neighbors_per_hop=g.max_neighbors_per_hop,
                max_candidate_chunks=g.max_candidate_chunks, strategy=strategy,
            )
        self._judge = None  # lazily built LLMJudge, only if sufficiency_check == "llm_judge"

    def retrieve(self, query: str, top_k: int, qvec=None) -> list[RetrievedChunk]:
        stage_times: dict[str, float] = {}
        trace: dict = {}

        router_cfg = self.cfg.adaptive.router
        t0 = time.perf_counter()
        if router_cfg.enabled:
            if router_cfg.method == "heuristic":
                use_graph, reason = heuristic_route(query, router_cfg.min_entities_for_graph)
            else:
                use_graph, reason = llm_route(query, self.llm)
        else:
            use_graph, reason = True, "router disabled - always use graph"
        stage_times["router"] = time.perf_counter() - t0
        trace["router_used_graph"] = use_graph
        trace["router_reason"] = reason

        if qvec is None:
            qvec = self.embedder.embed_query(query)

        base_retriever = self._graph_retriever if use_graph else self._vector_retriever

        t1 = time.perf_counter()
        pool = list(base_retriever.retrieve(query, top_k=top_k, qvec=qvec))
        stage_times["retrieval"] = time.perf_counter() - t1
        seen_ids = {c.chunk_id for c in pool}

        subq_cfg = self.cfg.adaptive.subquestions
        iterations = 0
        if use_graph and subq_cfg.enabled:
            t2 = time.perf_counter()
            pool, sub_trace, iterations = self._subquestion_loop(query, qvec, pool, seen_ids, top_k, subq_cfg)
            stage_times["subquestion"] = time.perf_counter() - t2
            trace.update(sub_trace)

        trace["iterations"] = iterations
        trace["stage_times"] = stage_times
        self.last_trace = trace

        pool.sort(key=lambda c: c.score, reverse=True)
        return pool[:top_k]

    def _subquestion_loop(self, query, qvec, pool, seen_ids, top_k, subq_cfg):
        pool = list(pool)
        subq1_list: list[str] = []
        entity_list: list[str | None] = []
        subq2_list: list[str | None] = []
        verdict_list: list[str] = []
        iterations = 0

        q = np.asarray(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

        for _ in range(max(1, subq_cfg.max_iterations)):
            iterations += 1

            subq1 = decompose_question(query, self.llm)
            subq1_list.append(subq1)

            sub_pool1 = self._graph_retriever.retrieve(subq1, top_k=top_k)
            for c in sub_pool1:
                if c.chunk_id not in seen_ids:
                    pool.append(c)
                    seen_ids.add(c.chunk_id)

            entity_key, entity_display = self._resolve_entity(sub_pool1)
            entity_list.append(entity_display)

            if entity_key is not None:
                self._add_graph_neighborhood(entity_key, seen_ids, pool, q)

                subq2 = followup_subquestion(query, subq1, entity_display, self.llm)
                subq2_list.append(subq2)

                sub_pool2 = self._graph_retriever.retrieve(subq2, top_k=top_k)
                for c in sub_pool2:
                    if c.chunk_id not in seen_ids:
                        pool.append(c)
                        seen_ids.add(c.chunk_id)
            else:
                subq2_list.append(None)

            sufficient, verdict = self._check_sufficiency(query, pool, subq_cfg)
            verdict_list.append(verdict)
            if sufficient:
                break

        trace = {
            "subquestion1": subq1_list,
            "entity_from_graph": entity_list,
            "subquestion2": subq2_list,
            "sufficiency_verdict": verdict_list,
        }
        return pool, trace, iterations

    def _resolve_entity(self, chunks: list[RetrievedChunk]) -> tuple[str | None, str | None]:
        """
        Extract candidate entity mentions from retrieved chunk text and
        resolve the first one that matches a real graph node - never
        arbitrary free text.
        """
        for chunk in chunks:
            for mention in extract_entities(chunk.text):
                matches = self.graph.match_entities(mention)
                if matches:
                    return matches[0], mention
        return None, None

    def _add_graph_neighborhood(self, entity_key: str, seen_ids: set[str], pool: list, q: np.ndarray) -> None:
        """Pull chunks_for()/neighbors() of an already-resolved entity directly, no new search."""
        chunk_ids = list(self.graph.chunks_for(entity_key))
        for n in self.graph.neighbors(entity_key, max_neighbors=self.cfg.graph.max_neighbors_per_hop):
            chunk_ids.extend(self.graph.chunks_for(n))

        for cid in chunk_ids:
            if cid in seen_ids:
                continue
            row = self.mapping.chunk_id_to_row(cid)
            if row is None:
                continue
            score = float(self.index.embeddings[row] @ q)
            pool.append(RetrievedChunk(score=score, **self.mapping.lookup(row)))
            seen_ids.add(cid)

    def _check_sufficiency(self, query: str, pool: list[RetrievedChunk], subq_cfg) -> tuple[bool, str]:
        draft = self._draft_answer(query, pool)

        if subq_cfg.sufficiency_check == "heuristic":
            halluc = hallucination_rate(draft, [c.text for c in pool])
            sufficient = halluc < 1.0
            return sufficient, f"heuristic: hallucination_rate={halluc}"

        if self._judge is None:
            from rag_system.judge import LLMJudge
            self._judge = LLMJudge(self.llm)
        context = "\n\n".join(c.text[:500] for c in pool[:5])
        result = self._judge.score(query, draft, context=context)
        avg = (result.scores.get("faithfulness", 5.0) + result.scores.get("relevance", 5.0)) / 2
        sufficient = avg >= 6.0
        return sufficient, f"llm_judge: avg(faithfulness,relevance)={avg:.1f}"

    def _draft_answer(self, query: str, pool: list[RetrievedChunk]) -> str:
        from rag_system.prompting import build_rag_messages
        messages = build_rag_messages(query, pool, max_context_chars=self.cfg.llm.max_context_chars)
        try:
            return self.llm.generate(messages).text
        except Exception:
            return "unknown"
