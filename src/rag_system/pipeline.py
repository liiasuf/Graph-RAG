from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from rag_system.chunking import chunk_document_words
from rag_system.embeddings import CachedEmbedder, EmbeddingCache, RemoteEmbedder
from rag_system.index import NumpyCosineIndex
from rag_system.latency import LatencySums
from rag_system.mapping import ChunkMapping
from rag_system.metrics import (
    MetricSums, exact_match, f1_score, hallucination_rate, mrr,
    ndcg_at_k, precision_at_k, recall_at_k, rouge_l,
)
from rag_system.prompting import build_rag_messages

try:
    from openai import BadRequestError as OpenAIBadRequestError
except ImportError:
    OpenAIBadRequestError = Exception  # fallback if openai not installed

logger = logging.getLogger(__name__)


def _max_corpus_chunks(cfg) -> int | None:
    """Return max_corpus_chunks regardless of which dataset config section is active."""
    return getattr(cfg.hotpotqa, "max_corpus_chunks", None)


def build_corpus_chunks(cfg, dataset, logger=None):
    """
    Load the train corpus for `dataset` and split it into chunks according
    to `cfg.chunking`. Optionally summarizes chunks via LLM if
    `cfg.summarization.enabled` is True.

    Shared between the baseline (vector) and GraphRAG pipelines so both
    operate on exactly the same corpus/chunks for a fair comparison.
    """
    _log = logger or logging.getLogger(__name__)

    _log.info("Building corpus for dataset: %s", dataset.name)
    docs = dataset.build_train_corpus()
    _log.info("Loaded train corpus: %d documents", len(docs))

    max_chunks = _max_corpus_chunks(cfg)
    chunks = []
    for d in tqdm(docs, desc="Chunking documents"):
        chunks.extend(
            chunk_document_words(
                doc_id=d.doc_id,
                title=d.title,
                text=d.text,
                max_words=cfg.chunking.max_words,
                overlap_words=cfg.chunking.overlap_words,
            )
        )
        if max_chunks is not None and len(chunks) >= max_chunks:
            chunks = chunks[:max_chunks]
            break

    _log.info("Total chunks: %d", len(chunks))

    if getattr(cfg, "summarization", None) and cfg.summarization.enabled:
        _log.info("Summarization enabled - building LLM for summarization")
        from rag_system.llm import RemoteLLM
        from rag_system.summarizer import summarize_chunks
        sum_llm = RemoteLLM(
            base_url=cfg.llm.base_url,
            api_key=cfg.llm.api_key,
            model=cfg.llm.model,
            temperature=0.0,
            max_tokens=128,
            timeout_s=cfg.llm.timeout_s,
        )
        chunks = summarize_chunks(
            chunks, sum_llm,
            max_input_chars=cfg.summarization.max_input_chars,
        )

    return docs, chunks


def build_embedder(cfg, logger=None):
    """
    Build a `RemoteEmbedder`, optionally wrapped in `CachedEmbedder` when
    `cfg.embeddings.use_cache` is set. Returns `(embedder, cache_or_none)`;
    call `cache.save()` after embedding to persist new entries.
    """
    _log = logger or logging.getLogger(__name__)

    base_embedder = RemoteEmbedder(
        base_url=cfg.embeddings.base_url,
        api_key=cfg.embeddings.api_key,
        model=cfg.embeddings.model,
        batch_size=cfg.embeddings.batch_size,
        show_progress=True,
    )

    cache = None
    embedder = base_embedder

    if cfg.embeddings.use_cache and cfg.embeddings.cache_path:
        cache = EmbeddingCache.load(cfg.embeddings.cache_path)
        embedder = _AutoSaveCachedEmbedder(base=base_embedder, cache=cache)
        _log.info("Embedding cache enabled (auto-save): %s", cfg.embeddings.cache_path)

    return embedder, cache


class _AutoSaveCachedEmbedder(CachedEmbedder):

    def embed_query(self, text: str):
        vec = super().embed_query(text)
        if self.cache is not None:
            self.cache.save()
        return vec


def build_vector_index(chunks, embedder, logger=None):
    """
    Embed chunks and build a NumpyCosineIndex + ChunkMapping.

    Returns (index, mapping):
      - index   - pure float32 embeddings, search() returns (row, score)
      - mapping - mapping.json: row -> {chunk_id, doc_id, title, text}

    Retrievers resolve row indices to text via mapping, never via the index.
    """
    _log = logger or logging.getLogger(__name__)
    _log.info("Embedding %d chunks...", len(chunks))

    chunk_texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(chunk_texts)

    mapping = ChunkMapping.build(chunks)
    index = NumpyCosineIndex.build(embeddings)
    _log.info("Index built: %d vectors", len(chunks))
    return index, mapping


def build_reranker(cfg, llm):
    """
    Build an LLMReranker if `cfg.reranker.enabled`, else return None.
    Pass the already-constructed `llm` to avoid a second model instance.
    """
    if not getattr(cfg, "reranker", None) or not cfg.reranker.enabled:
        return None
    from rag_system.reranker import LLMReranker
    logger.info("LLM re-ranker enabled (alpha=%.2f)", cfg.reranker.alpha)
    return LLMReranker(
        llm=llm,
        alpha=cfg.reranker.alpha,
        max_passage_chars=cfg.reranker.max_passage_chars,
    )


def maybe_expand_query(query: str, cfg, llm) -> str:
    """Return LLM-expanded query if query_expansion.enabled, else original."""
    if not getattr(cfg, "query_expansion", None) or not cfg.query_expansion.enabled:
        return query
    from rag_system.query_expansion import expand_query
    return expand_query(query, llm)


def build_qvec_map(cfg, eval_examples, cache=None, logger=None) -> dict:
    _log = logger or logging.getLogger(__name__)
    _log.info("Pre-embedding %d queries...", len(eval_examples))

    base_emb = RemoteEmbedder(
        base_url=cfg.embeddings.base_url,
        api_key=cfg.embeddings.api_key,
        model=cfg.embeddings.model,
        batch_size=cfg.embeddings.batch_size,
        show_progress=True,
    )
    questions = [ex.question for ex in eval_examples]
    qvecs = base_emb.embed_texts(questions)
    qvec_map = {ex.question: qvecs[i] for i, ex in enumerate(eval_examples)}

    if cache is not None:
        for ex in eval_examples:
            cache.set(ex.question, qvec_map[ex.question])
        cache.save()

    _log.info("Query pre-embedding done.")
    return qvec_map


def _aggregate_trace_stats(traces: list[dict]) -> dict:
    """
    Fold a list of per-example trace dicts into summary-level stats:
    bool fields -> usage rate, numeric fields -> mean. Heterogeneous /
    string fields are left out of the aggregate (they still appear
    per-example in per_example.jsonl).
    """
    if not traces:
        return {}

    keys: set[str] = set()
    for t in traces:
        keys.update(t.keys())

    stats: dict = {}
    for k in keys:
        vals = [t[k] for t in traces if t.get(k) is not None]
        if not vals:
            continue
        if all(isinstance(v, bool) for v in vals):
            stats[f"{k}_rate"] = round(sum(1 for v in vals if v) / len(traces), 5)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            stats[f"{k}_mean"] = round(sum(vals) / len(vals), 5)

    return stats


def run_eval_loop(
    *, cfg, eval_examples, llm, retriever, reranker, qvec_map, run_dir, index_time_s, logger=None,
) -> dict:
    """
    Shared retrieve -> [rerank] -> generate -> metrics -> per_example.jsonl
    loop, used by every e2e_* orchestrator (vector, in-memory graph, all
    Neo4j-backed modes, adaptive).

    `retriever` must implement `.retrieve(query, top_k, qvec=None) ->
    list[RetrievedChunk]`. If it also sets `retriever.last_trace` (a dict)
    during `retrieve()`, that dict is used two ways:
      - `last_trace["stage_times"]` (dict[str, float] seconds), if present,
        is folded into `LatencySums.add_stage(...)` for a per-stage latency
        breakdown (e.g. router / retrieval / subquestion for adaptive mode).
      - every other key is written into that example's `per_example.jsonl`
        row under `"trace"`, and aggregated across all examples into the
        returned `"trace_stats"` (bool -> usage rate, number -> mean) - e.g.
        `router_used_graph` -> `router_used_graph_rate` in summary.json.

    Returns {"metrics": MetricSums, "latency": LatencySums, "trace_stats": dict}.
    Writes `run_dir / "per_example.jsonl"`.
    """
    _log = logger or logging.getLogger(__name__)

    metrics = MetricSums()
    latency = LatencySums(index_time_s=index_time_s)
    traces: list[dict] = []

    per_example_path = Path(run_dir) / "per_example.jsonl"

    with per_example_path.open("w", encoding="utf-8") as f_out:
        for ex in tqdm(eval_examples, desc="Evaluating"):
            t_q0 = time.perf_counter()

            t_r0 = time.perf_counter()
            query = maybe_expand_query(ex.question, cfg, llm)
            retrieved = retriever.retrieve(query, top_k=cfg.retrieval.top_k, qvec=qvec_map.get(ex.question))
            if reranker is not None:
                retrieved = reranker.rerank(ex.question, retrieved)
            retrieval_time_s = time.perf_counter() - t_r0

            trace = getattr(retriever, "last_trace", None) or {}
            stage_times = trace.get("stage_times") or {}
            for stage_name, seconds in stage_times.items():
                latency.add_stage(stage_name, seconds)
            trace_fields = {k: v for k, v in trace.items() if k != "stage_times"}
            if trace_fields:
                traces.append(trace_fields)

            messages = build_rag_messages(
                ex.question, retrieved, max_context_chars=cfg.llm.max_context_chars
            )

            t_gen0 = time.perf_counter()
            try:
                llm_res = llm.generate(messages)
            except OpenAIBadRequestError as e:
                _log.warning("LLM 400 on qid=%s, using fallback: %s", ex.qid, e)
                pred = "unknown"
                llm_res = type(
                    "_Fallback",
                    (),
                    {
                        "text": pred,
                        "usage": type(
                            "_Usage",
                            (),
                            {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                        )(),
                    },
                )()
            generation_time_s = time.perf_counter() - t_gen0

            total_q_time = time.perf_counter() - t_q0

            pred = llm_res.text
            gold = ex.answer

            em = exact_match(pred, gold)
            f1 = f1_score(pred, gold)
            acc = em
            retrieved_titles = [r.title for r in retrieved]
            rec = recall_at_k(retrieved_titles, ex.supporting_titles)
            rr = mrr(retrieved_titles, ex.supporting_titles)
            halluc = hallucination_rate(pred, [r.text for r in retrieved])
            prec = precision_at_k(retrieved_titles, ex.supporting_titles)
            ndcg = ndcg_at_k(retrieved_titles, ex.supporting_titles)
            rl = rouge_l(pred, gold)

            metrics.add(em=em, f1=f1, accuracy=acc, recall=rec, mrr=rr,
                        precision=prec, ndcg=ndcg, rouge_l_val=rl, hallucination=halluc)

            latency.add_request(
                retrieval_time_s=retrieval_time_s,
                generation_time_s=generation_time_s,
                total_query_time_s=total_q_time,
                prompt_tokens=llm_res.usage.prompt_tokens,
                completion_tokens=llm_res.usage.completion_tokens,
                total_tokens=llm_res.usage.total_tokens,
            )

            row = {
                "id": ex.qid,
                "question": ex.question,
                "gold": gold,
                "pred": pred,
                "supporting_titles": ex.supporting_titles,
                "retrieved_titles": retrieved_titles,
                "em": em,
                "f1": f1,
                "accuracy": acc,
                "recall": rec,
                "mrr": rr,
                "precision": prec,
                "ndcg": ndcg,
                "rouge_l": rl,
                "hallucination": halluc,
                "latency": {
                    "retrieval_time_s": retrieval_time_s,
                    "generation_time_s": generation_time_s,
                    "total_query_time_s": total_q_time,
                    "usage": {
                        "prompt_tokens": llm_res.usage.prompt_tokens,
                        "completion_tokens": llm_res.usage.completion_tokens,
                        "total_tokens": llm_res.usage.total_tokens,
                    },
                },
                "retrieved": [
                    {"chunk_id": r.chunk_id, "title": r.title, "score": r.score, "text": r.text}
                    for r in retrieved
                ],
            }
            if trace_fields:
                row["trace"] = trace_fields

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "metrics": metrics,
        "latency": latency,
        "trace_stats": _aggregate_trace_stats(traces),
    }
