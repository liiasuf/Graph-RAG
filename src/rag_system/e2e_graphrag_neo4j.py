from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag_system.config import RunConfig, load_config
from rag_system.datasets import build_dataset
from rag_system.graph_neo4j import AdaptiveRetriever, Neo4jGraphIndex, Neo4jGraphRetriever
from rag_system.graph_neo4j.neo4j_rerank import Neo4jGraphReranker
from rag_system.llm import RemoteLLM
from rag_system.logging_utils import get_logger, setup_logging
from rag_system.pipeline import (
    build_corpus_chunks, build_embedder, build_qvec_map,
    build_reranker, build_vector_index, run_eval_loop,
)

# Every retrieval.mode this orchestrator supports, and the "system" value
# each writes to metrics.json (see docs/PROJECT.md retrieval.mode table).
_SYSTEM_NAMES = {
    "graph_neo4j": "graph_rag_neo4j",
    "graph_neo4j_only": "graph_rag_neo4j_only",
    "graph_neo4j_rerank": "graph_rag_neo4j_rerank",
    "adaptive": "adaptive",
}


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_retriever(cfg: RunConfig, embedder, index, mapping, graph, llm):
    """
    Build the retriever for `cfg.retrieval.mode`. All four Neo4j-backed
    modes share one orchestrator/eval-loop (`pipeline.run_eval_loop`)
    instead of four copy-pasted scripts - only the retriever differs.
    """
    mode = cfg.retrieval.mode

    if mode == "graph_neo4j":
        return Neo4jGraphRetriever(
            embedder=embedder, index=index, mapping=mapping, graph=graph,
            max_hops=cfg.graph.max_hops,
            max_neighbors_per_hop=cfg.graph.max_neighbors_per_hop,
            max_candidate_chunks=cfg.graph.max_candidate_chunks,
            strategy="overlay",
        )
    if mode == "graph_neo4j_only":
        return Neo4jGraphRetriever(
            embedder=embedder, index=index, mapping=mapping, graph=graph,
            max_hops=cfg.graph.max_hops,
            max_neighbors_per_hop=cfg.graph.max_neighbors_per_hop,
            max_candidate_chunks=cfg.graph.max_candidate_chunks,
            strategy="graph_only",
        )
    if mode == "graph_neo4j_rerank":
        gr = cfg.graph_rerank
        return Neo4jGraphReranker(
            embedder=embedder, index=index, mapping=mapping, graph=graph,
            vector_pool_k=gr.vector_pool_k, beta=gr.beta, hop_decay=gr.hop_decay, max_hops=gr.max_hops,
        )
    if mode == "adaptive":
        return AdaptiveRetriever(embedder=embedder, index=index, mapping=mapping, graph=graph, llm=llm, cfg=cfg)

    raise ValueError(
        f"e2e_graphrag_neo4j supports retrieval.mode in {sorted(_SYSTEM_NAMES)}, got {mode!r}"
    )


def run_e2e_graphrag_neo4j(config_path: str) -> None:
    """
    Neo4j-backed GraphRAG orchestrator. Handles every retrieval.mode that
    needs the Neo4j entity graph (graph_neo4j / graph_neo4j_only /
    graph_neo4j_rerank / adaptive - see `_build_retriever`), mirroring
    `rag_system.e2e_graphrag.run_e2e_graphrag` (same corpus/chunks/vector
    index from `pipeline.py`, same quality/latency metrics via
    `pipeline.run_eval_loop`, same output shape) so results are directly
    comparable to `vector_rag` and the in-memory `graph_rag` (H5 in
    docs/hypotheses.md). The entity co-occurrence graph is built into Neo4j
    (`Neo4jGraphIndex.build`) once per run and traversed via Cypher.

    Requires a running Neo4j instance (see `docker-compose.yml`) and the
    optional `neo4j` dependency: `pip install -e .[neo4j]`.
    """

    cfg: RunConfig = load_config(config_path)

    system_name = _SYSTEM_NAMES.get(cfg.retrieval.mode)
    if system_name is None:
        raise ValueError(
            f"e2e_graphrag_neo4j supports retrieval.mode in {sorted(_SYSTEM_NAMES)}, "
            f"got {cfg.retrieval.mode!r}"
        )

    setup_logging(logging.INFO)
    logger = get_logger("rag_system.e2e_graphrag_neo4j")
    console = Console()

    run_dir = Path(cfg.output_dir) / cfg.experiment_name / _ts()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Run dir:[/bold] {run_dir}")
    logger.info("Run dir: %s", run_dir)

    dataset = build_dataset(cfg)
    logger.info("Dataset: %s", dataset.name)

    t0 = time.perf_counter()

    docs, chunks = build_corpus_chunks(cfg, dataset, logger)

    embedder, cache = build_embedder(cfg, logger)

    index, mapping = build_vector_index(chunks, embedder)
    index_dir = run_dir / "artifacts" / "index"
    index.save(index_dir)
    mapping.save(index_dir)

    if cache is not None:
        cache.save()

    index_time_s = time.perf_counter() - t0
    console.print(
        f"Indexed [bold]{len(chunks)}[/bold] chunks from [bold]{len(docs)}[/bold] docs in "
        f"[bold]{index_time_s:.2f}s[/bold]"
    )
    logger.info("Index time: %.3fs", index_time_s)

    neo4j_cfg = cfg.graph.neo4j
    graph = Neo4jGraphIndex.connect(
        uri=neo4j_cfg.uri,
        user=neo4j_cfg.user,
        password=neo4j_cfg.password,
        database=neo4j_cfg.database,
        experiment=cfg.experiment_name,
    )

    try:
        if neo4j_cfg.clear_before_build:
            logger.info("Clearing existing Neo4j graph for experiment=%s", cfg.experiment_name)
            graph.clear()

        t_g0 = time.perf_counter()
        graph_stats = graph.build(
            chunks,
            title_as_entity=cfg.graph.title_as_entity,
            max_entity_words=cfg.graph.max_entity_words,
            min_entity_chars=cfg.graph.min_entity_chars,
            batch_size=neo4j_cfg.batch_size,
        )
        graph_build_time_s = time.perf_counter() - t_g0

        console.print(
            f"Neo4j graph: [bold]{graph_stats['n_entities']}[/bold] entities, "
            f"[bold]{graph_stats['n_edges']}[/bold] edges over "
            f"[bold]{graph_stats['n_chunks']}[/bold] chunks "
            f"in [bold]{graph_build_time_s:.2f}s[/bold]"
        )
        logger.info("Neo4j graph build time: %.3fs, stats: %s", graph_build_time_s, graph_stats)

        _write_json(
            run_dir / "artifacts" / "graph_stats.json",
            {**graph_stats, "neo4j_database": neo4j_cfg.database, "experiment": cfg.experiment_name},
        )

        llm = RemoteLLM(
            base_url=cfg.llm.base_url,
            api_key=cfg.llm.api_key,
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            timeout_s=cfg.llm.timeout_s,
        )

        retriever = _build_retriever(cfg, embedder, index, mapping, graph, llm)
        reranker = build_reranker(cfg, llm)

        eval_examples = list(dataset.iter_eval_examples())
        logger.info("Eval examples: %d", len(eval_examples))

        qvec_map = build_qvec_map(cfg, eval_examples, cache, logger)

        result = run_eval_loop(
            cfg=cfg, eval_examples=eval_examples, llm=llm, retriever=retriever,
            reranker=reranker, qvec_map=qvec_map, run_dir=run_dir,
            index_time_s=index_time_s, logger=logger,
        )
    finally:
        graph.close()

    metrics, latency, trace_stats = result["metrics"], result["latency"], result["trace_stats"]

    summary = {
        "config": cfg.model_dump(),
        "graph_stats": graph_stats,
        "graph_build_time_s": graph_build_time_s,
        "quality": metrics.mean(),
        "latency": latency.mean(),
        "trace_stats": trace_stats,
    }
    _write_json(run_dir / "summary.json", summary)

    _write_json(
        run_dir / "metrics.json",
        {
            "system": system_name,
            "experiment_name": cfg.experiment_name,
            "llm_model": cfg.llm.model,
            "embed_model": cfg.embeddings.model,
            "quality": {
                "n": summary["quality"]["n"],
                "em": summary["quality"]["em"],
                "f1": summary["quality"]["f1"],
                "accuracy": summary["quality"]["accuracy"],
                "recall": summary["quality"]["recall"],
                "precision": summary["quality"]["precision"],
                "mrr": summary["quality"]["mrr"],
                "ndcg": summary["quality"]["ndcg"],
                "rouge_l": summary["quality"]["rouge_l"],
                "hallucination_rate": summary["quality"].get("hallucination_rate", 0.0),
            },
            "latency": {
                "index_time_s": summary["latency"]["index_time_s"],
                "query_time_s": summary["latency"]["total_query_time_s"],
                "tokens_per_request": summary["latency"]["tokens_per_request"],
                "stage_times": summary["latency"].get("stage_times", {}),
            },
            "graph_stats": graph_stats,
            "graph_build_time_s": graph_build_time_s,
            "trace_stats": trace_stats,
        },
    )

    table = Table(title=f"{dataset.name} GraphRAG (Neo4j, mode={cfg.retrieval.mode}) results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    q = summary["quality"]
    l = summary["latency"]
    table.add_row("N", str(q["n"]))
    table.add_row("EM", f"{q['em']:.4f}")
    table.add_row("F1", f"{q['f1']:.4f}")
    table.add_row("Accuracy", f"{q['accuracy']:.4f}")
    table.add_row("Recall@k (support titles)", f"{q['recall']:.4f}")
    table.add_row("MRR (support titles)", f"{q['mrr']:.4f}")
    table.add_row("Index time (s)", f"{l['index_time_s']:.2f}")
    table.add_row("Graph build time (s)", f"{graph_build_time_s:.2f}")
    table.add_row("Query time avg (s)", f"{l['total_query_time_s']:.2f}")
    table.add_row(
        "Tokens/request avg", f"{l['tokens_per_request']:.2f}" if l["tokens_per_request"] else "n/a"
    )
    table.add_row("Graph entities", str(graph_stats["n_entities"]))
    table.add_row("Graph edges", str(graph_stats["n_edges"]))
    for stage, seconds in l.get("stage_times", {}).items():
        table.add_row(f"Stage: {stage} avg (s)", f"{seconds:.4f}")
    for k, v in trace_stats.items():
        table.add_row(f"trace.{k}", str(v))

    console.print(table)
    console.print(f"[bold]Saved:[/bold] {run_dir / 'summary.json'}")
