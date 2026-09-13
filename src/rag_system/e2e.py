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
from rag_system.llm import RemoteLLM
from rag_system.logging_utils import get_logger, setup_logging
from rag_system.pipeline import (
    build_corpus_chunks, build_embedder, build_qvec_map,
    build_reranker, build_vector_index, run_eval_loop,
)
from rag_system.retrieval import Retriever


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_e2e(config_path: str) -> None:
    """
    End-to-end baseline RAG run:
    1. build index on train corpus
    2. run retrieval + generation on eval examples
    3. compute quality + latency metrics
    4. persist artifacts and metrics JSON
    """

    cfg: RunConfig = load_config(config_path)

    setup_logging(logging.INFO)
    logger = get_logger("rag_system.e2e")
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
    llm = RemoteLLM(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        timeout_s=cfg.llm.timeout_s,
    )

    retriever = Retriever(embedder=embedder, index=index, mapping=mapping)
    reranker = build_reranker(cfg, llm)

    eval_examples = list(dataset.iter_eval_examples())
    logger.info("Eval examples: %d", len(eval_examples))

    qvec_map = build_qvec_map(cfg, eval_examples, cache, logger)

    result = run_eval_loop(
        cfg=cfg, eval_examples=eval_examples, llm=llm, retriever=retriever,
        reranker=reranker, qvec_map=qvec_map, run_dir=run_dir,
        index_time_s=index_time_s, logger=logger,
    )
    metrics, latency, trace_stats = result["metrics"], result["latency"], result["trace_stats"]

    summary = {
        "config": cfg.model_dump(),
        "quality": metrics.mean(),
        "latency": latency.mean(),
        "trace_stats": trace_stats,
    }
    _write_json(run_dir / "summary.json", summary)

    _write_json(
        run_dir / "metrics.json",
        {
            "system": "vector_rag",
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
            },
        },
    )

    table = Table(title=f"{dataset.name} RAG baseline results")
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
    table.add_row("Query time avg (s)", f"{l['total_query_time_s']:.2f}")
    table.add_row(
        "Tokens/request avg", f"{l['tokens_per_request']:.2f}" if l["tokens_per_request"] else "n/a"
    )

    console.print(table)
    console.print(f"[bold]Saved:[/bold] {run_dir / 'summary.json'}")
