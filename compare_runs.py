from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT_PATH = Path(__file__).resolve().parent
EXPERIMENTS_BASE_PATH = ROOT_PATH / "experiments"


def _find_latest_run_dir(experiment_name: str) -> Path | None:
    base = EXPERIMENTS_BASE_PATH / experiment_name
    if not base.exists():
        return None

    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    if not candidates:
        return None

    return max(candidates, key=lambda p: os.path.getmtime(p / "metrics.json"))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _delta(a, b, digits=4):
    if a is None or b is None:
        return "n/a"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.{digits}f}"


def compare(baseline_run: Path, graphrag_run: Path) -> dict:
    base_metrics = _load_json(baseline_run / "metrics.json")
    graph_metrics = _load_json(graphrag_run / "metrics.json")

    bq, gq = base_metrics["quality"], graph_metrics["quality"]
    bl, gl = base_metrics["latency"], graph_metrics["latency"]

    rows = [
        ("Experiment (system)", base_metrics.get("system", "n/a"), graph_metrics.get("system", "n/a"), None, 0),
        ("LLM model", base_metrics.get("llm_model", "n/a"), graph_metrics.get("llm_model", "n/a"), None, 0),
        ("Embed model", base_metrics.get("embed_model", "n/a"), graph_metrics.get("embed_model", "n/a"), None, 0),
        ("N (eval examples)", bq.get("n", "n/a"), gq.get("n", "n/a"), None, 0),
        ("EM", bq["em"], gq["em"], _delta(bq["em"], gq["em"]), 4),
        ("F1", bq["f1"], gq["f1"], _delta(bq["f1"], gq["f1"]), 4),
        ("Accuracy", bq["accuracy"], gq["accuracy"], _delta(bq["accuracy"], gq["accuracy"]), 4),
        (
            "Recall@k (support titles)",
            bq["recall"],
            gq["recall"],
            _delta(bq["recall"], gq["recall"]),
            4,
        ),
        ("MRR (support titles)", bq["mrr"], gq["mrr"], _delta(bq["mrr"], gq["mrr"]), 4),
        (
            "Hallucination rate ↓",
            bq.get("hallucination_rate"),
            gq.get("hallucination_rate"),
            _delta(bq.get("hallucination_rate", 0), gq.get("hallucination_rate", 0)),
            4,
        ),
        (
            "Index time (s)",
            bl["index_time_s"],
            gl["index_time_s"],
            _delta(bl["index_time_s"], gl["index_time_s"], 2),
            2,
        ),
        (
            "Query time avg (s)",
            bl["query_time_s"],
            gl["query_time_s"],
            _delta(bl["query_time_s"], gl["query_time_s"], 4),
            4,
        ),
        (
            "Tokens/request avg",
            bl["tokens_per_request"],
            gl["tokens_per_request"],
            _delta(bl["tokens_per_request"], gl["tokens_per_request"], 1),
            1,
        ),
    ]

    graph_stats = graph_metrics.get("graph_stats")
    graph_build_time_s = graph_metrics.get("graph_build_time_s")

    return {
        "baseline_run": str(baseline_run),
        "graphrag_run": str(graphrag_run),
        "rows": rows,
        "graph_stats": graph_stats,
        "graph_build_time_s": graph_build_time_s,
    }


def print_table(console: Console, result: dict) -> None:
    table = Table(title="RAG vs GraphRAG comparison")
    table.add_column("Metric")
    table.add_column("RAG (baseline)", justify="right")
    table.add_column("GraphRAG", justify="right")
    table.add_column("Δ (graph - baseline)", justify="right")

    for name, base_val, graph_val, delta, digits in result["rows"]:
        table.add_row(name, _fmt(base_val, digits), _fmt(graph_val, digits), delta or "")

    console.print(table)

    if result["graph_stats"]:
        gs = result["graph_stats"]
        console.print(
            f"[bold]Graph:[/bold] {gs['n_entities']} entities, {gs['n_edges']} edges over "
            f"{gs['n_chunks']} chunks "
            f"(build time: {result['graph_build_time_s']:.2f}s)"
        )


def write_markdown_report(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_comparison.md"

    lines = [
        "# RAG vs GraphRAG comparison",
        "",
        f"- Baseline run: `{result['baseline_run']}`",
        f"- GraphRAG run: `{result['graphrag_run']}`",
        "",
        "| Metric | RAG (baseline) | GraphRAG | Δ (graph - baseline) |",
        "|---|---|---|---|",
    ]

    for name, base_val, graph_val, delta, digits in result["rows"]:
        lines.append(f"| {name} | {_fmt(base_val, digits)} | {_fmt(graph_val, digits)} | {delta or ''} |")

    if result["graph_stats"]:
        gs = result["graph_stats"]
        lines += [
            "",
            "## Graph stats",
            "",
            f"- Entities: {gs['n_entities']}",
            f"- Edges: {gs['n_edges']}",
            f"- Chunks covered: {gs['n_chunks']}",
            f"- Graph build time: {result['graph_build_time_s']:.2f}s",
        ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ── N-way comparison (any number of systems side by side) ──────────────────
# Every metrics.json (vector_rag / graph_rag / graph_rag_neo4j /
# graph_rag_neo4j_only / graph_rag_neo4j_rerank / adaptive) shares the same
# quality/latency schema (pipeline.run_eval_loop writes it uniformly), so
# comparing N of them is just loading N metrics.json files and laying out
# one column per system - no special-casing per mode.

_WIDE_ROWS = [
    ("Experiment (system)", ("system",), 0),
    ("Experiment name", ("experiment_name",), 0),
    ("LLM model", ("llm_model",), 0),
    ("Embed model", ("embed_model",), 0),
    ("N", ("quality", "n"), 0),
    ("EM", ("quality", "em"), 4),
    ("F1", ("quality", "f1"), 4),
    ("Accuracy", ("quality", "accuracy"), 4),
    ("Recall@k", ("quality", "recall"), 4),
    ("Precision@k", ("quality", "precision"), 4),
    ("MRR", ("quality", "mrr"), 4),
    ("NDCG@k", ("quality", "ndcg"), 4),
    ("ROUGE-L", ("quality", "rouge_l"), 4),
    ("Hallucination ↓", ("quality", "hallucination_rate"), 4),
    ("Index time (s)", ("latency", "index_time_s"), 2),
    ("Query time avg (s)", ("latency", "query_time_s"), 4),
    ("Tokens/request avg", ("latency", "tokens_per_request"), 1),
]


def _get(d: dict, path: tuple[str, ...]):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _dataset_name(run_dir: Path | None) -> str | None:
    """Read dataset.name out of that run's summary.json (metrics.json doesn't carry it)."""
    if run_dir is None:
        return None
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return _load_json(summary_path)["config"]["dataset"]["name"]
    except (KeyError, json.JSONDecodeError):
        return None


def compare_many(experiment_names: list[str]) -> dict:
    """Load metrics.json for each experiment_name and lay out one column per system."""
    systems = {}
    datasets = {}
    for name in experiment_names:
        run = _find_latest_run_dir(name)
        systems[name] = _load_json(run / "metrics.json") if run else None
        datasets[name] = _dataset_name(run)

    rows = []
    for label, path, digits in _WIDE_ROWS:
        rows.append((label, [(_get(systems[n], path) if systems[n] else None) for n in experiment_names], digits))

    return {"experiment_names": experiment_names, "systems": systems, "datasets": datasets, "rows": rows}


def _column_header(name: str, m: dict | None, dataset: str | None) -> str:
    system = m.get("system", name) if m else f"{name} (missing)"
    return f"{dataset} / {system}" if dataset else system


def print_wide_table(console: Console, result: dict) -> None:
    table = Table(title="RAG system comparison")
    table.add_column("Metric")
    for name, m in result["systems"].items():
        table.add_column(_column_header(name, m, result["datasets"].get(name)), justify="right")

    for label, values, digits in result["rows"]:
        table.add_row(label, *[_fmt(v, digits) for v in values])

    console.print(table)


def write_wide_markdown_report(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_comparison_wide.md"

    headers = [
        _column_header(n, m, result["datasets"].get(n)) for n, m in result["systems"].items()
    ]
    lines = [
        "# RAG system comparison",
        "",
        "| Metric | " + " | ".join(headers) + " |",
        "|---|" + "---|" * len(headers),
    ]
    for label, values, digits in result["rows"]:
        lines.append(f"| {label} | " + " | ".join(_fmt(v, digits) for v in values) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare RAG system runs")
    parser.add_argument("--baseline", default="hotpotqa_rag_baseline", help="Baseline experiment name")
    parser.add_argument("--graphrag", default="hotpotqa_graphrag", help="GraphRAG experiment name")
    parser.add_argument("--baseline-run", default=None, help="Explicit path to a baseline run dir")
    parser.add_argument("--graphrag-run", default=None, help="Explicit path to a GraphRAG run dir")
    parser.add_argument(
        "--systems", default=None,
        help="Comma-separated experiment_names for an N-way comparison instead of the "
             "pairwise baseline/graphrag mode, e.g. "
             "hotpotqa_rag_baseline,hotpotqa_graph_overlay,"
             "hotpotqa_graph_only,hotpotqa_graph_rerank,hotpotqa_adaptive",
    )
    args = parser.parse_args()

    console = Console()

    if args.systems:
        names = [n.strip() for n in args.systems.split(",") if n.strip()]
        result = compare_many(names)
        print_wide_table(console, result)
        report_path = write_wide_markdown_report(result, EXPERIMENTS_BASE_PATH / "comparisons")
        console.print(f"[bold]Saved report:[/bold] {report_path}")
        raise SystemExit(0)

    baseline_run = Path(args.baseline_run) if args.baseline_run else _find_latest_run_dir(args.baseline)
    graphrag_run = Path(args.graphrag_run) if args.graphrag_run else _find_latest_run_dir(args.graphrag)

    if baseline_run is None:
        raise SystemExit(
            f"No run with metrics.json found for experiment '{args.baseline}'. "
            f"Run python run_e2e.py first."
        )
    if graphrag_run is None:
        raise SystemExit(
            f"No run with metrics.json found for experiment '{args.graphrag}'. "
            f"Run python run_e2e_graphrag_neo4j.py first."
        )

    result = compare(baseline_run, graphrag_run)
    print_table(console, result)

    report_path = write_markdown_report(result, EXPERIMENTS_BASE_PATH / "comparisons")
    console.print(f"[bold]Saved report:[/bold] {report_path}")
