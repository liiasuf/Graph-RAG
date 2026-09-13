from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"

# (experiment_name, system_label, dataset_label)
EXPERIMENT_PAIRS = [
    ("hotpotqa_rag_baseline",   "RAG",  "HotpotQA"),
    ("hotpotqa_graphrag",       "GraphRAG",    "HotpotQA"),
    ("musique_rag_baseline",    "RAG",  "MuSiQue"),
    ("musique_graphrag",        "GraphRAG",    "MuSiQue"),
    ("wiki2multihop_rag_baseline", "RAG", "2WikiMultiHop"),
    ("wiki2multihop_graphrag",     "GraphRAG",   "2WikiMultiHop"),
    ("nq_rag_baseline",         "RAG",  "NQ (single-hop)"),
    ("nq_graphrag",             "GraphRAG",    "NQ (single-hop)"),
    ("triviaqa_rag_baseline",   "RAG",  "TriviaQA (single-hop)"),
    ("triviaqa_graphrag",       "GraphRAG",    "TriviaQA (single-hop)"),
]


def _find_latest(name: str) -> Path | None:
    base = EXPERIMENTS / name
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    return max(candidates, key=lambda p: os.path.getmtime(p / "metrics.json")) if candidates else None


def _load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _fmt(v, digits=4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def build_summary() -> list[dict]:
    rows = []
    for exp_name, system, dataset in EXPERIMENT_PAIRS:
        run_dir = _find_latest(exp_name)
        if run_dir is None:
            rows.append({
                "dataset": dataset, "system": system,
                "em": None, "f1": None, "recall": None,
                "mrr": None, "hallucination_rate": None,
                "query_time_s": None, "n": None,
            })
            continue
        m = _load_metrics(run_dir)
        q = m.get("quality", {})
        l = m.get("latency", {})
        rows.append({
            "dataset": dataset,
            "system": system,
            "em": q.get("em"),
            "f1": q.get("f1"),
            "recall": q.get("recall"),
            "mrr": q.get("mrr"),
            "hallucination_rate": q.get("hallucination_rate"),
            "query_time_s": l.get("query_time_s"),
            "n": q.get("n"),
        })
    return rows


def print_summary_table(console: Console, rows: list[dict]) -> None:
    table = Table(title="RAG vs GraphRAG - All Datasets Summary")
    table.add_column("Dataset")
    table.add_column("System")
    table.add_column("N", justify="right")
    table.add_column("EM", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Recall@k", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Halluc↓", justify="right")
    table.add_column("Query(s)", justify="right")

    prev_dataset = None
    for r in rows:
        dataset = r["dataset"] if r["dataset"] != prev_dataset else ""
        prev_dataset = r["dataset"]
        table.add_row(
            dataset, r["system"],
            _fmt(r["n"], 0),
            _fmt(r["em"]),
            _fmt(r["f1"]),
            _fmt(r["recall"]),
            _fmt(r["mrr"]),
            _fmt(r["hallucination_rate"]),
            _fmt(r["query_time_s"], 2),
        )
    console.print(table)


def write_markdown(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# RAG vs GraphRAG - Full Results Table",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Dataset | System | N | EM | F1 | Recall@k | MRR | Halluc↓ | Query(s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['system']} "
            f"| {_fmt(r['n'], 0)} "
            f"| {_fmt(r['em'])} "
            f"| {_fmt(r['f1'])} "
            f"| {_fmt(r['recall'])} "
            f"| {_fmt(r['mrr'])} "
            f"| {_fmt(r['hallucination_rate'])} "
            f"| {_fmt(r['query_time_s'], 2)} |"
        )

    lines += [
        "",
        "**Note:** Halluc↓ = hallucination rate (lower is better). "
        "- = experiment not yet run.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Output markdown path")
    args = parser.parse_args()

    console = Console()
    rows = build_summary()
    print_summary_table(console, rows)

    out = Path(args.output) if args.output else EXPERIMENTS / "paper_table.md"
    write_markdown(rows, out)
    console.print(f"\n[bold]Saved:[/bold] {out}")
