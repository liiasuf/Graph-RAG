from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag_system.stats import compute_per_example_stats, significance_label

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
METRICS = ["em", "f1", "recall", "mrr", "hallucination"]


def _find_latest(name: str) -> Path | None:
    base = EXPERIMENTS / name
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "per_example.jsonl").exists()]
    return max(candidates, key=lambda p: os.path.getmtime(p / "per_example.jsonl")) if candidates else None


def _load_jsonl(path: Path) -> list[dict]:
    lines = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def _load_hotpotqa_types(eval_split: str = "validation", limit: int = 500) -> dict[str, str]:
    """Return {qid: type} for HotpotQA (bridge / comparison)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "distractor", split=eval_split)
        result = {}
        for row in ds:
            result[str(row["id"])] = str(row.get("type", "unknown")).lower()
            if len(result) >= limit:
                break
        return result
    except Exception:
        return {}


def win_loss_tie(base: list[dict], graph: list[dict], metric: str = "em") -> dict:
    base_map = {d["id"]: d for d in base}
    graph_map = {d["id"]: d for d in graph}
    common = sorted(set(base_map) & set(graph_map))

    wins, losses, ties = [], [], []
    for qid in common:
        bv = base_map[qid].get(metric, 0.0)
        gv = graph_map[qid].get(metric, 0.0)
        if gv > bv:
            wins.append(qid)
        elif gv < bv:
            losses.append(qid)
        else:
            ties.append(qid)
    return {"wins": wins, "losses": losses, "ties": ties, "base_map": base_map, "graph_map": graph_map}


def per_category_stats(
    base: list[dict],
    graph: list[dict],
    qtype_map: dict[str, str],
    metrics: list[str],
) -> dict[str, dict]:
    """Break down metrics by question type."""
    from collections import defaultdict

    base_map = {d["id"]: d for d in base}
    graph_map = {d["id"]: d for d in graph}
    common = sorted(set(base_map) & set(graph_map))

    by_type: dict[str, list[str]] = defaultdict(list)
    for qid in common:
        qtype = qtype_map.get(qid, "unknown")
        by_type[qtype].append(qid)

    result = {}
    for qtype, ids in by_type.items():
        result[qtype] = {}
        for m in metrics:
            bvals = [base_map[i].get(m, 0.0) for i in ids]
            gvals = [graph_map[i].get(m, 0.0) for i in ids]
            n = len(ids)
            result[qtype][m] = {
                "n": n,
                "baseline": round(sum(bvals) / n, 4) if n else 0,
                "graphrag": round(sum(gvals) / n, 4) if n else 0,
                "delta": round((sum(gvals) - sum(bvals)) / n, 4) if n else 0,
            }
    return result


def print_significance_table(console: Console, stats: list[dict]) -> None:
    table = Table(title="Statistical Significance (paired t-test + bootstrap 95% CI)")
    table.add_column("Metric")
    table.add_column("Baseline mean [95% CI]", justify="right")
    table.add_column("GraphRAG mean [95% CI]", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("sig", justify="center")
    table.add_column("Cohen's d", justify="right")

    for s in stats:
        b = s["baseline"]
        g = s["graphrag"]
        delta = g["mean"] - b["mean"]
        sign = "+" if delta >= 0 else ""
        table.add_row(
            s["metric"],
            f"{b['mean']:.4f} [{b['ci95_lo']:.4f}, {b['ci95_hi']:.4f}]",
            f"{g['mean']:.4f} [{g['ci95_lo']:.4f}, {g['ci95_hi']:.4f}]",
            f"{sign}{delta:.4f}",
            f"{s['p_value']:.4f}",
            s["significance"],
            f"{s['cohen_d']:.3f}",
        )
    console.print(table)


def print_category_table(console: Console, cat_stats: dict, metric: str = "f1") -> None:
    table = Table(title=f"Per-category breakdown: {metric}")
    table.add_column("Question type")
    table.add_column("N", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("GraphRAG", justify="right")
    table.add_column("Δ", justify="right")

    for qtype, metrics_data in sorted(cat_stats.items()):
        m = metrics_data.get(metric, {})
        delta = m.get("delta", 0)
        sign = "+" if delta >= 0 else ""
        table.add_row(
            qtype,
            str(m.get("n", 0)),
            f"{m.get('baseline', 0):.4f}",
            f"{m.get('graphrag', 0):.4f}",
            f"{sign}{delta:.4f}",
        )
    console.print(table)


def print_win_loss(console: Console, wlt: dict, n_examples: int = 3) -> None:
    wins = wlt["wins"][:n_examples]
    losses = wlt["losses"][:n_examples]
    base_map = wlt["base_map"]
    graph_map = wlt["graph_map"]

    console.print(f"\n[bold green]GraphRAG wins:[/bold green] {len(wlt['wins'])}  "
                  f"[bold red]losses:[/bold red] {len(wlt['losses'])}  "
                  f"[bold]ties:[/bold] {len(wlt['ties'])}")

    if wins:
        table = Table(title=f"GraphRAG WIN examples (EM: 0->1), top {n_examples}")
        table.add_column("Question", max_width=60)
        table.add_column("Gold")
        table.add_column("Baseline pred")
        table.add_column("GraphRAG pred")
        for qid in wins:
            b = base_map[qid]
            g = graph_map[qid]
            table.add_row(b["question"][:60], b["gold"], b["pred"][:30], g["pred"][:30])
        console.print(table)

    if losses:
        table = Table(title=f"GraphRAG LOSS examples (EM: 1->0), top {n_examples}")
        table.add_column("Question", max_width=60)
        table.add_column("Gold")
        table.add_column("Baseline pred")
        table.add_column("GraphRAG pred")
        for qid in losses:
            b = base_map[qid]
            g = graph_map[qid]
            table.add_row(b["question"][:60], b["gold"], b["pred"][:30], g["pred"][:30])
        console.print(table)


def write_report(
    out_dir: Path,
    sig_stats: list[dict],
    cat_stats: dict,
    wlt: dict,
    baseline_run: str,
    graphrag_run: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}_analysis.md"

    lines = [
        "# GraphRAG vs RAG - Deep Analysis",
        "",
        f"- Baseline: `{baseline_run}`",
        f"- GraphRAG: `{graphrag_run}`",
        f"- Generated: {datetime.now().isoformat()}",
        "",
        "## 1. Statistical Significance",
        "",
        "| Metric | Baseline [95% CI] | GraphRAG [95% CI] | Δ | p-value | sig | Cohen's d |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sig_stats:
        b, g = s["baseline"], s["graphrag"]
        delta = g["mean"] - b["mean"]
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {s['metric']} "
            f"| {b['mean']:.4f} [{b['ci95_lo']:.4f}, {b['ci95_hi']:.4f}] "
            f"| {g['mean']:.4f} [{g['ci95_lo']:.4f}, {g['ci95_hi']:.4f}] "
            f"| {sign}{delta:.4f} "
            f"| {s['p_value']:.4f} "
            f"| {s['significance']} "
            f"| {s['cohen_d']:.3f} |"
        )

    lines += [
        "",
        "> \\* p<0.05  \\*\\* p<0.01  \\*\\*\\* p<0.001  ns = not significant",
        "",
        "## 2. Per Question-Type Breakdown",
        "",
    ]
    for metric in ["em", "f1", "recall"]:
        lines += [
            f"### {metric.upper()}",
            "",
            "| Type | N | Baseline | GraphRAG | Δ |",
            "|---|---|---|---|---|",
        ]
        for qtype, md in sorted(cat_stats.items()):
            m = md.get(metric, {})
            d = m.get("delta", 0)
            sign = "+" if d >= 0 else ""
            lines.append(
                f"| {qtype} | {m.get('n',0)} "
                f"| {m.get('baseline',0):.4f} "
                f"| {m.get('graphrag',0):.4f} "
                f"| {sign}{d:.4f} |"
            )
        lines.append("")

    lines += [
        "## 3. Win / Loss / Tie (EM)",
        "",
        f"- GraphRAG wins: **{len(wlt['wins'])}**",
        f"- GraphRAG losses: **{len(wlt['losses'])}**",
        f"- Ties: **{len(wlt['ties'])}**",
        "",
        "### Top GraphRAG WIN examples",
        "",
        "| Question | Gold | Baseline pred | GraphRAG pred |",
        "|---|---|---|---|",
    ]
    base_map = wlt["base_map"]
    graph_map = wlt["graph_map"]
    for qid in wlt["wins"][:5]:
        b, g = base_map[qid], graph_map[qid]
        lines.append(f"| {b['question'][:80]} | {b['gold']} | {b['pred'][:40]} | {g['pred'][:40]} |")

    lines += [
        "",
        "### Top GraphRAG LOSS examples",
        "",
        "| Question | Gold | Baseline pred | GraphRAG pred |",
        "|---|---|---|---|",
    ]
    for qid in wlt["losses"][:5]:
        b, g = base_map[qid], graph_map[qid]
        lines.append(f"| {b['question'][:80]} | {b['gold']} | {b['pred'][:40]} | {g['pred'][:40]} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="hotpotqa_rag_baseline")
    parser.add_argument("--graphrag", default="hotpotqa_graphrag")
    parser.add_argument("--baseline-run", default=None)
    parser.add_argument("--graphrag-run", default=None)
    parser.add_argument("--dataset", default="hotpotqa", help="For per-category type loading")
    args = parser.parse_args()

    console = Console()

    baseline_run = Path(args.baseline_run) if args.baseline_run else _find_latest(args.baseline)
    graphrag_run = Path(args.graphrag_run) if args.graphrag_run else _find_latest(args.graphrag)

    if baseline_run is None or graphrag_run is None:
        raise SystemExit("Could not find run directories. Run experiments first.")

    base_examples = _load_jsonl(baseline_run / "per_example.jsonl")
    graph_examples = _load_jsonl(graphrag_run / "per_example.jsonl")

    console.print(f"[bold]Baseline:[/bold] {baseline_run} ({len(base_examples)} examples)")
    console.print(f"[bold]GraphRAG:[/bold] {graphrag_run} ({len(graph_examples)} examples)")

    # 1. Statistical significance
    sig_stats = []
    for metric in METRICS:
        if any(metric in d for d in base_examples):
            s = compute_per_example_stats(base_examples, graph_examples, metric=metric)
            sig_stats.append(s)
    print_significance_table(console, sig_stats)

    # 2. Per-category breakdown (HotpotQA only for now)
    qtype_map = {}
    if args.dataset == "hotpotqa":
        console.print("\n[dim]Loading HotpotQA question types...[/dim]")
        qtype_map = _load_hotpotqa_types(limit=max(len(base_examples), 500))

    if qtype_map:
        cat_stats = per_category_stats(base_examples, graph_examples, qtype_map, METRICS)
        for m in ["em", "f1", "recall"]:
            print_category_table(console, cat_stats, metric=m)
    else:
        cat_stats = {}
        console.print("[yellow]No question-type mapping - skipping per-category breakdown[/yellow]")

    # 3. Win/Loss/Tie
    wlt = win_loss_tie(base_examples, graph_examples, metric="em")
    print_win_loss(console, wlt)

    # Save report
    report = write_report(
        EXPERIMENTS / "analysis",
        sig_stats, cat_stats, wlt,
        str(baseline_run), str(graphrag_run),
    )
    console.print(f"\n[bold]Saved analysis:[/bold] {report}")
