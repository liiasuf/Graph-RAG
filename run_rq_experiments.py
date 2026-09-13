from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
EXPERIMENTS = ROOT / "experiments"
MODEL_SWEEP = EXPERIMENTS / "model_sweep"
MODEL = os.environ.get("RQ_MODEL", "qwen3.8")

DATASETS = {
    "hotpotqa":      {"label": "HotpotQA",       "type": "multi-hop",  "base": "hotpotqa_rag_baseline",      "graph": "hotpotqa_graph_rerank"},
    "musique":       {"label": "MuSiQue",         "type": "multi-hop",  "base": "musique_rag_baseline",       "graph": "musique_graph_rerank"},
    "wiki2multihop": {"label": "2WikiMultiHop",   "type": "multi-hop",  "base": "wiki2multihop_rag_baseline", "graph": "wiki2multihop_graph_rerank"},
    "nq":            {"label": "NQ",              "type": "single-hop", "base": "nq_rag_baseline",            "graph": "nq_graph_rerank"},
    "triviaqa":      {"label": "TriviaQA",        "type": "single-hop", "base": "triviaqa_rag_baseline",      "graph": "triviaqa_graph_rerank"},
}

ALL_SYSTEMS = ["rag_baseline", "graph_only", "graph_overlay", "graph_rerank"]
ALL_MODELS = ["qwen3.8", "gemma-4-26B-A4B-it", "qwen3-vl-30b"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_latest(exp_name: str, model: str = MODEL) -> Path | None:
    base = MODEL_SWEEP / f"{exp_name}__{model}"
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    return max(candidates, key=lambda p: os.path.getmtime(p / "metrics.json")) if candidates else None


def _load_metrics(exp_name: str, model: str = MODEL) -> dict | None:
    run = _find_latest(exp_name, model)
    return json.loads((run / "metrics.json").read_text()) if run else None


def _load_per_example(exp_name: str) -> list[dict]:
    run = _find_latest(exp_name)
    if not run:
        return []
    path = run / "per_example.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _delta(base: float, graph: float) -> str:
    d = graph - base
    sign = "+" if d >= 0 else ""
    arrow = "↑" if d > 0.001 else ("↓" if d < -0.001 else "=")
    return f"{sign}{d:.4f} {arrow}"


def _fmt(v) -> str:
    if v is None or v == "n/a":
        return "-"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


# ── RQ1: Retrieval quality ────────────────────────────────────────────────────

def rq1_retrieval(lines: list[str]) -> None:
    lines.append("## RQ1 - Retrieval Quality\n")
    lines.append(
        "> Does GraphRAG improve retrieval over RAG?\n"
        "> Metrics: Recall@k, Precision@k, MRR, NDCG@k\n"
        "> (methodology: LinkedIn GraphRAG, Barry et al. 2404.17723)\n"
    )
    header = "| Dataset | Type | System | Recall@k | Precision@k | MRR | NDCG@k |"
    sep    = "|---------|------|--------|----------|-------------|-----|--------|"
    lines += [header, sep]

    for ds, info in DATASETS.items():
        bm = _load_metrics(info["base"])
        gm = _load_metrics(info["graph"])
        for label, m in [("RAG", bm), ("GraphRAG", gm)]:
            if m is None:
                lines.append(f"| {info['label']} | {info['type']} | {label} | - | - | - | - |")
                continue
            q = m["quality"]
            lines.append(
                f"| {info['label']} | {info['type']} | {label} "
                f"| {_fmt(q.get('recall'))} "
                f"| {_fmt(q.get('precision'))} "
                f"| {_fmt(q.get('mrr'))} "
                f"| {_fmt(q.get('ndcg'))} |"
            )
        # delta row
        if bm and gm:
            bq, gq = bm["quality"], gm["quality"]
            lines.append(
                f"| | | **Δ (Graph−Base)** "
                f"| **{_delta(bq.get('recall',0), gq.get('recall',0))}** "
                f"| **{_delta(bq.get('precision',0), gq.get('precision',0))}** "
                f"| **{_delta(bq.get('mrr',0), gq.get('mrr',0))}** "
                f"| **{_delta(bq.get('ndcg',0), gq.get('ndcg',0))}** |"
            )
        lines.append("")

    lines.append(
        "\n**Finding:** the hypothesis that graph traversal helps mainly on multi-hop "
        "datasets does not hold as stated. Multi-hop gains from graph_rerank are small "
        "(+0.5-1.9pp Recall@k on HotpotQA/MuSiQue/2WikiMultiHop). The single-hop NQ "
        "benchmark instead shows the largest gain by far (Recall@k +0.41, NDCG@k +0.31) "
        "- graph traversal over MRQA/NQ's short single-paragraph contexts finds the gold "
        "passage far more reliably than plain cosine search. TriviaQA (single-hop) shows "
        "no gain, so the effect is dataset-specific, not a clean multi-hop/single-hop "
        "split. See `experiments/model_sweep/model_sweep_report.md` for the granular "
        "per-method (vector/graph_only/overlay/rerank) breakdown behind these deltas.\n"
    )


# ── RQ2: Answer quality ───────────────────────────────────────────────────────

def rq2_answer_quality(lines: list[str]) -> None:
    lines.append("## RQ2 - Answer Quality\n")
    lines.append(
        "> Does better retrieval translate to better answers?\n"
        "> Metrics: EM, F1, ROUGE-L\n"
        "> (methodology: GCR, 2410.13080, Table 1; LinkedIn GraphRAG, 2404.17723)\n"
    )
    header = "| Dataset | Type | System | EM | F1 | ROUGE-L |"
    sep    = "|---------|------|--------|----|----|---------|"
    lines += [header, sep]

    for ds, info in DATASETS.items():
        bm = _load_metrics(info["base"])
        gm = _load_metrics(info["graph"])
        for label, m in [("RAG", bm), ("GraphRAG", gm)]:
            if m is None:
                lines.append(f"| {info['label']} | {info['type']} | {label} | - | - | - |")
                continue
            q = m["quality"]
            lines.append(
                f"| {info['label']} | {info['type']} | {label} "
                f"| {_fmt(q.get('em'))} "
                f"| {_fmt(q.get('f1'))} "
                f"| {_fmt(q.get('rouge_l'))} |"
            )
        if bm and gm:
            bq, gq = bm["quality"], gm["quality"]
            lines.append(
                f"| | | **Δ** "
                f"| **{_delta(bq.get('em',0), gq.get('em',0))}** "
                f"| **{_delta(bq.get('f1',0), gq.get('f1',0))}** "
                f"| **{_delta(bq.get('rouge_l',0), gq.get('rouge_l',0))}** |"
            )
        lines.append("")


# ── RQ3: Hallucination ────────────────────────────────────────────────────────

def rq3_hallucination(lines: list[str]) -> None:
    lines.append("## RQ3 - Hallucination Rate\n")
    lines.append(
        "> Does GraphRAG reduce hallucination?\n"
        "> Metric: hallucination_rate (token-grounding proxy; 1.0 = hallucinated)\n"
        "> (methodology: GCR 2410.13080 Table 5; Barry et al. 2025)\n"
    )
    header = "| Dataset | Type | RAG | GraphRAG | Δ |"
    sep    = "|---------|------|------------|----------|---|"
    lines += [header, sep]

    for ds, info in DATASETS.items():
        bm = _load_metrics(info["base"])
        gm = _load_metrics(info["graph"])
        bv = bm["quality"].get("hallucination_rate") if bm else None
        gv = gm["quality"].get("hallucination_rate") if gm else None
        d = _delta(bv or 0, gv or 0) if (bv is not None and gv is not None) else "-"
        lines.append(f"| {info['label']} | {info['type']} | {_fmt(bv)} | {_fmt(gv)} | {d} |")

    lines.append("\n**Note:** Lower is better for hallucination_rate.\n")


# ── RQ4: Dataset type breakdown ───────────────────────────────────────────────

def rq4_dataset_breakdown(lines: list[str]) -> None:
    lines.append("## RQ4 - Multi-hop vs Single-hop Breakdown\n")
    lines.append(
        "> Does GraphRAG help more on multi-hop questions?\n"
        "> Aggregated over dataset type (multi-hop: HotpotQA, MuSiQue, 2Wiki; single-hop: NQ, TriviaQA)\n"
    )

    for dtype in ["multi-hop", "single-hop"]:
        ds_list = [info for info in DATASETS.values() if info["type"] == dtype]
        lines.append(f"### {dtype.capitalize()} datasets\n")
        header = "| Dataset | EM Δ | F1 Δ | Recall Δ | NDCG Δ | Halluc Δ |"
        sep    = "|---------|------|------|----------|--------|----------|"
        lines += [header, sep]

        for info in ds_list:
            bm = _load_metrics(info["base"])
            gm = _load_metrics(info["graph"])
            if not bm or not gm:
                lines.append(f"| {info['label']} | - | - | - | - | - |")
                continue
            bq, gq = bm["quality"], gm["quality"]
            lines.append(
                f"| {info['label']} "
                f"| {_delta(bq.get('em',0), gq.get('em',0))} "
                f"| {_delta(bq.get('f1',0), gq.get('f1',0))} "
                f"| {_delta(bq.get('recall',0), gq.get('recall',0))} "
                f"| {_delta(bq.get('ndcg',0), gq.get('ndcg',0))} "
                f"| {_delta(gq.get('hallucination_rate',0), bq.get('hallucination_rate',0))} |"
            )
        lines.append("")


# ── RQ5: Efficiency ───────────────────────────────────────────────────────────

def rq5_efficiency(lines: list[str]) -> None:
    lines.append("## RQ5 - Computational Overhead\n")
    lines.append(
        "> What is the latency cost of adding the knowledge graph?\n"
        "> (methodology: GCR 2410.13080, Table 2)\n"
    )
    header = "| Dataset | System | Index (s) | Graph build (s) | Query avg (s) | Tokens/req |"
    sep    = "|---------|--------|-----------|-----------------|---------------|------------|"
    lines += [header, sep]

    for ds, info in DATASETS.items():
        bm = _load_metrics(info["base"])
        gm = _load_metrics(info["graph"])
        for label, m in [("RAG", bm), ("GraphRAG", gm)]:
            if m is None:
                lines.append(f"| {info['label']} | {label} | - | - | - | - |")
                continue
            lat = m.get("latency", {})
            gb = _fmt(m.get("graph_build_time_s")) if label == "GraphRAG" else "-"
            tok = lat.get("tokens_per_request")
            lines.append(
                f"| {info['label']} | {label} "
                f"| {_fmt(lat.get('index_time_s'))} "
                f"| {gb} "
                f"| {_fmt(lat.get('query_time_s'))} "
                f"| {_fmt(tok) if tok else '-'} |"
            )
        lines.append("")

    lines.append(
        "**Note:** graph build (Neo4j entity/edge write, one-time offline cost) ranges "
        "1.9s (NQ, small entity count) to ~287s (HotpotQA); it does not need to repeat "
        "per query. Per-query time is dominated by LLM generation, not graph traversal - "
        "GraphRAG's query time is comparable to or faster than plain RAG's in this table.\n"
    )


# ── RQ6: Neo4j retrieval-mode comparison ──────────────────────────────────────

def rq6_neo4j_backends(lines: list[str]) -> None:
    lines.append("## RQ6 - Neo4j Retrieval Modes, Averaged Across All 5 Datasets\n")
    lines.append(
        "> vector vs graph-only vs overlay vs graph-as-reranker, x 3 chat models -\n"
        "> which Neo4j-backed strategy wins on quality/latency? n=500/dataset, uncapped\n"
        "> corpus. Per-dataset breakdown: `experiments/model_sweep/model_sweep_report.md`.\n"
    )
    header = "| System | Model | EM | F1 | Recall@k | NDCG@k | MRR | Halluc↓ | Query time avg (s) |"
    sep    = "|--------|-------|----|----|----------|--------|-----|---------|--------------------|"
    lines += [header, sep]

    for system in ALL_SYSTEMS:
        for model in ALL_MODELS:
            keys = ["em", "f1", "recall", "ndcg", "mrr", "hallucination_rate"]
            sums = dict.fromkeys(keys, 0.0)
            qt_sum, n = 0.0, 0
            for info in DATASETS.values():
                exp_name = info["base"].rsplit("_rag_baseline", 1)[0] + "_" + system if system != "rag_baseline" else info["base"]
                m = _load_metrics(exp_name, model)
                if m is None:
                    continue
                q, lat = m["quality"], m.get("latency", {})
                for k in keys:
                    sums[k] += q.get(k) or 0.0
                qt_sum += lat.get("query_time_s") or 0.0
                n += 1
            if n == 0:
                lines.append(f"| {system} | {model} | - | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| {system} | {model} "
                f"| {_fmt(sums['em']/n)} "
                f"| {_fmt(sums['f1']/n)} "
                f"| {_fmt(sums['recall']/n)} "
                f"| {_fmt(sums['ndcg']/n)} "
                f"| {_fmt(sums['mrr']/n)} "
                f"| {_fmt(sums['hallucination_rate']/n)} "
                f"| {_fmt(qt_sum/n)} |"
            )

    lines.append(
        "\n**Note:** averages over HotpotQA/MuSiQue/2WikiMultiHop/NQ/TriviaQA "
        "(n=500 each, real e2e runs against Neo4j 5). `adaptive` (router + "
        "subquestion decomposition) is excluded from this sweep by design and "
        "evaluated separately (MVP-scale, `configs/e2e_hotpotqa_adaptive_smoke.yaml`).\n"
    )


# ── Win/Loss/Tie summary ──────────────────────────────────────────────────────

def win_loss_tie(lines: list[str]) -> None:
    lines.append("## Win / Loss / Tie Analysis (EM)\n")
    lines.append("Counts per dataset: how often GraphRAG is better/worse/equal vs RAG.\n")
    header = "| Dataset | Type | Wins (Graph) | Losses (Graph) | Ties | Win% |"
    sep    = "|---------|------|-------------|----------------|------|------|"
    lines += [header, sep]

    for ds, info in DATASETS.items():
        base_ex = {e["id"]: e for e in _load_per_example(info["base"])}
        graph_ex = {e["id"]: e for e in _load_per_example(info["graph"])}
        common = set(base_ex) & set(graph_ex)
        if not common:
            lines.append(f"| {info['label']} | {info['type']} | - | - | - | - |")
            continue
        wins   = sum(1 for qid in common if graph_ex[qid].get("em", 0) > base_ex[qid].get("em", 0))
        losses = sum(1 for qid in common if graph_ex[qid].get("em", 0) < base_ex[qid].get("em", 0))
        ties   = len(common) - wins - losses
        win_pct = f"{100*wins/len(common):.1f}%"
        lines.append(f"| {info['label']} | {info['type']} | {wins} | {losses} | {ties} | {win_pct} |")

    lines.append("")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/rq_report.md")
    args = parser.parse_args()

    lines: list[str] = [
        "# Research Question Analysis\n",
        "GraphRAG vs RAG - structured comparison for paper.\n",
        f"Generated automatically from `experiments/model_sweep/*__{MODEL}/*/metrics.json` "
        "and `per_example.jsonl` (RQ1-RQ5), RQ6 averages all 3 chat models.\n",
        "---\n",
    ]

    rq1_retrieval(lines)
    lines.append("---\n")
    rq2_answer_quality(lines)
    lines.append("---\n")
    rq3_hallucination(lines)
    lines.append("---\n")
    rq4_dataset_breakdown(lines)
    lines.append("---\n")
    rq5_efficiency(lines)
    lines.append("---\n")
    rq6_neo4j_backends(lines)
    lines.append("---\n")
    win_loss_tie(lines)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")

    # also print to stdout
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
