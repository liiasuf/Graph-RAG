from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
EXPERIMENTS = ROOT / "experiments"

BASE_EXP   = "hotpotqa_rag_baseline"
GRAPH_EXP  = "hotpotqa_graphrag"


def _find_latest(exp_name: str) -> Path | None:
    base = EXPERIMENTS / exp_name
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "per_example.jsonl").exists()]
    return max(candidates, key=lambda p: os.path.getmtime(p / "per_example.jsonl")) if candidates else None


def _load_per_example(exp_name: str) -> dict[str, dict]:
    run = _find_latest(exp_name)
    if not run:
        return {}
    rows = {}
    with (run / "per_example.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                rows[d["id"]] = d
    return rows


def _load_hotpotqa_types() -> dict[str, str]:
    """Load question type labels from HuggingFace HotpotQA validation split."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
        return {ex["id"]: ex["type"] for ex in ds}
    except Exception as e:
        print(f"Warning: could not load HotpotQA types from HuggingFace: {e}")
        print("Using heuristic: questions containing 'both'/'which'/'who' -> comparison, else bridge")
        return {}


def _heuristic_type(question: str) -> str:
    q = question.lower()
    comparison_keywords = ["which", "both", "same", "different", "more", "less", "older", "younger",
                           "taller", "shorter", "earlier", "later", "larger", "smaller"]
    return "comparison" if any(k in q for k in comparison_keywords) else "bridge"


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/question_type_analysis.md")
    args = parser.parse_args()

    base_ex  = _load_per_example(BASE_EXP)
    graph_ex = _load_per_example(GRAPH_EXP)
    common = set(base_ex) & set(graph_ex)

    if not common:
        print("No overlapping examples found. Run both experiments first.")
        return

    # Load type labels
    type_map = _load_hotpotqa_types()
    if not type_map:
        print("Using heuristic type assignment.")
        type_map = {qid: _heuristic_type(base_ex[qid]["question"]) for qid in common}

    # Group by type
    by_type: dict[str, list[str]] = defaultdict(list)
    for qid in common:
        qtype = type_map.get(qid, _heuristic_type(base_ex[qid]["question"]))
        by_type[qtype].append(qid)

    lines = [
        "# Question Type Analysis: Bridge vs Comparison\n",
        f"Total examples: {len(common)} (common to both systems)\n",
        f"Types: {', '.join(f'{t}: {len(ids)}' for t, ids in sorted(by_type.items()))}\n",
        "---\n",
    ]

    for qtype in sorted(by_type.keys()):
        qids = by_type[qtype]
        lines.append(f"## {qtype.capitalize()} questions (n={len(qids)})\n")

        # aggregate metrics
        for sys_name, ex_map in [("RAG", base_ex), ("GraphRAG", graph_ex)]:
            em_vals   = [ex_map[q]["em"]   for q in qids if q in ex_map]
            f1_vals   = [ex_map[q]["f1"]   for q in qids if q in ex_map]
            rec_vals  = [ex_map[q].get("recall", 0) for q in qids if q in ex_map]
            ndcg_vals = [ex_map[q].get("ndcg", 0)   for q in qids if q in ex_map]
            lines.append(
                f"**{sys_name}**: EM={_avg(em_vals):.4f}  F1={_avg(f1_vals):.4f}  "
                f"Recall={_avg(rec_vals):.4f}  NDCG={_avg(ndcg_vals):.4f}"
            )

        # delta
        for metric in ["em", "f1", "recall", "ndcg"]:
            base_vals  = [base_ex[q].get(metric, 0)  for q in qids if q in base_ex]
            graph_vals = [graph_ex[q].get(metric, 0) for q in qids if q in graph_ex]
            d = _avg(graph_vals) - _avg(base_vals)
            sign = "+" if d >= 0 else ""
            lines.append(f"  Δ {metric}: {sign}{d:.4f} {'↑' if d > 0.001 else ('↓' if d < -0.001 else '=')}")
        lines.append("")

        # Win/Loss/Tie
        wins   = sum(1 for q in qids if graph_ex.get(q, {}).get("em", 0) > base_ex.get(q, {}).get("em", 0))
        losses = sum(1 for q in qids if graph_ex.get(q, {}).get("em", 0) < base_ex.get(q, {}).get("em", 0))
        ties   = len(qids) - wins - losses
        lines.append(f"**Win/Loss/Tie:** {wins} / {losses} / {ties}\n")

        # Qualitative examples
        lines.append("### Examples where GraphRAG won (EM: 0->1)\n")
        count = 0
        for q in qids:
            if base_ex.get(q, {}).get("em", 1) == 0 and graph_ex.get(q, {}).get("em", 0) == 1:
                ex = graph_ex[q]
                lines.append(f"**Q:** {ex['question']}")
                lines.append(f"**Gold:** {ex['gold']}")
                lines.append(f"**Graph pred:** {ex['pred']}")
                lines.append(f"**Graph retrieved:** {', '.join(ex.get('retrieved_titles', [])[:3])}")
                lines.append(f"**Supporting:** {', '.join(ex.get('supporting_titles', []))}")
                lines.append("")
                count += 1
                if count >= 3:
                    break
        if count == 0:
            lines.append("*(no examples yet - run full experiments)*\n")

        lines.append("### Examples where GraphRAG lost (EM: 1->0)\n")
        count = 0
        for q in qids:
            if base_ex.get(q, {}).get("em", 0) == 1 and graph_ex.get(q, {}).get("em", 1) == 0:
                ex = base_ex[q]
                lines.append(f"**Q:** {ex['question']}")
                lines.append(f"**Gold:** {ex['gold']}")
                lines.append(f"**Base pred:** {ex['pred']}  |  **Graph pred:** {graph_ex[q]['pred']}")
                lines.append("")
                count += 1
                if count >= 2:
                    break
        if count == 0:
            lines.append("*(no examples yet)*\n")

        lines.append("---\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
