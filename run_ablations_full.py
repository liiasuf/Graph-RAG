from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
EXPERIMENTS = ROOT / "experiments" / "ablations"

BASE_GRAPH_CFG    = ROOT / "configs" / "ablations" / "_base_graph_neo4j_only.yaml"
BASE_VEC_CFG      = ROOT / "configs" / "ablations" / "_base_vector.yaml"
BASE_ADAPTIVE_CFG = ROOT / "configs" / "ablations" / "_base_adaptive.yaml"

ABLATION_GRID = {
    "A1_hops": [
        {"name": "graph_hops1", "graph.max_hops": 1},
        {"name": "graph_hops2", "graph.max_hops": 2},
    ],
    "A2_neighbors": [
        {"name": "graph_n10",  "graph.max_neighbors_per_hop": 10},
        {"name": "graph_n20",  "graph.max_neighbors_per_hop": 20},
        {"name": "graph_n40",  "graph.max_neighbors_per_hop": 40},
    ],
    "A3_topk_graph": [
        {"name": "graph_top4",  "retrieval.top_k": 4},
        {"name": "graph_top8",  "retrieval.top_k": 8},
        {"name": "graph_top16", "retrieval.top_k": 16},
    ],
    "A4_title_entity": [
        {"name": "graph_title_true",  "graph.title_as_entity": True},
        {"name": "graph_title_false", "graph.title_as_entity": False},
    ],
    "A5_topk_vector": [
        {"name": "vector_top4",  "retrieval.top_k": 4},
        {"name": "vector_top8",  "retrieval.top_k": 8},
        {"name": "vector_top16", "retrieval.top_k": 16},
    ],
    # ── Stage 4 - adaptive method components (each independently toggled,
    # see AdaptiveMethodConfig in config.py) ──────────────────────────────
    "B1_router": [
        {"name": "adaptive_router_off",       "adaptive.router.enabled": False},
        {"name": "adaptive_router_heuristic", "adaptive.router.enabled": True,  "adaptive.router.method": "heuristic"},
        {"name": "adaptive_router_llm",       "adaptive.router.enabled": True,  "adaptive.router.method": "llm"},
    ],
    "B2_graph_rerank": [
        {"name": "adaptive_rerank_off_overlay",    "adaptive.graph_retrieval_mode": "graph_neo4j"},
        {"name": "adaptive_rerank_off_graph_only", "adaptive.graph_retrieval_mode": "graph_neo4j_only"},
        {"name": "adaptive_rerank_on",             "adaptive.graph_retrieval_mode": "graph_neo4j_rerank"},
    ],
    "B3_subquestions": [
        {"name": "adaptive_subq_off", "adaptive.subquestions.enabled": False},
        {"name": "adaptive_subq_on",  "adaptive.subquestions.enabled": True, "adaptive.subquestions.max_iterations": 1},
    ],
    "B4_combined": [
        {
            "name": "adaptive_all_off",
            "adaptive.router.enabled": False,
            "adaptive.subquestions.enabled": False,
            "adaptive.graph_retrieval_mode": "graph_neo4j",
        },
        {
            "name": "adaptive_all_on",
            "adaptive.router.enabled": True,
            "adaptive.router.method": "heuristic",
            "adaptive.subquestions.enabled": True,
            "adaptive.subquestions.max_iterations": 1,
            "adaptive.graph_retrieval_mode": "graph_neo4j_rerank",
        },
    ],
}

# which base config template + which run_e2e_*.py script each group uses.
# Neo4j is the sole graph backend for this project - A1-A4 run through
# run_e2e_graphrag_neo4j.py (retrieval.mode: graph_neo4j_only).
GROUP_RUNNER = {
    "A1_hops":          (BASE_GRAPH_CFG,    "run_e2e_graphrag_neo4j.py"),
    "A2_neighbors":     (BASE_GRAPH_CFG,    "run_e2e_graphrag_neo4j.py"),
    "A3_topk_graph":    (BASE_GRAPH_CFG,    "run_e2e_graphrag_neo4j.py"),
    "A4_title_entity":  (BASE_GRAPH_CFG,    "run_e2e_graphrag_neo4j.py"),
    "A5_topk_vector":   (BASE_VEC_CFG,      "run_e2e.py"),
    "B1_router":        (BASE_ADAPTIVE_CFG, "run_e2e_graphrag_neo4j.py"),
    "B2_graph_rerank":  (BASE_ADAPTIVE_CFG, "run_e2e_graphrag_neo4j.py"),
    "B3_subquestions":  (BASE_ADAPTIVE_CFG, "run_e2e_graphrag_neo4j.py"),
    "B4_combined":      (BASE_ADAPTIVE_CFG, "run_e2e_graphrag_neo4j.py"),
}


def _set_nested(d: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _build_config(base_cfg_path: Path, overrides: dict, exp_name: str) -> Path:
    with base_cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["experiment_name"] = f"ablation_{exp_name}"
    cfg["output_dir"] = str(EXPERIMENTS)
    for k, v in overrides.items():
        if k not in ("name",):
            _set_nested(cfg, k, v)
    out = ROOT / "configs" / "ablations" / "generated" / f"{exp_name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return out


def _run(config_path: Path, script_name: str) -> bool:
    script = ROOT / script_name
    cmd = [sys.executable, str(script), "--config", str(config_path)]
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def _find_latest(exp_name: str) -> Path | None:
    base = EXPERIMENTS / f"ablation_{exp_name}"
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    return max(candidates, key=lambda p: os.path.getmtime(p / "metrics.json")) if candidates else None


def _load_metrics(exp_name: str) -> dict | None:
    run = _find_latest(exp_name)
    return json.loads((run / "metrics.json").read_text()) if run else None


def _fmt(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "-"


def build_table(lines: list[str]) -> None:
    lines.append("# Ablation Study Results\n")
    lines.append("Dataset: HotpotQA distractor, 60 eval examples, same corpus/LLM for all variants.\n")

    for group_name, variants in ABLATION_GRID.items():
        lines.append(f"\n## {group_name}\n")
        header = "| Config | EM | F1 | Recall@k | NDCG@k | MRR | Halluc↓ |"
        sep    = "|--------|----|----|----------|--------|-----|---------|"
        lines += [header, sep]

        for v in variants:
            m = _load_metrics(v["name"])
            if m is None:
                lines.append(f"| {v['name']} | - | - | - | - | - | - |")
                continue
            q = m["quality"]
            lines.append(
                f"| {v['name']} "
                f"| {_fmt(q.get('em'))} "
                f"| {_fmt(q.get('f1'))} "
                f"| {_fmt(q.get('recall'))} "
                f"| {_fmt(q.get('ndcg'))} "
                f"| {_fmt(q.get('mrr'))} "
                f"| {_fmt(q.get('hallucination_rate'))} |"
            )
        lines.append("")

    lines.append("\n**Interpretation:**")
    lines.append("- A1: increasing hops should improve recall on bridge questions at cost of precision")
    lines.append("- A2: more neighbors -> more recall, but noise increases beyond N=20")
    lines.append("- A3: larger top_k always helps recall, diminishing returns on EM/F1")
    lines.append("- A4: title_as_entity=true should help bridge questions (title is the bridge entity)")
    lines.append("- A5: vector RAG top_k baseline comparison")
    lines.append("- B1: does skipping the graph for low-entity questions preserve quality at lower latency?")
    lines.append("- B2: does graph-hop re-ranking beat plain vector cosine on multi-hop questions?")
    lines.append("- B3: does subquestion decomposition recover bridge entities vector/graph search misses?")
    lines.append("- B4: combined effect of all three adaptive components vs. each in isolation\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-table", action="store_true")
    parser.add_argument("--group", default=None, help="Run only one group, e.g. A1_hops or B1_router")
    parser.add_argument("--out", default="experiments/ablations/ablation_table.md")
    args = parser.parse_args()

    if not args.only_table:
        for group_name, variants in ABLATION_GRID.items():
            if args.group and group_name != args.group:
                continue
            base_cfg, script_name = GROUP_RUNNER[group_name]
            for v in variants:
                overrides = {k: val for k, val in v.items() if k != "name"}
                cfg_path = _build_config(base_cfg, overrides, v["name"])
                ok = _run(cfg_path, script_name)
                if not ok:
                    print(f"WARNING: {v['name']} failed, continuing...")

    lines: list[str] = []
    build_table(lines)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {out}")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
