
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
EXPERIMENTS = ROOT / "experiments" / "model_sweep"
EXPERIMENTS_QUICK = ROOT / "experiments" / "model_sweep_quick"
CFG_DIR = ROOT / "configs" / "model_sweep"
SERVER_BASE_URL = os.environ.get("MODEL_SWEEP_BASE_URL")
SERVER_API_KEY = os.environ.get("MODEL_SWEEP_API_KEY")

CHAT_MODELS = {
    "qwen3.8": "qwen3.8",
    "gemma-4-26B-A4B-it": "openai/gemma-4-26B-A4B-it",
    "qwen3-vl-30b": "openai/qwen3-vl-30b",
}

EMBED_MODEL = "openai/bge-m3"

SWEEP_MAX_TOKENS = int(os.environ.get("MODEL_SWEEP_MAX_TOKENS", "2048"))
DATASET_CONFIGS: list[tuple[str, str]] = []
for _ds in ("hotpotqa", "musique", "nq", "triviaqa", "wiki2multihop"):
    DATASET_CONFIGS += [
        (f"configs/e2e_{_ds}.yaml", "run_e2e.py"),
        (f"configs/e2e_{_ds}_graph_only.yaml", "run_e2e_graphrag_neo4j.py"),
        (f"configs/e2e_{_ds}_graph_overlay.yaml", "run_e2e_graphrag_neo4j.py"),
        (f"configs/e2e_{_ds}_graph_rerank.yaml", "run_e2e_graphrag_neo4j.py"),
    ]

QUICK_MAX_EVAL_EXAMPLES = 8
QUICK_MAX_CORPUS_DOCS = 300


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def _apply_quick_overrides(cfg: dict) -> None:
    for value in cfg.values():
        if isinstance(value, dict):
            if "max_eval_examples" in value:
                value["max_eval_examples"] = QUICK_MAX_EVAL_EXAMPLES
            if "max_corpus_docs" in value:
                value["max_corpus_docs"] = QUICK_MAX_CORPUS_DOCS


def _build_config(base_cfg_path: Path, model_tag: str, quick: bool) -> Path:
    """
    Write a per-(dataset, model) config with just experiment_name/output_dir/
    cache_path (and, for --quick, corpus/eval size) overridden. The actual
    base_url/api_key/model for llm and embeddings are supplied at run time
    via env vars in _run(), so no secret ever lands in a file under configs/.
    """
    with base_cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    base_experiment_name = cfg["experiment_name"]
    suffix = "__quick" if quick else ""
    cfg["experiment_name"] = f"{base_experiment_name}__{_slugify(model_tag)}{suffix}"
    cfg["output_dir"] = str(EXPERIMENTS_QUICK if quick else EXPERIMENTS)
    # shared across the 3 chat-model runs for this dataset (embeddings don't
    # depend on the chat model), kept separate from the local nomic-embed cache
    cache_tag = f"{_slugify(base_cfg_path.stem)}{suffix}"
    cfg["embeddings"]["cache_path"] = f"cache/embeddings_bge_m3_{cache_tag}.json"

    if quick:
        _apply_quick_overrides(cfg)

    sub_dir = CFG_DIR / ("quick" if quick else "full")
    out = sub_dir / f"{base_cfg_path.stem}__{_slugify(model_tag)}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return out


def _run(config_path: Path, script_name: str, model_name: str) -> bool:
    script = ROOT / script_name
    cmd = [sys.executable, str(script), "--config", str(config_path)]
    env = {
        **os.environ,
        "LLM_BASE_URL": SERVER_BASE_URL,
        "LLM_API_KEY": SERVER_API_KEY,
        "LLM_MODEL": model_name,
        "EMBED_BASE_URL": SERVER_BASE_URL,
        "EMBED_API_KEY": SERVER_API_KEY,
        "EMBED_MODEL": EMBED_MODEL,
        "LLM_MAX_TOKENS": str(SWEEP_MAX_TOKENS),
    }
    print(f"\n>>> Running: {' '.join(cmd)}  (model={model_name})")
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    return result.returncode == 0


def _find_latest(experiments_dir: Path, exp_name: str) -> Path | None:
    base = experiments_dir / exp_name
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    return max(candidates, key=lambda p: os.path.getmtime(p / "metrics.json")) if candidates else None


def _load_metrics(experiments_dir: Path, exp_name: str) -> dict | None:
    run = _find_latest(experiments_dir, exp_name)
    return json.loads((run / "metrics.json").read_text()) if run else None


def _fmt(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "-"


def build_report(out_path: Path, models: list[str], quick: bool) -> None:
    experiments_dir = EXPERIMENTS_QUICK if quick else EXPERIMENTS
    suffix = "__quick" if quick else ""

    lines = [
        "# Model Sweep Results" + (" (QUICK smoke test - not final numbers)" if quick else "") + "\n",
        f"Server: `{SERVER_BASE_URL}` - chat models: {', '.join(models)}; embeddings: `{EMBED_MODEL}`.\n",
        f"Built from `{experiments_dir.relative_to(ROOT)}/*/metrics.json` (real e2e runs).\n",
    ]

    for cfg_rel, _script_name in DATASET_CONFIGS:
        base_experiment_name = yaml.safe_load((ROOT / cfg_rel).read_text())["experiment_name"]
        lines.append(f"\n## {base_experiment_name}\n")
        header = "| Model | EM | F1 | Recall@k | NDCG@k | MRR | Halluc↓ | Query time (s) |"
        sep = "|-------|----|----|----------|--------|-----|---------|----------------|"
        lines += [header, sep]
        for model_tag in models:
            exp_name = f"{base_experiment_name}__{_slugify(model_tag)}{suffix}"
            m = _load_metrics(experiments_dir, exp_name)
            if m is None:
                lines.append(f"| {model_tag} | - | - | - | - | - | - | - |")
                continue
            q, lat = m["quality"], m["latency"]
            lines.append(
                f"| {model_tag} "
                f"| {_fmt(q.get('em'))} | {_fmt(q.get('f1'))} | {_fmt(q.get('recall'))} "
                f"| {_fmt(q.get('ndcg'))} | {_fmt(q.get('mrr'))} | {_fmt(q.get('hallucination_rate'))} "
                f"| {_fmt(lat.get('query_time_s'))} |"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run vector + graph_only + graph_overlay + graph_rerank (Neo4j) experiments "
            "across multiple remote chat models, for every dataset. Excludes 'adaptive'. "
            "Requires: docker compose up -d ; pip install -e .[neo4j]"
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"Smoke test: {QUICK_MAX_EVAL_EXAMPLES} eval examples, "
            f"{QUICK_MAX_CORPUS_DOCS} corpus docs, separate output dir "
            "(experiments/model_sweep_quick/) so it can't be confused with a real run."
        ),
    )
    parser.add_argument(
        "--only-table",
        action="store_true",
        help="Skip running experiments, just rebuild the report from existing metrics.json files",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=list(CHAT_MODELS),
        help="Restrict to one or more models (repeatable). Default: all.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Restrict to configs whose filename contains this substring (e.g. hotpotqa). Repeatable.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip (dataset, method, model) combos that already have a metrics.json "
            "(e.g. after a partial sweep died from a network outage) instead of "
            "recomputing everything from scratch."
        ),
    )
    parser.add_argument("--out", default=None, help="Report path (default depends on --quick)")
    args = parser.parse_args()

    models = args.model or list(CHAT_MODELS)
    out_path = Path(
        args.out
        or (
            "experiments/model_sweep_quick/model_sweep_report.md"
            if args.quick
            else "experiments/model_sweep/model_sweep_report.md"
        )
    )

    if not args.only_table:
        if not SERVER_BASE_URL or not SERVER_API_KEY:
            raise SystemExit(
                "Set MODEL_SWEEP_BASE_URL and MODEL_SWEEP_API_KEY in .env before running "
                "the sweep (see .env.example)."
            )
        experiments_dir = EXPERIMENTS_QUICK if args.quick else EXPERIMENTS
        for cfg_rel, script_name in DATASET_CONFIGS:
            if args.dataset and not any(d in cfg_rel for d in args.dataset):
                continue
            base_cfg_path = ROOT / cfg_rel
            base_experiment_name = yaml.safe_load(base_cfg_path.read_text())["experiment_name"]
            for model_tag in models:
                if args.skip_existing:
                    suffix = "__quick" if args.quick else ""
                    exp_name = f"{base_experiment_name}__{_slugify(model_tag)}{suffix}"
                    if _find_latest(experiments_dir, exp_name) is not None:
                        print(f"SKIP (already has metrics.json): {cfg_rel} x {model_tag}")
                        continue
                cfg_path = _build_config(base_cfg_path, model_tag, quick=args.quick)
                ok = _run(cfg_path, script_name=script_name, model_name=CHAT_MODELS[model_tag])
                if not ok:
                    print(f"WARNING: {cfg_rel} x {model_tag} failed, continuing...")
                # rebuild after every run (cheap: just re-reads existing metrics.json
                # files) so the report reflects progress instead of only appearing
                # once the entire sweep - which can take a long time - finishes
                build_report(out_path, models, quick=args.quick)

    build_report(out_path, models, quick=args.quick)


if __name__ == "__main__":
    main()
