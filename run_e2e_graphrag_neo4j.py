import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()


ROOT_PATH = Path(__file__).resolve().parent
SRC_PATH = ROOT_PATH / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_system.e2e_graphrag_neo4j import run_e2e_graphrag_neo4j

MAIN_PATH = os.getenv("MAIN_PATH", str(ROOT_PATH))

CONFIG_PATH = str(ROOT_PATH / "configs" / "e2e_hotpotqa_graphrag_neo4j.yaml")
EXPERIMENTS_BASE_PATH = os.path.join(MAIN_PATH, "experiments")
METRICS_HISTORY_PATH = os.path.join(EXPERIMENTS_BASE_PATH, "metrics_history.json")


class HotpotQAGraphRAGNeo4jRunner:
    """
    A runner for the stage-3 GraphRAG-on-Neo4j MVP pipeline. Mirrors
    `run_e2e.HotpotQARAGRunner` / `run_e2e_graphrag.HotpotQAGraphRAGRunner`,
    writing to the same metrics history file so all three systems
    (vector_rag / graph_rag / graph_rag_neo4j) can be compared.
    """

    def __init__(self, config_path=None, experiments_base_path=None, metrics_history_path=None):
        self.config_path = config_path or CONFIG_PATH
        self.experiments_base_path = experiments_base_path or EXPERIMENTS_BASE_PATH
        self.metrics_history_path = metrics_history_path or METRICS_HISTORY_PATH

        os.makedirs(self.experiments_base_path, exist_ok=True)

    def json_converter(self, value):
        """
        Convert values to JSON-serializable formats.
        """

        if isinstance(value, (np.integer)):
            return int(value)
        if isinstance(value, (np.floating)):
            return float(round(float(value), 6))
        if isinstance(value, dict):
            return {k: self.json_converter(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.json_converter(v) for v in value]
        return value

    def _find_latest_metrics_path(self):
        """
        Find the most recent 'metrics.json' for this run's experiment_name
        inside the experiments directory.
        """
        latest_path = None
        latest_mtime = -1.0

        for root, _dirs, files in os.walk(self.experiments_base_path):
            if "metrics.json" not in files:
                continue
            candidate = os.path.join(root, "metrics.json")
            try:
                mtime = os.path.getmtime(candidate)
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = candidate

        return latest_path

    def _append_to_metrics_history(self, metrics):
        """
        Append the latest run metrics into a history file.
        """
        os.makedirs(os.path.dirname(self.metrics_history_path), exist_ok=True)

        if os.path.exists(self.metrics_history_path):
            try:
                with open(self.metrics_history_path, encoding="utf-8") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = []
        else:
            history = []

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.json_converter(metrics),
        }
        history.append(record)

        with open(self.metrics_history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

        print(f"Metrics history has been updated: {self.metrics_history_path}")

    def run(self):
        """
        Run the GraphRAG-on-Neo4j MVP pipeline and register metrics in history.
        """

        print(f"Running HotpotQA GraphRAG (Neo4j) with config: {self.config_path}")
        run_e2e_graphrag_neo4j(self.config_path)

        latest_metrics_path = self._find_latest_metrics_path()
        if not latest_metrics_path:
            print("Warning: no metrics.json found under experiments directory.")
            return

        print(f"Latest metrics file: {latest_metrics_path}")

        try:
            with open(latest_metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error reading metrics file {latest_metrics_path}: {exc}")
            return

        self._append_to_metrics_history(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HotpotQA GraphRAG (Neo4j MVP) end-to-end")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: configs/e2e_hotpotqa_graphrag_neo4j.yaml)",
    )
    args = parser.parse_args()

    config_path = args.config or CONFIG_PATH
    if config_path and not os.path.isabs(config_path):
        config_path = str(ROOT_PATH / config_path)

    runner = HotpotQAGraphRAGNeo4jRunner(
        config_path=config_path,
        experiments_base_path=None,
        metrics_history_path=None,
    )
    runner.run()
