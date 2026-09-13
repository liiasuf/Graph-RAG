from __future__ import annotations

import logging

from rag_system.graph.entities import extract_entities
from rag_system.llm import RemoteLLM

logger = logging.getLogger(__name__)

_ROUTE_SYSTEM = (
    "You are a routing classifier for a question-answering system. Decide "
    "whether answering the question benefits from multi-hop entity-graph "
    "traversal (bridge/comparison questions spanning multiple entities) or "
    "whether plain semantic search over passages is enough (single-fact "
    "lookup). Respond with exactly one word: 'graph' or 'vector'."
)


def heuristic_route(question: str, min_entities_for_graph: int) -> tuple[bool, str]:
    """
    Route via a cheap, DB-free heuristic (retrieval.mode == "adaptive",
    router.method == "heuristic"): count candidate entity mentions in the
    question text with the same regex heuristic used to build the graph
    (`extract_entities`) - no Neo4j round-trip. Below
    `min_entities_for_graph` mentions, the question is treated as
    effectively single-hop and the graph is skipped entirely for it (zero
    Neo4j calls for that question).
    """
    n = len(extract_entities(question))
    use_graph = n >= min_entities_for_graph
    reason = f"heuristic: {n} entity mention(s) found (threshold={min_entities_for_graph})"
    return use_graph, reason


def llm_route(question: str, llm: RemoteLLM) -> tuple[bool, str]:
    """
    Route via a short LLM classification call (router.method == "llm").
    Falls back to using the graph (the safer default) if the call fails or
    the response is ambiguous.
    """
    messages = [
        {"role": "system", "content": _ROUTE_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nRoute:"},
    ]
    try:
        result = llm.generate(messages)
        verdict = result.text.strip().lower()
        use_graph = "vector" not in verdict
        return use_graph, f"llm: {verdict!r}"
    except Exception as e:
        logger.warning("LLM router call failed, defaulting to graph: %s", e)
        return True, f"llm route failed ({e}), defaulting to graph"
