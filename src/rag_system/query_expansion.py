from __future__ import annotations

import logging

from rag_system.llm import RemoteLLM

logger = logging.getLogger(__name__)

_EXPAND_SYSTEM = (
    "You are a search query optimizer. "
    "Given a question, rewrite it as a more detailed search query that will help "
    "retrieve relevant Wikipedia passages. Add relevant context, synonyms, and "
    "related terms. Keep it under 3 sentences. Output ONLY the expanded query."
)


def expand_query(query: str, llm: RemoteLLM) -> str:
    """
    Return an LLM-expanded version of `query`.
    Falls back to the original query if the LLM call fails.
    """
    messages = [
        {"role": "system", "content": _EXPAND_SYSTEM},
        {"role": "user", "content": f"Question: {query}\n\nExpanded search query:"},
    ]
    try:
        result = llm.generate(messages)
        expanded = result.text.strip()
        if expanded and len(expanded) > len(query):
            logger.debug("Query expanded: %r -> %r", query[:60], expanded[:80])
            return expanded
        return query
    except Exception as e:
        logger.warning("Query expansion failed: %s", e)
        return query
