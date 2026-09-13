from __future__ import annotations

import logging

from rag_system.llm import RemoteLLM

logger = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = (
    "You are a multi-hop question decomposer. Given a complex question, "
    "produce ONE simpler sub-question that, if answered first, would help "
    "answer the original question. Output ONLY the sub-question, no "
    "preamble or numbering."
)

_FOLLOWUP_SYSTEM = (
    "You are a multi-hop question decomposer. Given the original question, "
    "a sub-question already asked, and an entity discovered while "
    "answering it, produce ONE follow-up sub-question centered on that "
    "entity that helps answer the original question. Output ONLY the "
    "sub-question, no preamble or numbering."
)


def decompose_question(question: str, llm: RemoteLLM) -> str:
    """
    Return the first sub-question for `question` (adaptive mode, step 1 of
    the subquestion loop). Falls back to the original question if the LLM
    call fails or returns nothing usable.
    """
    messages = [
        {"role": "system", "content": _DECOMPOSE_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nSub-question:"},
    ]
    try:
        result = llm.generate(messages)
        sub = result.text.strip()
        return sub or question
    except Exception as e:
        logger.warning("Sub-question decomposition failed: %s", e)
        return question


def followup_subquestion(question: str, subquestion1: str, entity: str, llm: RemoteLLM) -> str:
    """
    Return a follow-up sub-question centered on `entity` (adaptive mode,
    step 4 of the subquestion loop - `entity` must already be a graph node
    resolved via `Neo4jGraphIndex.match_entities`, not arbitrary text).
    Falls back to a templated question if the LLM call fails.
    """
    messages = [
        {"role": "system", "content": _FOLLOWUP_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Original question: {question}\n"
                f"First sub-question: {subquestion1}\n"
                f"Entity found: {entity}\n\n"
                "Follow-up sub-question:"
            ),
        },
    ]
    try:
        result = llm.generate(messages)
        sub = result.text.strip()
        return sub or f"What else is relevant about {entity} for: {question}"
    except Exception as e:
        logger.warning("Follow-up sub-question generation failed: %s", e)
        return f"What else is relevant about {entity} for: {question}"
