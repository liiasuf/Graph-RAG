from __future__ import annotations

import logging

from tqdm import tqdm

from rag_system.llm import RemoteLLM
from rag_system.schema import Chunk

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM = (
    "You are a text summarizer. Summarize the passage to its key facts in 1-2 "
    "sentences. Keep named entities, dates, and numbers. Output ONLY the summary."
)


def summarize_chunk_text(text: str, llm: RemoteLLM, max_input_chars: int = 1000) -> str:
    """Summarize a single chunk's text. Returns original on failure."""
    passage = text[:max_input_chars]
    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": f"Passage:\n{passage}\n\nSummary:"},
    ]
    try:
        result = llm.generate(messages)
        summary = result.text.strip()
        return summary if summary else text
    except Exception as e:
        logger.warning("Summarization failed: %s", e)
        return text


def summarize_chunks(
    chunks: list[Chunk],
    llm: RemoteLLM,
    *,
    max_input_chars: int = 1000,
) -> list[Chunk]:
    """
    Return a new list of Chunk objects with summarized text.
    Original chunks are not modified.
    """
    logger.info("Summarizing %d chunks (this increases indexing time)...", len(chunks))
    summarized = []
    for chunk in tqdm(chunks, desc="Summarizing chunks"):
        summary = summarize_chunk_text(chunk.text, llm, max_input_chars=max_input_chars)
        summarized.append(Chunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            text=summary,
        ))
    logger.info("Summarization complete.")
    return summarized
