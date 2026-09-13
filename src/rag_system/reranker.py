from __future__ import annotations

import logging
from dataclasses import dataclass

from rag_system.index import RetrievedChunk
from rag_system.llm import RemoteLLM

logger = logging.getLogger(__name__)

_RERANK_SYSTEM = (
    "You are a relevance judge. Given a question and a passage, "
    "rate how relevant the passage is for answering the question. "
    "Respond with a single integer from 0 (completely irrelevant) "
    "to 10 (perfectly relevant). Output ONLY the number."
)


@dataclass
class LLMReranker:
    """
    Re-ranks a list of RetrievedChunk objects using LLM relevance scores.

    Parameters
    ----------
    llm : RemoteLLM
        The LLM used for scoring. Can be a smaller/faster model than the
        answer-generation LLM (e.g., a 7B model for scoring, 20B for generation).
    alpha : float
        Weight of LLM score in the combined score:
        combined = alpha * llm_score_norm + (1 - alpha) * cosine_score
        alpha=1.0 -> pure LLM rerank; alpha=0.0 -> pure cosine (no rerank).
    max_passage_chars : int
        Truncate each passage to this length before sending to the LLM to
        control token cost per scoring call.
    """

    llm: RemoteLLM
    alpha: float = 0.5
    max_passage_chars: int = 500

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Score each chunk and return them sorted by combined score (descending).
        """
        if not chunks:
            return chunks

        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            llm_score = self._score(query, chunk)
            cosine_norm = max(0.0, min(1.0, float(chunk.score)))
            llm_norm = llm_score / 10.0
            combined = self.alpha * llm_norm + (1 - self.alpha) * cosine_norm

            scored.append((combined, RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                text=chunk.text,
                score=combined,
            )))

        scored.sort(key=lambda t: t[0], reverse=True)
        reranked = [c for _, c in scored]
        logger.debug(
            "Reranked %d chunks for query: %s...", len(reranked), query[:60]
        )
        return reranked

    def _score(self, query: str, chunk: RetrievedChunk) -> float:
        passage = chunk.text[: self.max_passage_chars]
        user_msg = f"Question: {query}\n\nPassage: {passage}\n\nRelevance score (0-10):"
        messages = [
            {"role": "system", "content": _RERANK_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        try:
            result = self.llm.generate(messages)
            raw = result.text.strip().split()[0]
            score = float(raw)
            return max(0.0, min(10.0, score))
        except Exception as e:
            logger.warning("Reranker LLM call failed for chunk %s: %s", chunk.chunk_id, e)
            return 5.0  # neutral fallback
