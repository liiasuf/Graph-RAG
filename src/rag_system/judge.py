from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_COMPARE_PROMPT = """\
You are an impartial judge evaluating two AI system answers to a question.

Question: {question}

Retrieved context (same for both systems):
{context}

=== Answer A ===
{answer_a}

=== Answer B ===
{answer_b}

Rate each answer on the following criteria (0-10 scale):
1. Comprehensiveness: covers all aspects of the question
2. Faithfulness: grounded in the retrieved context, no hallucination
3. Relevance: directly answers what was asked

Respond ONLY with valid JSON:
{{
  "scores_a": {{"comprehensiveness": <int>, "faithfulness": <int>, "relevance": <int>}},
  "scores_b": {{"comprehensiveness": <int>, "faithfulness": <int>, "relevance": <int>}},
  "winner": "<A|B|Tie>",
  "reasoning": "<one sentence>"
}}
"""

_SINGLE_PROMPT = """\
You are an impartial judge evaluating an AI system answer to a question.

Question: {question}

Retrieved context:
{context}

Answer:
{answer}

Rate this answer on the following criteria (0-10 scale):
1. Comprehensiveness: covers all aspects of the question
2. Faithfulness: grounded in the retrieved context, no hallucination
3. Relevance: directly answers what was asked

Respond ONLY with valid JSON:
{{
  "comprehensiveness": <int>,
  "faithfulness": <int>,
  "relevance": <int>,
  "reasoning": "<one sentence>"
}}
"""


@dataclass
class CompareResult:
    winner: str                          # "A", "B", or "Tie"
    scores_a: dict[str, float]
    scores_b: dict[str, float]
    reasoning: str = ""
    raw: str = ""


@dataclass
class ScoreResult:
    scores: dict[str, float]
    reasoning: str = ""
    raw: str = ""


class LLMJudge:
    """
    Wraps a RemoteLLM to act as an evaluation judge.

    Uses structured JSON output to compare or score answers.
    Falls back gracefully on parse errors (returns Tie / zero scores).
    """

    def __init__(self, llm, max_context_chars: int = 2000) -> None:
        self.llm = llm
        self.max_context_chars = max_context_chars

    def _call(self, prompt: str) -> str:
        from rag_system.llm import RemoteLLM  # avoid circular at module level
        messages = [
            {"role": "system", "content": "You are an impartial evaluation judge. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            res = self.llm.generate(messages)
            return res.text
        except Exception as e:
            logger.warning("LLM judge call failed: %s", e)
            return ""

    def _parse_json(self, text: str) -> dict:
        # strip markdown code fences if present
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # try to find first {...} block
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return {}

    def compare(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        context: str = "",
    ) -> CompareResult:
        """
        Head-to-head comparison of answer_a vs answer_b.
        Returns winner + per-criterion scores for each.
        """
        ctx = context[: self.max_context_chars]
        prompt = _COMPARE_PROMPT.format(
            question=question, context=ctx, answer_a=answer_a, answer_b=answer_b
        )
        raw = self._call(prompt)
        data = self._parse_json(raw)

        default_scores = {"comprehensiveness": 5, "faithfulness": 5, "relevance": 5}
        return CompareResult(
            winner=data.get("winner", "Tie"),
            scores_a=data.get("scores_a", default_scores),
            scores_b=data.get("scores_b", default_scores),
            reasoning=data.get("reasoning", ""),
            raw=raw,
        )

    def score(
        self,
        question: str,
        answer: str,
        context: str = "",
    ) -> ScoreResult:
        """Score a single answer on comprehensiveness, faithfulness, relevance."""
        ctx = context[: self.max_context_chars]
        prompt = _SINGLE_PROMPT.format(question=question, context=ctx, answer=answer)
        raw = self._call(prompt)
        data = self._parse_json(raw)
        scores = {
            "comprehensiveness": float(data.get("comprehensiveness", 5)),
            "faithfulness": float(data.get("faithfulness", 5)),
            "relevance": float(data.get("relevance", 5)),
        }
        return ScoreResult(scores=scores, reasoning=data.get("reasoning", ""), raw=raw)
