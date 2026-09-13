from __future__ import annotations

import math
import re
from dataclasses import dataclass


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common: dict[str, int] = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall_val = num_same / len(gold_tokens)
    return (2 * precision * recall_val) / (precision + recall_val)


def rouge_l(pred: str, gold: str) -> float:
    """
    ROUGE-L: longest common subsequence F1.
    Standard NLG metric (Barry et al. 2025, LinkedIn GraphRAG 2404.17723).
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return (2 * p * r) / (p + r)


def precision_at_k(retrieved_titles: list[str], supporting_titles: list[str]) -> float:
    """Fraction of retrieved docs that are relevant."""
    if not retrieved_titles or not supporting_titles:
        return 0.0
    supporting = set(supporting_titles)
    return sum(1 for t in retrieved_titles if t in supporting) / len(retrieved_titles)


def recall_at_k(retrieved_titles: list[str], supporting_titles: list[str]) -> float:
    """Fraction of relevant docs that were retrieved."""
    if not supporting_titles:
        return 0.0
    retrieved_set = set(retrieved_titles)
    return sum(1 for t in supporting_titles if t in retrieved_set) / len(supporting_titles)


def mrr(retrieved_titles: list[str], supporting_titles: list[str]) -> float:
    """1 / rank of first relevant document."""
    supporting = set(supporting_titles)
    for rank, title in enumerate(retrieved_titles, start=1):
        if title in supporting:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_titles: list[str], supporting_titles: list[str], k: int | None = None) -> float:
    """
    Normalised Discounted Cumulative Gain (binary relevance, per unique title).
    Penalises relevant docs ranked low, not just absent.
    Stronger than Recall@k for ranking quality assessment.
    Standard in IR papers (LinkedIn GraphRAG, 2404.17723).

    Relevance is per-document (title), not per-chunk: a title only earns a
    DCG contribution once, at the best (earliest) rank it appears at.
    Retrieval operates on chunks, and a single document is often split into
    several chunks that share its title - without dedup, a document with
    multiple chunks in the top-k would be credited multiple times, letting
    DCG exceed IDCG (NDCG > 1).
    """
    if not supporting_titles or not retrieved_titles:
        return 0.0
    supporting = set(supporting_titles)
    ranked = retrieved_titles[:k] if k else retrieved_titles
    seen: set[str] = set()
    dcg = 0.0
    for i, t in enumerate(ranked):
        if t in supporting and t not in seen:
            seen.add(t)
            dcg += 1.0 / math.log2(i + 2)
    ideal_k = min(len(supporting), len(ranked))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
    return dcg / idcg if idcg > 0 else 0.0


def hallucination_rate(pred: str, retrieved_texts: list[str]) -> float:
    """
    Faithfulness proxy: 1.0 if fewer than 50% of answer tokens appear in
    the retrieved context (Barry et al. 2025, Lavrinovics et al. 2025).
    Returns 1.0 = likely hallucinated | 0.0 = grounded in context.
    """
    if not pred or not retrieved_texts:
        return 1.0
    pred_tokens = set(normalize_answer(pred).split())
    if not pred_tokens:
        return 1.0
    context_tokens = set(normalize_answer(" ".join(retrieved_texts)).split())
    overlap = pred_tokens & context_tokens
    return 0.0 if len(overlap) / len(pred_tokens) >= 0.5 else 1.0


@dataclass
class MetricSums:
    n: int = 0
    em: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    rouge_l_sum: float = 0.0
    hallucination: float = 0.0

    def add(
        self, *, em, f1, accuracy, recall, mrr,
        precision=0.0, ndcg=0.0, rouge_l_val=0.0, hallucination=0.0,
    ) -> None:
        self.n += 1
        self.em += em
        self.f1 += f1
        self.accuracy += accuracy
        self.recall += recall
        self.precision += precision
        self.mrr += mrr
        self.ndcg += ndcg
        self.rouge_l_sum += rouge_l_val
        self.hallucination += hallucination

    def mean(self) -> dict:
        if self.n == 0:
            return {"n": 0, "em": 0.0, "f1": 0.0, "accuracy": 0.0,
                    "recall": 0.0, "precision": 0.0, "mrr": 0.0,
                    "ndcg": 0.0, "rouge_l": 0.0, "hallucination_rate": 0.0}
        r = lambda v: round(v / self.n, 5)
        return {
            "n": self.n,
            "em": r(self.em),
            "f1": r(self.f1),
            "accuracy": r(self.accuracy),
            "recall": r(self.recall),
            "precision": r(self.precision),
            "mrr": r(self.mrr),
            "ndcg": r(self.ndcg),
            "rouge_l": r(self.rouge_l_sum),
            "hallucination_rate": r(self.hallucination),
        }
