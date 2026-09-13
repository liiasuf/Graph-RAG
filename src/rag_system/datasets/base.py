from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class QAExample:
    """
    Generic QA example for evaluation.
    """

    qid: str
    question: str
    answer: str
    supporting_titles: tuple[str, ...]


class QADataset(Protocol):
    """
    Protocol for QA datasets usable by the baseline end-to-end runner.
    """

    name: str

    def build_train_corpus(self): ...

    def iter_eval_examples(self): ...
