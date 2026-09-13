from __future__ import annotations

from dataclasses import dataclass

from rag_system.data.natural_questions import build_nq_corpus, iter_nq_examples
from rag_system.datasets.base import QADataset, QAExample


@dataclass
class NaturalQuestionsDataset(QADataset):
    eval_split: str
    max_corpus_docs: int | None
    max_eval_examples: int | None

    name: str = "natural_questions"

    def build_train_corpus(self):
        # NQ-open uses Wikipedia as corpus; load from full NQ validation docs
        examples = list(iter_nq_examples(split=self.eval_split, limit=self.max_eval_examples))
        return build_nq_corpus(examples, max_docs=self.max_corpus_docs)

    def iter_eval_examples(self):
        for ex in iter_nq_examples(split=self.eval_split, limit=self.max_eval_examples):
            yield QAExample(
                qid=ex.qid,
                question=ex.question,
                answer=ex.answer,
                supporting_titles=ex.supporting_titles,
            )
