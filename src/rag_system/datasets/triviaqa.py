from __future__ import annotations

from dataclasses import dataclass

from rag_system.data.triviaqa import build_triviaqa_corpus, iter_triviaqa_examples
from rag_system.datasets.base import QADataset, QAExample


@dataclass
class TriviaQADataset(QADataset):
    eval_split: str
    max_corpus_docs: int | None
    max_eval_examples: int | None

    name: str = "triviaqa"

    def build_train_corpus(self):
        examples = list(iter_triviaqa_examples(split=self.eval_split, limit=self.max_eval_examples))
        return build_triviaqa_corpus(examples, max_docs=self.max_corpus_docs)

    def iter_eval_examples(self):
        for ex in iter_triviaqa_examples(split=self.eval_split, limit=self.max_eval_examples):
            if not ex.answer:
                continue
            yield QAExample(
                qid=ex.qid,
                question=ex.question,
                answer=ex.answer,
                supporting_titles=ex.supporting_titles,
            )
