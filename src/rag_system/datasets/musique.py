from __future__ import annotations

from dataclasses import dataclass

from rag_system.data.musique import build_musique_corpus, iter_musique_examples
from rag_system.datasets.base import QADataset, QAExample


@dataclass
class MusiqueDataset(QADataset):
    train_split: str
    eval_split: str
    max_corpus_docs: int | None
    max_eval_examples: int | None

    name: str = "musique"

    def build_train_corpus(self):
        # Pool contexts from eval_split (not train_split): guarantees every
        # eval question's supporting docs are in the corpus (see the same
        # fix in datasets/hotpotqa.py for the full rationale).
        #
        # Scan far more rows (20_000, effectively the whole split) than
        # max_eval_examples: the tested questions are the first rows in this
        # same iteration order, so their gold docs are pooled first and
        # guaranteed present, while the extra rows add genuine distractor
        # documents - otherwise the corpus is just the tested questions' own
        # paragraphs, too easy for vector/graph search to differ on.
        from rag_system.data.musique import iter_musique_examples as _iter
        examples = list(_iter(split=self.eval_split, limit=20_000))
        return build_musique_corpus(examples, max_docs=self.max_corpus_docs)

    def iter_eval_examples(self):
        for ex in iter_musique_examples(split=self.eval_split, limit=self.max_eval_examples):
            if not ex.answer:
                continue
            yield QAExample(
                qid=ex.qid,
                question=ex.question,
                answer=ex.answer,
                supporting_titles=ex.supporting_titles,
            )
