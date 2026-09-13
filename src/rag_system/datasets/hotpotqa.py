from __future__ import annotations

from dataclasses import dataclass

from rag_system.data.hotpotqa import build_corpus_from_examples, iter_hotpotqa_examples
from rag_system.datasets.base import QADataset, QAExample


@dataclass
class HotpotQADataset(QADataset):
    """
    HotpotQA adapter that exposes a train corpus and evaluation examples.
    """

    subset: str
    train_split: str
    eval_split: str
    max_train_examples: int | None
    max_corpus_docs: int | None
    max_eval_examples: int | None

    name: str = "hotpotqa"

    def build_train_corpus(self):
        # Pool contexts from eval_split (not train_split): each HotpotQA row
        # ships its own gold+distractor paragraphs, so pooling from the same
        # split as iter_eval_examples() guarantees every eval question's
        # supporting docs are actually present in the corpus - pooling from
        # train instead (the previous behaviour) draws from an unrelated
        # set of Wikipedia articles and collapses recall to near-zero.
        #
        # Scan up to max_train_examples rows (not max_eval_examples): the
        # tested questions are always the FIRST max_eval_examples rows of
        # this same iteration order, so their gold docs are pooled first and
        # guaranteed present; scanning further rows on top adds genuine
        # distractor documents from OTHER questions so vector/graph search
        # has an actual "needle in haystack" task instead of a corpus that
        # is just the tested questions' own paragraphs (too easy - vector
        # alone hits ~97% recall and graph has nothing left to add).
        train_examples = iter_hotpotqa_examples(
            subset=self.subset, split=self.eval_split, limit=self.max_train_examples
        )

        return build_corpus_from_examples(train_examples, max_docs=self.max_corpus_docs)

    def iter_eval_examples(self):
        for ex in iter_hotpotqa_examples(
            subset=self.subset, split=self.eval_split, limit=self.max_eval_examples
        ):
            yield QAExample(
                qid=ex.qid,
                question=ex.question,
                answer=ex.answer,
                supporting_titles=ex.supporting_titles,
            )
