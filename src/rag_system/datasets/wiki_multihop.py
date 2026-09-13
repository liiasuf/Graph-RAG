from __future__ import annotations

from dataclasses import dataclass

from rag_system.data.wiki_multihop import build_wiki2_corpus, iter_wiki2_examples
from rag_system.datasets.base import QADataset, QAExample


@dataclass
class Wiki2MultiHopDataset(QADataset):
    train_split: str
    eval_split: str
    max_corpus_docs: int | None
    max_eval_examples: int | None

    name: str = "wiki2multihop"

    def build_train_corpus(self):
        # Pool contexts from eval_split (not train_split): guarantees every
        # eval question's supporting docs are in the corpus (see the same
        # fix in datasets/hotpotqa.py for the full rationale).
        #
        # Scan far more rows (20_000, effectively the whole split) than
        # max_eval_examples - see the same reasoning in datasets/musique.py.
        examples = list(iter_wiki2_examples(split=self.eval_split, limit=20_000))
        return build_wiki2_corpus(examples, max_docs=self.max_corpus_docs)

    def iter_eval_examples(self):
        for ex in iter_wiki2_examples(split=self.eval_split, limit=self.max_eval_examples):
            if not ex.answer:
                continue
            yield QAExample(
                qid=ex.qid,
                question=ex.question,
                answer=ex.answer,
                supporting_titles=ex.supporting_titles,
            )
