from __future__ import annotations

from rag_system.datasets.hotpotqa import HotpotQADataset
from rag_system.datasets.local_files import LocalFilesDataset
from rag_system.datasets.musique import MusiqueDataset
from rag_system.datasets.natural_questions import NaturalQuestionsDataset
from rag_system.datasets.triviaqa import TriviaQADataset
from rag_system.datasets.wiki_multihop import Wiki2MultiHopDataset


def build_dataset(cfg):
    """
    Build a dataset adapter from the run configuration.
    """

    if cfg.dataset.name == "hotpotqa":
        hp = cfg.hotpotqa
        return HotpotQADataset(
            subset=hp.subset,
            train_split=hp.train_split,
            eval_split=hp.eval_split,
            max_train_examples=hp.max_train_examples,
            max_corpus_docs=hp.max_corpus_docs,
            max_eval_examples=hp.max_eval_examples,
        )

    if cfg.dataset.name == "natural_questions":
        nq = cfg.natural_questions
        return NaturalQuestionsDataset(
            eval_split=nq.eval_split,
            max_corpus_docs=nq.max_corpus_docs,
            max_eval_examples=nq.max_eval_examples,
        )

    if cfg.dataset.name == "musique":
        mq = cfg.musique
        return MusiqueDataset(
            train_split=mq.train_split,
            eval_split=mq.eval_split,
            max_corpus_docs=mq.max_corpus_docs,
            max_eval_examples=mq.max_eval_examples,
        )

    if cfg.dataset.name == "wiki2multihop":
        w2 = cfg.wiki2multihop
        return Wiki2MultiHopDataset(
            train_split=w2.train_split,
            eval_split=w2.eval_split,
            max_corpus_docs=w2.max_corpus_docs,
            max_eval_examples=w2.max_eval_examples,
        )

    if cfg.dataset.name == "triviaqa":
        tqa = cfg.triviaqa
        return TriviaQADataset(
            eval_split=tqa.eval_split,
            max_corpus_docs=tqa.max_corpus_docs,
            max_eval_examples=tqa.max_eval_examples,
        )

    if cfg.dataset.name == "local_files":
        lf = cfg.local_files
        return LocalFilesDataset(
            corpus_dir=lf.corpus_dir,
            eval_jsonl=lf.eval_jsonl,
            max_corpus_docs=lf.max_corpus_docs,
        )

    raise ValueError(f"Unknown dataset name: {cfg.dataset.name!r}")
