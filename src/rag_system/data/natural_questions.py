
from __future__ import annotations

import re
from collections.abc import Iterator

from rag_system.data._common import RawExample, build_corpus_from_contexts
from rag_system.schema import Document

HF_DATASET_ID = "mrqa-workshop/mrqa"
NQ_SUBSET_NAME = "NaturalQuestionsShort"

_TLE_RE = re.compile(r"\[TLE\](.*?)\[PAR\]", re.DOTALL)
_MARKER_RE = re.compile(r"\[(TLE|PAR|DOC|SEP)\]")


def _mrqa_title(context: str, fallback: str) -> str:
    m = _TLE_RE.search(context)
    if not m:
        return fallback
    title = m.group(1).strip()
    return title[:200] if title else fallback


def _clean_mrqa_context(context: str) -> str:
    return _MARKER_RE.sub(" ", context).strip()


def iter_nq_examples(split: str, limit: int | None = None) -> Iterator[RawExample]:
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(HF_DATASET_ID, split=split)
    ds = ds.filter(lambda row: row.get("subset") == NQ_SUBSET_NAME)

    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        answers = row.get("answers") or []
        if not answers:
            continue
        qid = str(row["qid"])
        raw_context = row["context"]
        title = _mrqa_title(raw_context, fallback=f"nq_{qid}")
        text = _clean_mrqa_context(raw_context)
        yield RawExample(
            qid=qid,
            question=row["question"],
            answer=answers[0],
            supporting_titles=(title,),
            contexts=((title, text),),
        )


def build_nq_corpus(examples, max_docs: int | None = None) -> list[Document]:
    return build_corpus_from_contexts(examples, max_docs=max_docs)
