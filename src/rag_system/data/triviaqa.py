
from __future__ import annotations

from collections.abc import Iterator

from rag_system.data._common import RawExample, build_corpus_from_contexts
from rag_system.schema import Document

HF_DATASET_ID = "mandarjoshi/trivia_qa"
HF_DATASET_CONFIG = "rc"


def iter_triviaqa_examples(split: str, limit: int | None = None) -> Iterator[RawExample]:
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(HF_DATASET_ID, HF_DATASET_CONFIG, split=split)

    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        answer_obj = row.get("answer") or {}
        answer = answer_obj.get("value") or ""
        if not answer:
            continue
        titles = row.get("entity_pages", {}).get("title", [])
        wiki_contexts = row.get("entity_pages", {}).get("wiki_context", [])
        contexts = tuple(
            (t, c) for t, c in zip(titles, wiki_contexts, strict=False) if t and c
        )
        yield RawExample(
            qid=str(row["question_id"]),
            question=row["question"],
            answer=answer,
            supporting_titles=tuple(t for t, _ in contexts),
            contexts=contexts,
        )


def build_triviaqa_corpus(examples, max_docs: int | None = None) -> list[Document]:
    return build_corpus_from_contexts(examples, max_docs=max_docs)
