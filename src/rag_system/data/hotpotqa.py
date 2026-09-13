from __future__ import annotations

from collections.abc import Iterator

from rag_system.data._common import (
    RawExample,
    build_corpus_from_contexts,
    extract_hotpot_style_context,
    extract_hotpot_style_supporting_titles,
)
from rag_system.schema import Document

HF_DATASET_ID = "hotpotqa/hotpot_qa"


def iter_hotpotqa_examples(subset: str, split: str, limit: int | None = None) -> Iterator[RawExample]:
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(HF_DATASET_ID, subset, split=split)

    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        contexts = tuple(extract_hotpot_style_context(row["context"]))
        supporting_titles = extract_hotpot_style_supporting_titles(row["supporting_facts"])
        yield RawExample(
            qid=str(row["id"]),
            question=row["question"],
            answer=row["answer"],
            supporting_titles=supporting_titles,
            contexts=contexts,
        )


def build_corpus_from_examples(examples, max_docs: int | None = None) -> list[Document]:
    return build_corpus_from_contexts(examples, max_docs=max_docs)
