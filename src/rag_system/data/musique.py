
from __future__ import annotations

from collections.abc import Iterator

from rag_system.data._common import RawExample, build_corpus_from_contexts
from rag_system.schema import Document

HF_DATASET_ID = "dgslibisey/MuSiQue"


def iter_musique_examples(split: str, limit: int | None = None) -> Iterator[RawExample]:
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(HF_DATASET_ID, split=split)

    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        answer = row.get("answer") or ""
        if not answer:
            continue
        paragraphs = row["paragraphs"]
        contexts = tuple((p["title"], p["paragraph_text"]) for p in paragraphs)
        supporting_titles = tuple(
            sorted({p["title"] for p in paragraphs if p.get("is_supporting")})
        )
        yield RawExample(
            qid=str(row.get("id") or row.get("_id")),
            question=row["question"],
            answer=answer,
            supporting_titles=supporting_titles,
            contexts=contexts,
        )


def build_musique_corpus(examples, max_docs: int | None = None) -> list[Document]:
    return build_corpus_from_contexts(examples, max_docs=max_docs)
