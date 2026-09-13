from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rag_system.schema import Document


@dataclass(frozen=True)
class RawExample:
    """
    One raw QA example as loaded from a HuggingFace dataset, together with
    the context paragraphs it shipped with.

    `contexts` is what build_corpus_from_contexts() below turns into a
    training corpus. Every loader in this package is "self-contained": the
    gold/distractor passages used to build the RAG corpus come from the same
    HF rows as the eval questions, so no separate multi-GB Wikipedia dump is
    needed to run these experiments locally.
    """

    qid: str
    question: str
    answer: str
    supporting_titles: tuple[str, ...]
    contexts: tuple[tuple[str, str], ...]  # (title, paragraph_text) pairs


def _doc_id(title: str) -> str:
    return hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]


def build_corpus_from_contexts(examples, max_docs: int | None = None) -> list[Document]:
    """
    Deduplicate (by title, first occurrence wins) the context paragraphs
    carried by `examples` into a flat list of Document, capped at `max_docs`.
    """
    seen: set[str] = set()
    docs: list[Document] = []
    for ex in examples:
        for title, text in ex.contexts:
            title = (title or "").strip()
            text = (text or "").strip()
            if not title or not text or title in seen:
                continue
            seen.add(title)
            docs.append(Document(doc_id=_doc_id(title), title=title, text=text))
            if max_docs is not None and len(docs) >= max_docs:
                return docs
    return docs


def extract_hotpot_style_context(context) -> list[tuple[str, str]]:
    """
    HotpotQA-family datasets (HotpotQA itself and most 2WikiMultiHopQA HF
    mirrors, since 2Wiki reuses HotpotQA's JSON layout) ship `context` either
    as a struct-of-arrays ({"title": [...], "sentences": [[...]]}) - the
    shape Arrow/HF typically normalizes mixed-type nested lists into - or as
    a list of [title, sentences] pairs. Handle both.
    """
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        return [(t, " ".join(s)) for t, s in zip(titles, sentences, strict=False)]
    pairs = []
    for item in context:
        title, sentences = item[0], item[1]
        text = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
        pairs.append((title, text))
    return pairs


def extract_hotpot_style_supporting_titles(supporting_facts) -> tuple[str, ...]:
    """Same struct-of-arrays vs list-of-pairs ambiguity as context, for supporting_facts."""
    if isinstance(supporting_facts, dict):
        return tuple(sorted(set(supporting_facts.get("title", []))))
    return tuple(sorted({item[0] for item in supporting_facts}))
