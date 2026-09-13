from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from rag_system.data._common import (
    RawExample,
    build_corpus_from_contexts,
    extract_hotpot_style_context,
    extract_hotpot_style_supporting_titles,
)
from rag_system.schema import Document

HF_DATASET_ID = "xanhho/2WikiMultihopQA"
LOCAL_DATA_DIR_ENV = "WIKI2_DATA_DIR"
_LOCAL_SPLIT_FILENAMES = {"train": "train.json", "validation": "dev.json", "dev": "dev.json", "test": "test.json"}


def _load_local_json_rows(split: str) -> list[dict] | None:
    data_dir = os.environ.get(LOCAL_DATA_DIR_ENV)
    if not data_dir:
        return None
    filename = _LOCAL_SPLIT_FILENAMES.get(split, f"{split}.json")
    path = Path(data_dir) / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# The yaml configs use "dev" (matching the official release's dev.json,
# used by the local-JSON fallback), but the HF parquet mirror's split is
# named "validation" - normalize only for the HF path.
_HF_SPLIT_ALIASES = {"dev": "validation"}


def _load_hf_rows(split: str):
    import datasets as hf_datasets

    hf_split = _HF_SPLIT_ALIASES.get(split, split)
    try:
        return hf_datasets.load_dataset(HF_DATASET_ID, split=hf_split)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" not in str(e):
            raise
        return hf_datasets.load_dataset(HF_DATASET_ID, split=hf_split, revision="refs/convert/parquet")


def _load_rows(split: str):
    local_rows = _load_local_json_rows(split)
    if local_rows is not None:
        return local_rows
    try:
        return _load_hf_rows(split)
    except Exception as e:
        raise RuntimeError(
            "Could not load 2WikiMultiHopQA from the HF Hub "
            f"({HF_DATASET_ID!r}, split={split!r}): {e}\n"
            "Fallback: download the official release from "
            "https://github.com/Alab-NII/2wikimultihop (Google Drive link in "
            "its README, file data_ids_april7.zip), unzip it, and set "
            f"{LOCAL_DATA_DIR_ENV}=/path/to/that/folder before running."
        ) from e


def _maybe_json_loads(value):
    """
    The HF parquet conversion of xanhho/2WikiMultihopQA stores context/
    supporting_facts as JSON-encoded strings rather than native nested
    lists (confirmed by inspecting a row: `context` comes back as
    `'[["Title", ["sentence", ...]], ...]'`). Local JSON files (the official
    release, via WIKI2_DATA_DIR) already have these as real nested lists, so
    only parse when we actually got a string.
    """
    if isinstance(value, str):
        return json.loads(value)
    return value


def iter_wiki2_examples(split: str, limit: int | None = None) -> Iterator[RawExample]:
    rows = _load_rows(split)

    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        contexts = tuple(extract_hotpot_style_context(_maybe_json_loads(row["context"])))
        supporting_titles = extract_hotpot_style_supporting_titles(
            _maybe_json_loads(row["supporting_facts"])
        )
        qid = str(row.get("_id") or row.get("id") or row.get("qid"))
        answer = row.get("answer") or ""
        if not answer:
            continue
        yield RawExample(
            qid=qid,
            question=row["question"],
            answer=answer,
            supporting_titles=supporting_titles,
            contexts=contexts,
        )


def build_wiki2_corpus(examples, max_docs: int | None = None) -> list[Document]:
    return build_corpus_from_contexts(examples, max_docs=max_docs)
