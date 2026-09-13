from __future__ import annotations

from dataclasses import dataclass

from rag_system.data.documents import load_documents_from_dir
from rag_system.datasets.base import QADataset, QAExample


@dataclass
class LocalFilesDataset(QADataset):
    """
    Dataset adapter for a local directory of PDF/DOCX/TXT files.
    Eval examples are loaded from an optional JSONL file (qid, question, answer).
    If no eval file is provided, returns an empty eval set - useful for
    corpus indexing + ad-hoc querying without ground-truth evaluation.
    """
    corpus_dir: str
    eval_jsonl: str | None
    max_corpus_docs: int | None

    name: str = "local_files"

    def build_train_corpus(self):
        return load_documents_from_dir(self.corpus_dir, max_docs=self.max_corpus_docs)

    def iter_eval_examples(self):
        if not self.eval_jsonl:
            return
        import json
        from pathlib import Path
        path = Path(self.eval_jsonl)
        if not path.exists():
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield QAExample(
                    qid=str(d.get("id", d.get("qid", ""))),
                    question=str(d["question"]),
                    answer=str(d.get("answer", "")),
                    supporting_titles=tuple(d.get("supporting_titles", [])),
                )
