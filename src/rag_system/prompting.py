from __future__ import annotations

from rag_system.index import RetrievedChunk


def build_rag_messages(
    question: str, contexts: list[RetrievedChunk], max_context_chars: int = 12000
):
    parts = []
    total = 0

    for i, c in enumerate(contexts):
        s = f"[{i+1}] Title: {c.title}\n{c.text}"
        if total + len(s) > max_context_chars and total > 0:
            break
        parts.append(s)
        total += len(s)
    context_block = "\n\n".join(parts)

    system = (
        "You are a question answering assistant. "
        "Answer using ONLY the provided context. "
        "If the answer is a person/place/entity, output just the answer string. "
        "If it's a yes/no question, output exactly 'yes' or 'no'. "
        "If the context is insufficient, output 'unknown'."
    )

    user = f"Question: {question}\n\nContext:\n{context_block}\n\nAnswer:"

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
