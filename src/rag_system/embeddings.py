from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm


def _hash_text(text):
    """
    Compute a stable hash for text to use as cache key
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class RemoteEmbedder:
    """
    Embedding client for OpenAI-API-compatible HTTP servers.
    Use base_url and model to switch between different providers and models.
    """

    base_url: str
    api_key: str
    model: str
    batch_size: int = 64
    show_progress: bool = True

    def _client(self):
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def embed_texts(self, texts, _retries: int = 5):
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        import time
        client = self._client()
        embs: list[list[float]] = []

        it: Iterable[range] = range(0, len(texts), self.batch_size)
        if self.show_progress:
            it = tqdm(
                it, desc="Embedding", total=(len(texts) + self.batch_size - 1) // self.batch_size
            )

        for start in it:
            batch = texts[start : start + self.batch_size]
            for attempt in range(_retries):
                try:
                    resp = client.embeddings.create(model=self.model, input=batch)
                    for item in resp.data:
                        embs.append(item.embedding)
                    break
                except Exception:
                    if attempt < _retries - 1:
                        # increasing backoff (3s/6s/12s/24s) instead of a
                        # flat 2s - a VPN drop can take longer than that to
                        # recover, and this is corpus indexing (thousands of
                        # chunks): losing the whole run here is expensive.
                        time.sleep(min(3 * (2**attempt), 30))
                        client = self._client()
                    else:
                        raise

        arr = np.asarray(embs, dtype=np.float32)

        return arr

    def embed_query(self, text):
        vecs = self.embed_texts([text])
        return vecs[0]


@dataclass
class EmbeddingCache:
    """
    Simple file-backed cache for text embeddings.
    The cache is keyed by a hash of the input text.
    """

    path: Path
    _store: dict[str, list[float]]

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {}
        return cls(path=path, _store=data)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.path.write_text(json.dumps(self._store), encoding="utf-8")

    def get(self, text):
        key = _hash_text(text)
        vec = self._store.get(key)

        if vec is None:
            return None
        return np.asarray(vec, dtype=np.float32)

    def set(self, text, embedding):
        key = _hash_text(text)
        self._store[key] = embedding.astype(np.float32).tolist()


@dataclass
class CachedEmbedder:
    """
    Wrapper around RemoteEmbedder with an embedding cache.
    The public API mirrors the base embedder: embed_texts and embed_query.
    """

    base: RemoteEmbedder
    cache: EmbeddingCache | None = None

    def embed_texts(self, texts):
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        cached_vectors: list[np.ndarray | None] = [None] * len(texts)
        to_embed: list[tuple[int, str]] = []

        for i, t in enumerate(texts):
            if self.cache is not None:
                vec = self.cache.get(t)
                if vec is not None:
                    cached_vectors[i] = vec
                    continue
            to_embed.append((i, t))

        if to_embed:
            batch_texts = [t for _, t in to_embed]
            new_embs = self.base.embed_texts(batch_texts)
            for (i, _), vec in zip(to_embed, new_embs, strict=False):
                cached_vectors[i] = vec
                if self.cache is not None:
                    self.cache.set(texts[i], vec)

        mat = np.stack(cached_vectors, axis=0)
        return mat.astype(np.float32)

    def embed_query(self, text):
        mat = self.embed_texts([text])
        return mat[0]
