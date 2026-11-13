"""Retrieval is independent of LangGraph. Only the embedding protocol needs an API."""

import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Literal, Protocol

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from pharma_agent.documents import CHUNKER_VERSION, Chunk, corpus_version


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> np.ndarray: ...


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())


def normalize(vectors: np.ndarray) -> np.ndarray:
    data = np.asarray(vectors, dtype=np.float32)
    if data.ndim != 2 or not data.shape[1] or not np.isfinite(data).all():
        raise ValueError("Invalid embedding matrix")
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero-length embedding vector")
    return np.ascontiguousarray(data / norms)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    if k < 1:
        raise ValueError("RRF k must be positive")
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            if item not in seen:
                scores[item] += 1 / (k + rank)
                seen.add(item)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class RetrievalEngine:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder):
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Corpus/vector count mismatch")
        self.chunks = chunks
        self.by_id = {chunk.id: chunk for chunk in chunks}
        self.embedder = embedder
        self.vectors = normalize(vectors)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)
        self.bm25 = BM25Okapi([tokenize(c.embedding_text) for c in chunks])
        self.version = corpus_version(chunks)
        self.cache_hit = False

    @classmethod
    async def build(cls, chunks, embedder, cache_dir: Path, model: str, provider: str = ""):
        manifest = {
            "model": model,
            "provider_hash": hashlib.sha256(provider.encode()).hexdigest(),
            "corpus_version": corpus_version(chunks),
            "chunker": CHUNKER_VERSION,
            "ids": [chunk.id for chunk in chunks],
        }
        key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{key}.npz"
        vectors, hit = None, False
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    meta = json.loads(str(cached["metadata"]))
                    loaded = cached["vectors"]
                    valid = (
                        meta["manifest"] == manifest
                        and loaded.ndim == 2
                        and loaded.shape == (len(chunks), meta["dimension"])
                        and meta["checksum"] == hashlib.sha256(loaded.tobytes()).hexdigest()
                    )
                    if valid:
                        vectors = normalize(loaded)
                        hit = True
            except (OSError, ValueError, KeyError, TypeError):
                pass  # Corrupted/stale cache is rebuilt, never deserialized with pickle.
        if vectors is None:
            vectors = normalize(await embedder.embed([c.embedding_text for c in chunks]))
            if len(vectors) != len(chunks):
                raise ValueError("Embedding count mismatch")
            meta = {
                "manifest": manifest,
                "dimension": vectors.shape[1],
                "checksum": hashlib.sha256(vectors.tobytes()).hexdigest(),
            }
            handle, temp_path = tempfile.mkstemp(suffix=".npz", dir=cache_dir)
            try:
                with os.fdopen(handle, "wb") as stream:
                    np.savez_compressed(stream, vectors=vectors, metadata=json.dumps(meta))
                os.replace(temp_path, cache_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        engine = cls(chunks, vectors, embedder)
        engine.cache_hit = hit
        return engine

    def dense(self, query_vector: np.ndarray, top_k: int = 6) -> list[tuple[int, float]]:
        query = normalize(np.asarray(query_vector).reshape(1, -1))
        if query.shape[1] != self.index.d:
            raise ValueError(
                "Embedding dimensions changed; rebuild the cache with the current model"
            )
        scores, indices = self.index.search(query, min(top_k, len(self.chunks)))
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0], strict=True)
            if idx >= 0
        ]

    def lexical(self, query: str, top_k: int = 6) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        indices = sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))
        return [(idx, float(scores[idx])) for idx in indices[:top_k] if scores[idx] > 0]

    def rank(self, query: str, vector: np.ndarray, top_k=6, mode="hybrid"):
        if mode == "dense":
            return self.dense(vector, top_k)
        if mode != "hybrid":
            raise ValueError("Unknown retrieval mode")
        candidates = max(24, top_k * 4)
        return reciprocal_rank_fusion(
            [
                [idx for idx, _ in self.dense(vector, candidates)],
                [idx for idx, _ in self.lexical(query, candidates)],
            ]
        )[:top_k]

    async def search(self, query: str, top_k: int = 6, mode: Literal["dense", "hybrid"] = "hybrid"):
        if not query.strip() or len(query) > 2000 or not 1 <= top_k <= 12:
            raise ValueError("Invalid search query or top_k")
        vector = (await self.embedder.embed([query]))[0]
        return [(self.chunks[idx], score) for idx, score in self.rank(query, vector, top_k, mode)]

    def read_section(self, source: str, section: str, offset=0, limit=6) -> dict:
        # Lookups use only in-memory inventory; model-supplied paths are never opened.
        if offset < 0 or not 1 <= limit <= 6:
            raise ValueError("Invalid section pagination")
        available = sorted({c.section for c in self.chunks if c.source == source})
        if not available:
            raise ValueError("Unknown document. Use source names returned by search_documents.")
        matches = [s for s in available if s.casefold() == section.casefold()]
        if not matches:
            matches = [s for s in available if section.casefold() in s.casefold()]
        if len(matches) != 1:
            return {
                "error": "Section missing or ambiguous; choose an exact section.",
                "sections": available,
            }
        chunks = [c for c in self.chunks if c.source == source and c.section == matches[0]]
        return {
            "evidence": chunks[offset : offset + limit],
            "total": len(chunks),
            "next_offset": offset + limit if offset + limit < len(chunks) else None,
        }
