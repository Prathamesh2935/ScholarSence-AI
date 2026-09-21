"""Vector store & hybrid retrieval (Backend Core, Step 3).

Hybrid search = dense vector embeddings (``text-embedding-3-small`` via
the OpenAI SDK) fused with BM25 keyword matching via Reciprocal Rank
Fusion (RRF) for high-precision citation retrieval.

Graceful degradation (everything works offline):
- No ``OPENAI_API_KEY`` / no openai package -> deterministic hashed
  bag-of-words embeddings (L2-normalized, dim 256).
- No reachable Qdrant server -> ``QdrantClient(location=":memory:")`` ->
  pure-Python dict store with brute-force cosine.
- No rank-bm25 -> small built-in Okapi BM25 implementation.

Every chunk and every hit carries citation attribution:
``paper_id`` / ``section`` / ``page_number`` / ``paragraph_ids``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
from collections import Counter
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_DIM = 1536  # text-embedding-3-small
HASH_DIM = 256  # offline fallback embedding width

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333") or 6333)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "scholarsense_papers")

CHUNK_TARGET_CHARS = 1000
CHUNK_OVERLAP_PARAS = 1  # carry last paragraph as overlap when possible

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


# ---------------------------------------------------------------------------
# Chunk models + chunking (page / paragraph attribution preserved)
# ---------------------------------------------------------------------------


class TextChunk(BaseModel):
    chunk_id: str
    paper_id: str
    section: str = "Introduction"
    text: str
    page_number: int  # first page spanned by the chunk
    pages: list[int] = Field(default_factory=list)
    paragraph_ids: list[str] = Field(default_factory=list)
    token_count: int = 0


class Citation(BaseModel):
    paper_id: str
    section: str
    page_number: int
    pages: list[int] = Field(default_factory=list)
    paragraph_ids: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk: TextChunk
    citation: Citation
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0


def chunk_paper(
    paper: Any,
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_paras: int = CHUNK_OVERLAP_PARAS,
) -> list[TextChunk]:
    """Split a parsed paper into attributed chunks.

    Accepts a :class:`ParsedPaper` (or its ``model_dump()`` dict). Paragraph
    boundaries are never split; a chunk accumulates whole paragraphs up to
    ``target_chars``, then carries the last paragraph over as overlap.
    """
    data = paper.model_dump() if hasattr(paper, "model_dump") else dict(paper)
    paper_id = str(data.get("paper_id", "unknown"))
    chunks: list[TextChunk] = []
    pending: list[dict[str, Any]] = []  # paragraph dicts with section attached

    for sec in data.get("sections", []) or []:
        for para in sec.get("paragraphs", []) or []:
            if (para.get("text") or "").strip():
                pending.append({**para, "_section": sec.get("name", "Introduction")})

    idx = 0
    carry: list[dict[str, Any]] = []
    while idx < len(pending) or carry:
        current = list(carry)
        carry = []
        start_idx = idx
        size = sum(len(p.get("text", "")) for p in current)
        while idx < len(pending) and size < target_chars:
            current.append(pending[idx])
            size += len(pending[idx].get("text", ""))
            idx += 1
        if not current:
            break
        # Long single paragraph: hard-split on sentence boundaries.
        if len(current) == 1 and len(current[0].get("text", "")) > target_chars * 2:
            current = _hard_split_long_paragraph(current[0], target_chars)
        n = len(chunks)
        pages = sorted({int(p.get("page_number", 1)) for p in current})
        texts = [p.get("text", "") for p in current]
        chunk_text = "\n\n".join(texts)
        chunks.append(
            TextChunk(
                chunk_id=f"{paper_id}:c{n}",
                paper_id=paper_id,
                section=current[0].get("_section", "Introduction"),
                text=chunk_text,
                page_number=pages[0] if pages else 1,
                pages=pages,
                paragraph_ids=[p.get("paragraph_id", f"p{i}") for i, p in enumerate(current)],
                token_count=len(tokenize(chunk_text)),
            )
        )
        # Overlap only when new paragraphs were consumed; otherwise the next
        # iteration would re-emit the same carry forever.
        if overlap_paras and idx < len(pending) and idx > start_idx:
            carry = current[-overlap_paras:]
    return chunks


def _hard_split_long_paragraph(para: dict[str, Any], target: int) -> list[dict[str, Any]]:
    sentences = re.split(r"(?<=[.!?])\s+", para.get("text", ""))
    parts, buf = [], ""
    for sent in sentences:
        if buf and len(buf) + len(sent) > target:
            parts.append(buf)
            buf = sent
        else:
            buf = (buf + " " + sent).strip()
    if buf:
        parts.append(buf)
    return [{**para, "text": p} for p in parts] if parts else [para]


# ---------------------------------------------------------------------------
# Embeddings: OpenAI SDK with offline hash fallback
# ---------------------------------------------------------------------------


class EmbeddingBackend:
    name = "base"
    dim = HASH_DIM

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashEmbeddingBackend(EmbeddingBackend):
    """Deterministic hashed bag-of-words embeddings (offline fallback)."""

    name = "hash-fallback"
    dim = HASH_DIM

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
            vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """Dense embeddings via the OpenAI SDK (text-embedding-3-small)."""

    name = "openai:text-embedding-3-small"
    dim = OPENAI_DIM

    def __init__(self, model: str = EMBEDDING_MODEL):
        from openai import OpenAI  # type: ignore

        self._client = OpenAI()
        self._model = model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, max(len(texts), 1), 100):
            batch = list(texts[i : i + 100]) or [""]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend([list(map(float, d.embedding)) for d in resp.data])
        return out[: len(texts)]

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=[text or " "])
        return list(map(float, resp.data[0].embedding))


def get_embedding_backend() -> EmbeddingBackend:
    if os.getenv("OPENAI_API_KEY"):
        try:
            backend = OpenAIEmbeddingBackend()
            logger.info("embeddings: using %s", backend.name)
            return backend
        except Exception as exc:
            logger.warning("openai embeddings unavailable (%s); hash fallback", exc)
    return HashEmbeddingBackend()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


# ---------------------------------------------------------------------------
# BM25: rank-bm25 when available, else built-in Okapi
# ---------------------------------------------------------------------------


class _BuiltinBM25:
    """Minimal Okapi BM25 (k1=1.5, b=0.75) over tokenized docs."""

    def __init__(self, docs: list[list[str]]):
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1) or 1.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for doc in self.docs:
            tf: Counter[str] = Counter(doc)
            dl = len(doc) or 1
            s = 0.0
            for t in query:
                if t not in self.idf:
                    continue
                f = tf.get(t, 0)
                denom = f + 1.5 * (1 - 0.75 + 0.75 * dl / self.avgdl)
                s += self.idf[t] * (f * 2.5 / denom if denom else 0.0)
            scores.append(s)
        return scores


# ---------------------------------------------------------------------------
# Hybrid store
# ---------------------------------------------------------------------------


class HybridVectorStore:
    """Chunk index with dense + BM25 hybrid search and RRF fusion.

    Dense vectors live in Qdrant (server, :memory:, or pure-Python fallback);
    BM25 always runs over the local chunk registry so keyword search never
    depends on the network.
    """

    def __init__(
        self,
        collection: str = QDRANT_COLLECTION,
        embedding_backend: Optional[EmbeddingBackend] = None,
    ):
        self.collection = collection
        self.embeddings = embedding_backend or get_embedding_backend()
        self._lock = threading.RLock()
        self._chunks: dict[str, TextChunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._qdrant = None
        self._qdrant_mode = "python"
        self._connect_qdrant()

    # -- Qdrant connection (server -> :memory: -> pure python) --------------
    def _connect_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import Distance, VectorParams  # type: ignore
        except Exception as exc:
            logger.info("qdrant-client unavailable (%s); python fallback", exc)
            return
        try:
            client = QdrantClient(
                host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=5
            )
            client.get_collections()  # probe: server reachable?
            mode = "server"
        except Exception as exc:
            logger.warning("qdrant server unreachable (%s); :memory: fallback", exc)
            try:
                client = QdrantClient(location=":memory:")
                mode = "memory"
            except Exception as exc2:
                logger.warning("qdrant :memory: failed (%s); python fallback", exc2)
                return
        try:
            from qdrant_client.models import Distance, VectorParams  # type: ignore

            existing = client.get_collection(self.collection)
            if existing.config.params.vectors.size != self.embeddings.dim:  # type: ignore[union-attr]
                client.delete_collection(self.collection)
                raise ValueError("dim mismatch -> recreate")
        except Exception:
            try:
                client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.embeddings.dim, distance=Distance.COSINE
                    ),
                )
            except Exception as exc:
                logger.warning("qdrant collection setup failed (%s)", exc)
                return
        self._qdrant = client
        self._qdrant_mode = mode
        logger.info("qdrant backend: %s (collection=%s)", mode, self.collection)

    @property
    def backend_names(self) -> dict[str, str]:
        return {"embeddings": self.embeddings.name, "qdrant": self._qdrant_mode}

    # -- writes --------------------------------------------------------------
    def upsert_paper(
        self,
        paper: Any,
        target_chars: int = CHUNK_TARGET_CHARS,
        overlap_paras: int = CHUNK_OVERLAP_PARAS,
    ) -> int:
        chunks = chunk_paper(
            paper, target_chars=target_chars, overlap_paras=overlap_paras
        )
        if not chunks:
            return 0
        with self._lock:
            vectors = self.embeddings.embed_documents([c.text for c in chunks])
            for chunk, vec in zip(chunks, vectors):
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = list(map(float, vec))
            self._push_to_qdrant(chunks, vectors)
        logger.info(
            "indexed paper=%s chunks=%d", chunks[0].paper_id, len(chunks)
        )
        return len(chunks)

    def _push_to_qdrant(
        self, chunks: list[TextChunk], vectors: list[list[float]]
    ) -> None:
        if self._qdrant is None:
            return
        try:
            from qdrant_client.models import PointStruct  # type: ignore

            self._qdrant.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=abs(int(hashlib.md5(c.chunk_id.encode()).hexdigest(), 16))
                        % (2**63),
                        vector=list(map(float, v)),
                        payload={
                            "chunk_id": c.chunk_id,
                            "paper_id": c.paper_id,
                            "section": c.section,
                        },
                    )
                    for c, v in zip(chunks, vectors)
                ],
            )
        except Exception as exc:
            logger.warning("qdrant upsert failed (%s); local index still current", exc)

    def delete_paper(self, paper_id: str) -> int:
        with self._lock:
            doomed = [cid for cid, c in self._chunks.items() if c.paper_id == paper_id]
            for cid in doomed:
                self._chunks.pop(cid, None)
                self._vectors.pop(cid, None)
            if self._qdrant is not None and doomed:
                try:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

                    self._qdrant.delete(
                        collection_name=self.collection,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="paper_id", match=MatchValue(value=paper_id)
                                )
                            ]
                        ),
                    )
                except Exception as exc:
                    logger.warning("qdrant delete failed (%s)", exc)
            return len(doomed)

    # -- reads ---------------------------------------------------------------
    def _dense_ranked(
        self, query_vec: list[float], candidates: list[TextChunk], limit: int
    ) -> list[tuple[str, float]]:
        if self._qdrant is not None:
            try:
                hits = self._qdrant.query_points(
                    collection_name=self.collection,
                    query=list(map(float, query_vec)),
                    limit=limit,
                ).points
                out = []
                for h in hits:
                    payload = h.payload or {}
                    cid = payload.get("chunk_id")
                    if cid in self._chunks:
                        out.append((cid, float(h.score)))
                if out:
                    return out
            except Exception as exc:
                logger.debug("qdrant query failed (%s); local cosine", exc)
        scored = [
            (c.chunk_id, cosine(query_vec, self._vectors.get(c.chunk_id, [])))
            for c in candidates
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:limit]

    def _bm25_scores(
        self, query: str, candidates: list[TextChunk]
    ) -> dict[str, float]:
        if not candidates:
            return {}
        tokenized = [tokenize(c.text) for c in candidates]
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            scores = BM25Okapi(tokenized).get_scores(tokenize(query))
        except Exception:
            scores = _BuiltinBM25(tokenized).get_scores(tokenize(query))
        return {c.chunk_id: float(s) for c, s in zip(candidates, scores)}

    def hybrid_search(
        self,
        query: str,
        paper_ids: Optional[Sequence[str]] = None,
        top_k: int = 5,
        dense_weight: float = 0.6,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """Dense + BM25 hybrid search fused with Reciprocal Rank Fusion.

        ``paper_ids`` scopes retrieval (multi-paper comparison uses several).
        """
        if not (query or "").strip():
            return []
        with self._lock:
            candidates = [
                c
                for c in self._chunks.values()
                if paper_ids is None or c.paper_id in set(paper_ids)
            ]
            if not candidates:
                return []
            pool = max(top_k * 2, 10)
            query_vec = self.embeddings.embed_query(query)
            dense_ranked = self._dense_ranked(query_vec, candidates, pool)
            bm25_map = self._bm25_scores(query, candidates)
            bm25_ranked = sorted(bm25_map.items(), key=lambda t: t[1], reverse=True)[
                :pool
            ]

            rrf: dict[str, float] = {}
            for rank, (cid, _) in enumerate(dense_ranked, start=1):
                rrf[cid] = rrf.get(cid, 0.0) + dense_weight / (rrf_k + rank)
            for rank, (cid, _) in enumerate(bm25_ranked, start=1):
                if bm25_map.get(cid, 0.0) <= 0:
                    continue
                rrf[cid] = rrf.get(cid, 0.0) + (1 - dense_weight) / (rrf_k + rank)

            # Fallback: pure lexical overlap when both rankers miss (e.g. tiny
            # toy corpora where BM25 idf collapses) — keeps citation retrieval
            # functional rather than empty.
            if not rrf:
                qset = set(tokenize(query))
                for c in candidates:
                    overlap = len(qset & set(tokenize(c.text)))
                    if overlap:
                        rrf[c.chunk_id] = float(overlap)

            dense_map = dict(dense_ranked)
            ranked = sorted(rrf.items(), key=lambda t: t[1], reverse=True)[:top_k]
            results = []
            for cid, fused in ranked:
                chunk = self._chunks[cid]
                results.append(
                    RetrievedChunk(
                        chunk=chunk,
                        citation=Citation(
                            paper_id=chunk.paper_id,
                            section=chunk.section,
                            page_number=chunk.page_number,
                            pages=chunk.pages,
                            paragraph_ids=chunk.paragraph_ids,
                        ),
                        score=round(fused, 6),
                        dense_score=round(dense_map.get(cid, 0.0), 6),
                        bm25_score=round(bm25_map.get(cid, 0.0), 6),
                    )
                )
            return results


_STORE: Optional[HybridVectorStore] = None
_STORE_LOCK = threading.Lock()


def get_store() -> HybridVectorStore:
    """Process-wide singleton store (shared by API routers)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = HybridVectorStore()
        return _STORE


def reset_store(
    embedding_backend: Optional[EmbeddingBackend] = None,
    collection: str = QDRANT_COLLECTION,
) -> HybridVectorStore:
    """Replace the singleton (tests use this to isolate state)."""
    global _STORE
    with _STORE_LOCK:
        _STORE = HybridVectorStore(
            collection=collection,
            embedding_backend=embedding_backend or HashEmbeddingBackend(),
        )
        return _STORE
