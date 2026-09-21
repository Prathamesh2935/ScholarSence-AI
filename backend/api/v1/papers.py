"""Papers API — upload / parse endpoints (Module A)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

try:  # `uvicorn backend.main:app` from repo root
    from backend.services.ingestion import ParsedPaper, parse_pdf_bytes
    from backend.services.vector_store import RetrievedChunk, chunk_paper, get_store
except ImportError:  # `uvicorn main:app` from /backend
    from services.ingestion import ParsedPaper, parse_pdf_bytes  # type: ignore[no-redef]
    from services.vector_store import RetrievedChunk, chunk_paper, get_store  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB

# In-memory paper store (Step 5 compare/gaps build on these paper_ids).
# Swapped for a persistent store in production.
PAPER_STORE: dict[str, dict[str, Any]] = {}


class UploadResponse(BaseModel):
    paper: ParsedPaper


def _validate_pdf_upload(filename: str | None, content_type: str | None) -> None:
    name = (filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")
    if content_type and content_type not in (
        "application/pdf",
        "application/octet-stream",
        "binary/octet-stream",
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {content_type}. Expected a PDF.",
        )


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_paper(
    file: UploadFile = File(..., description="Research paper PDF"),
    enrich: bool = Query(
        default=True,
        description="Enrich title/authors/year via Semantic Scholar when possible.",
    ),
) -> UploadResponse:
    """Upload a PDF, parse sections + math, extract metadata, return JSON."""
    _validate_pdf_upload(file.filename, file.content_type)
    raw = await file.read()
    if len(raw) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds 50 MB limit.")
    try:
        # CPU-bound parsing off the event loop.
        paper = await asyncio.to_thread(
            parse_pdf_bytes, raw, file.filename or "paper.pdf", enrich
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ImportError as exc:
        logger.exception("missing backend dependency")
        raise HTTPException(status_code=500, detail=f"Server misconfigured: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("paper parsing failed")
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {exc}") from exc

    PAPER_STORE[paper.paper_id] = paper.model_dump()

    # Best-effort retrieval indexing (Step 3) — never fail an upload on it.
    try:
        indexed = await asyncio.to_thread(get_store().upsert_paper, paper)
        logger.debug("indexed paper=%s chunks=%d", paper.paper_id, indexed)
    except Exception as exc:
        logger.warning("indexing failed for paper=%s: %s", paper.paper_id, exc)

    return UploadResponse(paper=paper)


@router.get("")
def list_papers() -> dict[str, Any]:
    """List parsed papers held in the in-memory store."""
    return {
        "count": len(PAPER_STORE),
        "papers": [
            {
                "paper_id": p["paper_id"],
                "filename": p["filename"],
                "title": (p.get("metadata") or {}).get("title"),
                "year": (p.get("metadata") or {}).get("year"),
                "page_count": p.get("page_count"),
            }
            for p in PAPER_STORE.values()
        ],
    }


class SearchResponse(BaseModel):
    query: str
    backends: dict[str, str]
    results: list[RetrievedChunk]


@router.get("/search", response_model=SearchResponse)
def search_papers(
    q: str = Query(..., description="Natural-language or keyword query"),
    paper_id: list[str] | None = Query(
        default=None, description="Scope to paper_ids (repeatable)"
    ),
    top_k: int = Query(default=5, ge=1, le=20),
) -> SearchResponse:
    """Hybrid (dense + BM25) citation retrieval scoped to paper_ids."""
    store = get_store()
    results = store.hybrid_search(
        query=q, paper_ids=paper_id, top_k=top_k
    )
    return SearchResponse(query=q, backends=store.backend_names, results=results)


@router.get("/{paper_id}")
def get_paper(paper_id: str) -> dict[str, Any]:
    paper = PAPER_STORE.get(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper
