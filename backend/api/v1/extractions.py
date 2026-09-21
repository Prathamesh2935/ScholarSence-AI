"""Extraction API — structured summaries / methodology / results (Module B)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

try:  # `uvicorn backend.main:app` from repo root
    from backend.api.v1.papers import PAPER_STORE
    from backend.schemas.paper_schema import PaperExtraction
    from backend.services.extractor import extract_paper, validate_citations
except ImportError:  # `uvicorn main:app` from /backend
    from api.v1.papers import PAPER_STORE  # type: ignore[no-redef]
    from schemas.paper_schema import PaperExtraction  # type: ignore[no-redef]
    from services.extractor import extract_paper, validate_citations  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_paper_or_404(paper_id: str) -> dict:
    paper = PAPER_STORE.get(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper


@router.post("/{paper_id}/extract", response_model=PaperExtraction)
async def extract_structured(
    paper_id: str,
    prefer: str = Query(
        default="auto",
        description="'auto' (OpenAI when configured, else heuristic), 'openai', or 'heuristic'",
    ),
    model: str = Query(default="gpt-4o-mini", description="OpenAI model for extraction"),
) -> PaperExtraction:
    """Run the structured extraction engine over a parsed paper."""
    if prefer not in ("auto", "openai", "heuristic"):
        raise HTTPException(status_code=400, detail="prefer must be auto|openai|heuristic")
    paper = _get_paper_or_404(paper_id)
    try:
        extraction = await asyncio.to_thread(extract_paper, paper, prefer, model)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("extraction failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
    PAPER_STORE[paper_id]["extraction"] = extraction.model_dump()
    return extraction


@router.get("/{paper_id}/extraction", response_model=PaperExtraction)
def get_extraction(paper_id: str) -> PaperExtraction:
    """Return the stored extraction for a paper (404 if never extracted)."""
    paper = _get_paper_or_404(paper_id)
    stored = paper.get("extraction")
    if stored is None:
        raise HTTPException(
            status_code=404, detail="No extraction yet. POST /extract first."
        )
    return PaperExtraction.model_validate(stored)


@router.get("/{paper_id}/extraction/citations")
def check_extraction_citations(paper_id: str) -> dict:
    """Citation-attribution integrity check for a paper's extraction."""
    paper = _get_paper_or_404(paper_id)
    stored = paper.get("extraction")
    if stored is None:
        raise HTTPException(
            status_code=404, detail="No extraction yet. POST /extract first."
        )
    extraction = PaperExtraction.model_validate(stored)
    issues = validate_citations(extraction, paper)
    return {"paper_id": paper_id, "clean": not issues, "issues": issues}
