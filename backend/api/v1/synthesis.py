"""Synthesis API — multi-paper compare & research gaps (Modules C & D)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

try:  # `uvicorn backend.main:app` from repo root
    from backend.api.v1.papers import PAPER_STORE
    from backend.schemas.synthesis_schema import (
        CompareRequest,
        GapsRequest,
        PaperComparison,
        ResearchGaps,
    )
    from backend.services.synthesis import compare_papers, identify_research_gaps
except ImportError:  # `uvicorn main:app` from /backend
    from api.v1.papers import PAPER_STORE  # type: ignore[no-redef]
    from schemas.synthesis_schema import (  # type: ignore[no-redef]
        CompareRequest,
        GapsRequest,
        PaperComparison,
        ResearchGaps,
    )
    from services.synthesis import compare_papers, identify_research_gaps  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_or_404(paper_ids: list[str]) -> list[dict]:
    papers = []
    missing = []
    for pid in paper_ids:
        paper = PAPER_STORE.get(pid)
        if paper is None:
            missing.append(pid)
        else:
            papers.append(paper)
    if missing:
        raise HTTPException(
            status_code=404, detail=f"Papers not found: {', '.join(missing)}"
        )
    return papers


@router.post("/compare", response_model=PaperComparison)
async def compare_endpoint(body: CompareRequest) -> PaperComparison:
    """Comparative matrix across selected papers (consensus vs contradictions)."""
    papers = _resolve_or_404(body.paper_ids)
    try:
        return await asyncio.to_thread(compare_papers, papers)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("compare failed")
        raise HTTPException(status_code=500, detail=f"Compare failed: {exc}") from exc


@router.post("/gaps", response_model=ResearchGaps)
async def gaps_endpoint(body: GapsRequest) -> ResearchGaps:
    """Synthesized research gaps across selected papers."""
    papers = _resolve_or_404(body.paper_ids)
    try:
        return await asyncio.to_thread(identify_research_gaps, papers)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("gaps failed")
        raise HTTPException(status_code=500, detail=f"Gap analysis failed: {exc}") from exc
