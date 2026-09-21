"""Comparison & research-gap schemas (Backend Modules C & D, Step 5)."""

from __future__ import annotations

from pydantic import BaseModel, Field

try:  # repo-root vs /backend cwd dual import
    from backend.schemas.paper_schema import SourceTag
except ImportError:  # pragma: no cover
    from schemas.paper_schema import SourceTag  # type: ignore[no-redef]


class PaperRef(BaseModel):
    paper_id: str
    title: str = ""
    year: int | None = None


class ComparisonCell(BaseModel):
    paper_id: str
    value: str = Field(description="Cell value, '—' when not applicable")
    sources: list[SourceTag] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    dimension: str = Field(description="e.g. 'architecture', 'hyperparameter:learning_rate'")
    kind: str = Field(description="metadata|methodology|dataset|benchmark")
    cells: list[ComparisonCell] = Field(default_factory=list)
    verdict: str = Field(
        default="n/a",
        description="consensus|contradiction|divergent|single-source|n/a",
    )
    note: str = Field(default="")


class Finding(BaseModel):
    statement: str
    paper_ids: list[str] = Field(default_factory=list)
    sources: list[SourceTag] = Field(default_factory=list)


class PaperComparison(BaseModel):
    paper_ids: list[str]
    papers: list[PaperRef] = Field(default_factory=list)
    rows: list[ComparisonRow] = Field(default_factory=list)
    consensus_points: list[Finding] = Field(default_factory=list)
    contradictions: list[Finding] = Field(default_factory=list)
    generated_by: str = "heuristic"


class GapItem(BaseModel):
    statement: str
    category: str = Field(
        description="failure_modes|dataset_shortfalls|methodological_blinds"
    )
    paper_ids: list[str] = Field(default_factory=list)
    sources: list[SourceTag] = Field(default_factory=list)


class ResearchGaps(BaseModel):
    paper_ids: list[str]
    failure_modes: list[GapItem] = Field(
        default_factory=list,
        description="1. Unresolved benchmark failure modes",
    )
    dataset_shortfalls: list[GapItem] = Field(
        default_factory=list, description="2. Dataset & evaluation shortfalls"
    )
    methodological_blinds: list[GapItem] = Field(
        default_factory=list,
        description="3. Missing ablations & methodological blinds",
    )
    cross_cutting: list[GapItem] = Field(
        default_factory=list,
        description="Themes recurring across >=2 papers",
    )
    generated_by: str = "heuristic"


class CompareRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=1, max_length=20)


class GapsRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=1, max_length=20)
