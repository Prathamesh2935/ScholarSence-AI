"""Structured extraction schemas (Backend Module B, Step 4).

Every extracted claim carries inline source tags (``page_number`` +
``paragraph_id``) so the frontend can cite back to the source PDF.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SourceTag(BaseModel):
    """Inline citation pointing at the originating PDF location."""

    page_number: int = Field(ge=1, description="1-based PDF page number")
    paragraph_id: str = Field(description="Paragraph id, e.g. 'p7'")
    section: str = Field(default="", description="Canonical section name")


# ---------------------------------------------------------------------------
# HierarchicalSummary
# ---------------------------------------------------------------------------


class SummaryBullet(BaseModel):
    text: str = Field(description="One crisp executive-summary bullet")
    sources: list[SourceTag] = Field(
        default_factory=list, description="Source tags grounding this bullet"
    )


class SectionalEntry(BaseModel):
    section: str = Field(description="Canonical section name")
    summary: str = Field(description="2-3 sentence summary of the section")
    sources: list[SourceTag] = Field(default_factory=list)


class HierarchicalSummary(BaseModel):
    executive_summary: list[SummaryBullet] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 executive-summary bullet points",
    )
    sectional_summary: list[SectionalEntry] = Field(
        default_factory=list, description="Per-section summaries"
    )
    technical_tldr: str = Field(description="1-sentence technical TL;DR")
    technical_tldr_sources: list[SourceTag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------


class Hyperparameter(BaseModel):
    name: str = Field(description="e.g. 'learning_rate', 'optimizer', 'layers'")
    value: str = Field(description="e.g. '3e-4', 'Adam', '12'")
    sources: list[SourceTag] = Field(default_factory=list)


class LossFunction(BaseModel):
    name: str = Field(default="", description="e.g. 'cross-entropy', 'ELBO'")
    expression_latex: str = Field(
        default="",
        description="Loss in standard LaTeX, e.g. '$L_{loss} = \\lambda \\cdot L_{rec}$'",
    )
    description: str = Field(default="")
    sources: list[SourceTag] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    name: str = Field(description="e.g. 'WMT14', 'QM9', 'ImageNet'")
    details: str = Field(default="", description="Size / split / task info")
    sources: list[SourceTag] = Field(default_factory=list)


class Methodology(BaseModel):
    architecture: str = Field(default="", description="Model/system architecture")
    architecture_sources: list[SourceTag] = Field(default_factory=list)
    hyperparameters: list[Hyperparameter] = Field(default_factory=list)
    loss_functions: list[LossFunction] = Field(default_factory=list)
    datasets: list[DatasetInfo] = Field(default_factory=list)
    hardware_requirements: str = Field(
        default="", description="e.g. '8x NVIDIA V100, 3 days'"
    )
    hardware_sources: list[SourceTag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# QuantitativeResults
# ---------------------------------------------------------------------------


class BenchmarkResult(BaseModel):
    task: str = Field(description="e.g. 'WMT14 EN-DE translation'")
    metric_name: str = Field(description="e.g. 'BLEU'")
    baseline_score: float = Field(default=0.0, description="Baseline/SOTA score")
    baseline_label: str = Field(default="", description="e.g. 'Transformer (2017)'")
    paper_score: float = Field(default=0.0, description="This paper's score")
    sota_delta: float = Field(
        default=0.0, description="paper_score - baseline_score (signed)"
    )
    higher_is_better: bool = Field(default=True)
    score_context: str = Field(
        default="", description="Units / split / qualifier, e.g. 'test BLEU'"
    )
    sources: list[SourceTag] = Field(default_factory=list)


class QuantitativeResults(BaseModel):
    results: list[BenchmarkResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level extraction envelope
# ---------------------------------------------------------------------------


class PaperExtraction(BaseModel):
    paper_id: str
    summary: HierarchicalSummary
    methodology: Methodology = Field(default_factory=Methodology)
    quantitative_results: QuantitativeResults = Field(
        default_factory=QuantitativeResults
    )
    extraction_source: Literal["openai-structured", "heuristic-fallback"] = (
        "heuristic-fallback"
    )

    @field_validator("summary")
    @classmethod
    def _three_bullets(cls, v: HierarchicalSummary) -> HierarchicalSummary:
        if len(v.executive_summary) != 3:
            raise ValueError("executive_summary must have exactly 3 bullets")
        return v
