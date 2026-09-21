"""Multi-paper comparison & research-gap synthesizer (Modules C & D, Step 5).

Deterministic and fully sourced: every matrix cell and gap item carries the
``SourceTag``s of the extraction claims (or section sentences) behind it.
Papers without a stored extraction are extracted on the fly with the
offline heuristic extractor, so compare/gaps work straight after upload.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

try:  # repo-root vs /backend cwd dual import
    from backend.schemas.paper_schema import PaperExtraction, SourceTag
    from backend.schemas.synthesis_schema import (
        ComparisonCell,
        ComparisonRow,
        Finding,
        GapItem,
        PaperComparison,
        PaperRef,
        ResearchGaps,
    )
    from backend.services.extractor import _sentences, heuristic_extract
except ImportError:  # pragma: no cover
    from schemas.paper_schema import PaperExtraction, SourceTag  # type: ignore[no-redef]
    from schemas.synthesis_schema import (  # type: ignore[no-redef]
        ComparisonCell,
        ComparisonRow,
        Finding,
        GapItem,
        PaperComparison,
        PaperRef,
        ResearchGaps,
    )
    from services.extractor import _sentences, heuristic_extract  # type: ignore[no-redef]

_MISSING = "—"


def _extraction_for(paper: dict[str, Any]) -> PaperExtraction:
    stored = paper.get("extraction")
    if stored:
        try:
            return PaperExtraction.model_validate(stored)
        except Exception as exc:
            logger.warning("stored extraction invalid, re-extracting: %s", exc)
    return heuristic_extract(paper)


def _title_of(paper: dict[str, Any]) -> str:
    return str((paper.get("metadata", {}) or {}).get("title", "") or "Untitled")


# ---------------------------------------------------------------------------
# Module C: compare_papers
# ---------------------------------------------------------------------------


def compare_papers(papers: list[dict[str, Any]]) -> PaperComparison:
    """Align methodology, datasets, and benchmarks across *papers*."""
    if not papers:
        raise ValueError("compare_papers needs at least one paper.")
    exts = {str(p.get("paper_id")): _extraction_for(p) for p in papers}
    pids = [str(p.get("paper_id")) for p in papers]
    refs = [
        PaperRef(
            paper_id=pid,
            title=_title_of(p),
            year=(p.get("metadata", {}) or {}).get("year"),
        )
        for p, pid in zip(papers, pids)
    ]

    rows: list[ComparisonRow] = []
    consensus: list[Finding] = []
    contradictions: list[Finding] = []

    def row(dimension: str, kind: str, values: dict[str, tuple[str, list[SourceTag]]]):
        cells = [
            ComparisonCell(
                paper_id=pid,
                value=values.get(pid, (_MISSING, []))[0],
                sources=values.get(pid, (_MISSING, []))[1],
            )
            for pid in pids
        ]
        return ComparisonRow(dimension=dimension, kind=kind, cells=cells)

    # -- TL;DR + architecture -------------------------------------------------
    r = row(
        "tl_dr",
        "metadata",
        {pid: (e.summary.technical_tldr or _MISSING, e.summary.technical_tldr_sources) for pid, e in exts.items()},
    )
    rows.append(r)

    arch_vals = {
        pid: (
            (e.methodology.architecture[:300] or _MISSING),
            e.methodology.architecture_sources,
        )
        for pid, e in exts.items()
    }
    r = row("architecture", "methodology", arch_vals)
    arch_set = {v for v, _ in arch_vals.values() if v != _MISSING}
    if len(pids) > 1:
        if len(arch_set) <= 1 and arch_set:
            r.verdict, r.note = "consensus", "Papers describe a matching architecture."
        elif len(arch_set) > 1:
            r.verdict, r.note = "divergent", "Papers use different architectures."
        else:
            r.verdict = "n/a"
    else:
        r.verdict = "single-source"
    rows.append(r)

    # -- hyperparameters (union over names) ------------------------------------
    hp_names: list[str] = []
    for e in exts.values():
        for h in e.methodology.hyperparameters:
            if h.name not in hp_names:
                hp_names.append(h.name)
    for hp_name in hp_names:
        vals: dict[str, tuple[str, list[SourceTag]]] = {}
        for pid, e in exts.items():
            hit = next((h for h in e.methodology.hyperparameters if h.name == hp_name), None)
            if hit:
                vals[pid] = (hit.value or _MISSING, hit.sources)
        rr = row(f"hyperparameter:{hp_name}", "methodology", vals)
        distinct = {v for v, _ in vals.values() if v != _MISSING}
        if len(vals) <= 1:
            rr.verdict = "single-source"
        elif len(distinct) == 1:
            rr.verdict, rr.note = "consensus", f"All papers set {hp_name}={distinct.pop()}."
        else:
            rr.verdict, rr.note = "divergent", f"{hp_name} differs across papers."
        rows.append(rr)

    # -- datasets (union over names) --------------------------------------------
    ds_names: list[str] = []
    for e in exts.values():
        for d in e.methodology.datasets:
            if d.name not in ds_names:
                ds_names.append(d.name)
    for ds in ds_names:
        vals = {}
        for pid, e in exts.items():
            hit = next((d for d in e.methodology.datasets if d.name == ds), None)
            if hit:
                vals[pid] = ((hit.details[:200] or "evaluated"), hit.sources)
        rr = row(f"dataset:{ds}", "dataset", vals)
        if len(vals) >= 2:
            rr.verdict, rr.note = "consensus", f"Shared evaluation on {ds}."
            consensus.append(
                Finding(
                    statement=f"Shared evaluation dataset: {ds}.",
                    paper_ids=sorted(vals),
                    sources=[t for _, tags in vals.values() for t in tags][:6],
                )
            )
        else:
            rr.verdict = "single-source"
        rows.append(rr)

    # -- benchmarks grouped by (task, metric) ------------------------------------
    def _norm(s: str) -> str:
        return re.sub(r"\W+", "", (s or "").lower())

    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    labels: dict[tuple[str, str], tuple[str, str]] = {}
    for pid, e in exts.items():
        for b in e.quantitative_results.results:
            key = (_norm(b.task), _norm(b.metric_name))
            labels[key] = (b.task, b.metric_name)
            groups[key][pid] = b
    for key, per_paper in groups.items():
        task, metric = labels[key]
        vals = {}
        for pid, b in per_paper.items():
            vals[pid] = (
                f"{b.baseline_score:g} → {b.paper_score:g} (Δ{b.sota_delta:+.4g} {b.metric_name})",
                b.sources,
            )
        rr = row(f"benchmark:{task} [{metric}]", "benchmark", vals)
        if len(per_paper) >= 2:
            deltas = [b.sota_delta for b in per_paper.values()]
            pos = [d for d in deltas if d > 0]
            neg = [d for d in deltas if d < 0]
            if pos and neg:
                rr.verdict = "contradiction"
                rr.note = (
                    f"Conflicting {metric} results on {task}: "
                    + ", ".join(
                        f"{pid} Δ{d:+.4g}" for pid, d in zip(per_paper, deltas)
                    )
                    + "."
                )
                contradictions.append(
                    Finding(
                        statement=rr.note,
                        paper_ids=sorted(per_paper),
                        sources=[t for b in per_paper.values() for t in b.sources][:8],
                    )
                )
            else:
                rr.verdict = "consensus"
                rr.note = f"Consistent direction on {task} [{metric}] across papers."
                consensus.append(
                    Finding(
                        statement=rr.note,
                        paper_ids=sorted(per_paper),
                        sources=[t for b in per_paper.values() for t in b.sources][:8],
                    )
                )
        else:
            rr.verdict = "single-source"
        rows.append(rr)

    return PaperComparison(
        paper_ids=pids, papers=refs, rows=rows,
        consensus_points=consensus, contradictions=contradictions,
    )


# ---------------------------------------------------------------------------
# Module D: identify_research_gaps
# ---------------------------------------------------------------------------

_FAILURE_RES = [
    r"fail", r"error", r"poor", r"weak", r"degrad", r"\bdrop", r"struggl",
    r"untest", r"unexplor", r"\bonly\b.{0,20}(evaluat|test|english|small)",
    r"limit", r"brittle", r"instab", r"long\b.{0,15}(document|context|sequence)",
    r"hallucinat", r"bias",
]
_DATASET_RES = [
    r"dataset", r"\bdata\b", r"benchmark", r"evaluat", r"language",
    r"domain", r"generali", r"distribution", r"out-of", r"corpus",
    r"multilingual", r"low-resource", r"test set", r"real-world",
    r"modality", r"modalities",
]
_ABLATION_RES = [
    r"ablat", r"hyperparameter", r"tuning", r"comparison", r"baseline",
    r"assum", r"blind", r"sensitiv", r"architectur.{0,10}choice",
    r"\bscale\b", r"comput", r"\bgpu", r"resource", r"carbon",
    r"theor", r"interpret",
]

_CATEGORY_ORDER = ("failure_modes", "dataset_shortfalls", "methodological_blinds")
_CATEGORY_RES = {
    "failure_modes": [re.compile(p, re.I) for p in _FAILURE_RES],
    "dataset_shortfalls": [re.compile(p, re.I) for p in _DATASET_RES],
    "methodological_blinds": [re.compile(p, re.I) for p in _ABLATION_RES],
}

_THEME_TOKENS = [
    "multilingual", "long", "ablation", "generali", "scale", "robust",
    "real-world", "low-resource", "interpret", "efficiency", "hallucinat",
]


def _harvest_gap_sentences(
    paper: dict[str, Any],
) -> list[tuple[str, str, SourceTag]]:
    """(sentence, section, tag) from Limitations + Future Work (+ Discussion)."""
    out: list[tuple[str, str, SourceTag]] = []
    for sec in paper.get("sections", []) or []:
        name = sec.get("name", "")
        if name not in ("Limitations", "Future Work", "Discussion"):
            continue
        for p in sec.get("paragraphs", []) or []:
            tag = SourceTag(
                page_number=int(p.get("page_number", 1) or 1),
                paragraph_id=str(p.get("paragraph_id", "")),
                section=name,
            )
            for sent in _sentences(p.get("text", "")):
                if len(sent) >= 20:
                    out.append((sent.strip(), name, tag))
    return out


def _classify(sentence: str) -> tuple[str, int]:
    scores = {
        cat: sum(1 for rx in rxs if rx.search(sentence))
        for cat, rxs in _CATEGORY_RES.items()
    }
    best = max(_CATEGORY_ORDER, key=lambda c: (scores[c], -_CATEGORY_ORDER.index(c)))
    # tie-break prefers failure_modes, then dataset_shortfalls
    top = max(scores.values())
    if top == 0:
        return "methodological_blinds", 0
    for cat in _CATEGORY_ORDER:
        if scores[cat] == top:
            return cat, top
    return best, top


def identify_research_gaps(papers: list[dict[str, Any]]) -> ResearchGaps:
    """Aggregate Limitations + Future Work into the three gap buckets."""
    if not papers:
        raise ValueError("identify_research_gaps needs at least one paper.")
    pids = [str(p.get("paper_id")) for p in papers]
    buckets: dict[str, list[GapItem]] = {c: [] for c in _CATEGORY_ORDER}
    seen: set[str] = set()

    for paper, pid in zip(papers, pids):
        for sent, section, tag in _harvest_gap_sentences(paper):
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            cat, _ = _classify(sent)
            buckets[cat].append(
                GapItem(
                    statement=sent,
                    category=cat,
                    paper_ids=[pid],
                    sources=[tag],
                )
            )

    for cat in buckets:
        buckets[cat] = buckets[cat][:12]

    # -- cross-cutting themes (shared stems across >=2 papers) ------------------
    by_theme: dict[str, GapItem] = {}
    for cat, items in buckets.items():
        for item in items:
            low = item.statement.lower()
            for theme in _THEME_TOKENS:
                if theme in low:
                    agg = by_theme.setdefault(
                        theme,
                        GapItem(
                            statement=f"Recurring theme: '{theme}'.",
                            category=cat,
                            paper_ids=[],
                            sources=[],
                        ),
                    )
                    for pid in item.paper_ids:
                        if pid not in agg.paper_ids:
                            agg.paper_ids.append(pid)
                    agg.sources.extend(item.sources)
    cross = [
        GapItem(
            statement=f"{a.statement} {len(a.paper_ids)} papers: "
            + "; ".join(
                next(
                    (
                        i.statement[:120]
                        for items in buckets.values()
                        for i in items
                        if pid in i.paper_ids and theme in i.statement.lower()
                    ),
                    "",
                )
                for pid in a.paper_ids
            ),
            category=a.category,
            paper_ids=sorted(a.paper_ids),
            sources=a.sources[:8],
        )
        for theme, a in by_theme.items()
        if len(a.paper_ids) >= 2
    ][:10]

    return ResearchGaps(
        paper_ids=pids,
        failure_modes=buckets["failure_modes"],
        dataset_shortfalls=buckets["dataset_shortfalls"],
        methodological_blinds=buckets["methodological_blinds"],
        cross_cutting=cross,
    )
