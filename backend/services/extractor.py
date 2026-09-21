"""Structured extraction & summarization engine (Backend Module B, Step 4).

Primary path: OpenAI Structured Outputs (``chat.completions.parse`` with the
``PaperExtraction`` Pydantic model as JSON-schema-forced ``response_format``).
Fallback path: deterministic heuristic extractor (regex + section priors)
used when no ``OPENAI_API_KEY`` is set, the ``openai`` package is missing,
or the API call fails. Both paths guarantee:

- exactly-3-bullet executive summary, sectional summaries, 1-sentence TL;DR
- methodology (architecture / hyperparameters / loss / datasets / hardware)
- quantitative benchmark results (task / metric / baseline / paper / delta)
- every claim carries inline source tags (``page_number``, ``paragraph_id``)
  repaired against the parsed paper (``repair_sources``).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # `uvicorn backend.main:app` from repo root
    from backend.schemas.paper_schema import (
        BenchmarkResult,
        DatasetInfo,
        HierarchicalSummary,
        Hyperparameter,
        LossFunction,
        Methodology,
        PaperExtraction,
        QuantitativeResults,
        SectionalEntry,
        SourceTag,
        SummaryBullet,
    )
except ImportError:  # `uvicorn main:app` from /backend
    from schemas.paper_schema import (  # type: ignore[no-redef]
        BenchmarkResult,
        DatasetInfo,
        HierarchicalSummary,
        Hyperparameter,
        LossFunction,
        Methodology,
        PaperExtraction,
        QuantitativeResults,
        SectionalEntry,
        SourceTag,
        SummaryBullet,
    )

EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gpt-4o-mini")
CONTEXT_BUDGET_CHARS = 15000

# Priority order when packing sections into the LLM context window.
_CONTEXT_PRIORITY = [
    "Abstract",
    "Results",
    "Methodology",
    "Introduction",
    "Discussion",
    "Limitations",
    "Future Work",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\(])")

_METRIC_LEXICON = [
    "BLEU",
    "ROUGE",
    "METEOR",
    "F1",
    "accuracy",
    "precision",
    "recall",
    "AUC",
    "MAE",
    "MSE",
    "RMSE",
    "perplexity",
    "PPL",
    "WER",
    "CER",
    "mAP",
    "IoU",
    "reward",
]
_LOWER_BETTER = {"mae", "mse", "rmse", "perplexity", "ppl", "wer", "cer", "loss"}

_KNOWN_DATASETS = [
    "WMT14",
    "WMT",
    "QM9",
    "ImageNet",
    "GLUE",
    "SQuAD",
    "CIFAR-10",
    "CIFAR",
    "COCO",
    "LibriSpeech",
    "WMT-14",
]

_ARCH_KEYWORDS = [
    "transformer",
    "attention",
    "cnn",
    "convolutional",
    "gnn",
    "graph",
    "resnet",
    "bert",
    "gpt",
    "lstm",
    "gru",
    "diffusion",
    "vae",
    "gan",
    "mlp",
    "encoder",
    "decoder",
    "layer",
    "network",
    "architecture",
    "model",
]

_HPARAM_PATTERNS = [
    ("learning_rate", re.compile(r"(?:learning rate|lr)\s*(?:of|=|:)?\s*([0-9.eE\-+]+)", re.I)),
    ("layers", re.compile(r"(\d+)\s*[- ]layer", re.I)),
    ("optimizer", re.compile(r"\b(AdamW?|SGD|RMSprop|Adagrad|Lion)\b")),
    ("batch_size", re.compile(r"batch size\s*(?:of|=|:)?\s*(\d+)", re.I)),
    ("epochs", re.compile(r"(\d+)\s*epochs?", re.I)),
    ("dropout", re.compile(r"dropout\s*(?:of|=|:|rate)?\s*([0-9.]+)", re.I)),
    ("hidden_size", re.compile(r"hidden (?:size|dim(?:ension)?)\s*(?:of|=|:)?\s*(\d+)", re.I)),
    ("heads", re.compile(r"(\d+)\s*(?:attention )?heads?", re.I)),
]

_HARDWARE_RE = re.compile(
    r"(\d+\s*x\s*(?:NVIDIA\s*)?(?:V100|A100|H100|A6000|RTX\s*\d+|GTX\s*\d+|GPU|TPU)s?"
    r"|(?:V100|A100|H100|TPU v?\d+)(?:\s*x\s*\d+)?"
    r"|\d+\s*GPU-hours?|\d+\s*(?:GPU|TPU)\s*days?)",
    re.I,
)

_LATEX_SPAN_RE = re.compile(r"\$\$.+?\$\$|\$.+?\$|\\\[.+?\\\]|\\\(.+?\\\)", re.DOTALL)

_FROM_TO_RE = re.compile(
    r"from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)", re.I
)
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Context packing
# ---------------------------------------------------------------------------


def _as_dict(paper: Any) -> dict[str, Any]:
    return paper.model_dump() if hasattr(paper, "model_dump") else dict(paper)


def _paras_by_section(paper_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for sec in paper_data.get("sections", []) or []:
        out.setdefault(sec.get("name", "?"), []).extend(sec.get("paragraphs", []) or [])
    return out


def build_context(paper: Any, budget: int = CONTEXT_BUDGET_CHARS) -> str:
    """Serialize tagged sections, highest-signal first, within *budget*."""
    data = _as_dict(paper)
    by_section = _paras_by_section(data)
    names = list(by_section)
    ordered = [n for n in _CONTEXT_PRIORITY if n in by_section]
    ordered += [n for n in names if n not in ordered]
    chunks: list[str] = []
    used = 0
    meta = data.get("metadata", {}) or {}
    head = (
        f"TITLE: {meta.get('title', '')}\nAUTHORS: {', '.join(meta.get('authors', []) or [])}\n"
        f"YEAR: {meta.get('year', '')}\n\n"
    )
    chunks.append(head)
    used += len(head)
    for name in ordered:
        lines = [f"## {name}"]
        for p in by_section[name]:
            lines.append(
                f"[{p.get('paragraph_id', '?')}|p.{p.get('page_number', '?')}] {p.get('text', '')}"
            )
        block = "\n".join(lines) + "\n\n"
        if used + len(block) > budget and chunks:
            # Still include a truncated head of high-priority sections.
            room = max(budget - used, 0)
            if room > 200 and name in _CONTEXT_PRIORITY[:3]:
                chunks.append(block[:room] + "\n[truncated]\n\n")
                used = budget
            break
        chunks.append(block)
        used += len(block)
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Heuristic fallback extractor (offline, deterministic, fully sourced)
# ---------------------------------------------------------------------------


def _tag(para: dict[str, Any], section: str) -> SourceTag:
    return SourceTag(
        page_number=int(para.get("page_number", 1) or 1),
        paragraph_id=str(para.get("paragraph_id", "p1")),
        section=section,
    )


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]
    return parts or ([(text or "").strip()] if (text or "").strip() else [])


def heuristic_extract(paper: Any) -> PaperExtraction:
    data = _as_dict(paper)
    paper_id = str(data.get("paper_id", "unknown"))
    by_section = _paras_by_section(data)

    def lead_sentences(name: str, n: int = 2) -> tuple[str, list[SourceTag]]:
        paras = by_section.get(name, [])
        sents: list[str] = []
        tags: list[SourceTag] = []
        for p in paras:
            if len(sents) >= max(n, 1):
                break
            for s in _sentences(p.get("text", "")):
                if len(sents) >= max(n, 1):
                    break
                sents.append(s)
            tags.append(_tag(p, name))
            if len(" ".join(sents)) > 600:
                break
        return " ".join(sents), tags[:3]

    # -- executive bullets: abstract / results / limitations+future ----------
    b1, t1 = lead_sentences("Abstract", 1)
    b2, t2 = lead_sentences("Results", 1)
    lim, tlim = lead_sentences("Limitations", 1)
    fut, tfut = lead_sentences("Future Work", 1)
    b3 = " ".join(s for s in (lim, fut) if s).strip()
    t3 = (tlim + tfut)[:3]
    if not b1:
        b1, t1 = lead_sentences("Introduction", 1)
    if not b2:
        b2, t2 = lead_sentences("Discussion", 1)
    if not b3:
        b3, t3 = lead_sentences("Discussion", 1)
    bullets = [
        SummaryBullet(text=b1 or "No abstract extracted.", sources=t1),
        SummaryBullet(text=b2 or "No results extracted.", sources=t2),
        SummaryBullet(text=b3 or "No limitations extracted.", sources=t3),
    ]

    # -- sectional summary ----------------------------------------------------
    sectional: list[SectionalEntry] = []
    for name in _CONTEXT_PRIORITY:
        if not by_section.get(name):
            continue
        summ, tags = lead_sentences(name, 2)
        if summ:
            sectional.append(SectionalEntry(section=name, summary=summ, sources=tags))

    # -- TL;DR -----------------------------------------------------------------
    tldr, tldr_tags = lead_sentences("Abstract", 1)
    if not tldr:
        meta = data.get("metadata", {}) or {}
        tldr = str(meta.get("title", "Untitled paper.")) + "."
    tldr = _sentences(tldr)[0] if _sentences(tldr) else tldr

    summary = HierarchicalSummary(
        executive_summary=bullets,
        sectional_summary=sectional,
        technical_tldr=tldr,
        technical_tldr_sources=tldr_tags,
    )

    # -- methodology ------------------------------------------------------------
    method_paras = by_section.get("Methodology", []) + by_section.get("Abstract", [])
    arch_text, arch_tags = "", []
    for p in method_paras:
        txt = p.get("text", "")
        low = txt.lower()
        if any(k in low for k in _ARCH_KEYWORDS):
            arch_text = _sentences(txt)[0]
            # section attribution: find owner
            owner = next(
                (n for n, ps in by_section.items() if p in ps), "Methodology"
            )
            arch_tags = [_tag(p, owner)]
            break
    if not arch_text and method_paras:
        arch_text = _sentences(method_paras[0].get("text", ""))[0]
        owner = next(
            (n for n, ps in by_section.items() if method_paras[0] in ps),
            "Methodology",
        )
        arch_tags = [_tag(method_paras[0], owner)]

    hyperparams: list[Hyperparameter] = []
    seen_hp: set[str] = set()
    for sec_name, plist in by_section.items():
        for p in plist:
            for hp_name, rx in _HPARAM_PATTERNS:
                m = rx.search(p.get("text", ""))
                if m and hp_name not in seen_hp:
                    seen_hp.add(hp_name)
                    val = (m.group(1) if m.lastindex else m.group(0)).strip().rstrip(".,;")
                    hyperparams.append(
                        Hyperparameter(name=hp_name, value=val, sources=[_tag(p, sec_name)])
                    )

    losses: list[LossFunction] = []
    for sec_name in ("Methodology", "Abstract", "Introduction"):
        for p in by_section.get(sec_name, []):
            for m in _LATEX_SPAN_RE.finditer(p.get("text", "")):
                expr = m.group(0).strip()
                if len(expr) < 4 or any(
                    l.expression_latex == expr for l in losses
                ):
                    continue
                sent = next(
                    (s for s in _sentences(p.get("text", "")) if expr[:20] in s),
                    _sentences(p.get("text", ""))[0] if _sentences(p.get("text", "")) else "",
                )
                losses.append(
                    LossFunction(
                        expression_latex=expr,
                        description=sent[:300],
                        sources=[_tag(p, sec_name)],
                    )
                )
            if losses:
                break

    datasets: list[DatasetInfo] = []
    seen_ds: set[str] = set()
    for sec_name, plist in by_section.items():
        for p in plist:
            txt = p.get("text", "")
            for known in _KNOWN_DATASETS:
                if re.search(rf"\b{re.escape(known)}\b", txt) and known.lower() not in seen_ds:
                    seen_ds.add(known.lower())
                    sent = _sentences(txt)[0] if _sentences(txt) else txt[:200]
                    datasets.append(
                        DatasetInfo(name=known, details=sent[:300], sources=[_tag(p, sec_name)])
                    )
            for m in re.finditer(
                r"\b([A-Z][\w\-]*\d+)\s+(dataset|corpus|benchmark)\b", txt
            ):
                if m.group(1).lower() not in seen_ds:
                    seen_ds.add(m.group(1).lower())
                    datasets.append(
                        DatasetInfo(
                            name=m.group(1),
                            details=m.group(0) + ": " + (_sentences(txt)[0][:200] if _sentences(txt) else ""),
                            sources=[_tag(p, sec_name)],
                        )
                    )

    hw_text, hw_tags = "", []
    for sec_name, plist in by_section.items():
        for p in plist:
            m = _HARDWARE_RE.search(p.get("text", ""))
            if m:
                hw_text = m.group(0)
                hw_tags = [_tag(p, sec_name)]
                break
        if hw_text:
            break

    methodology = Methodology(
        architecture=arch_text,
        architecture_sources=arch_tags,
        hyperparameters=hyperparams,
        loss_functions=losses,
        datasets=datasets,
        hardware_requirements=hw_text,
        hardware_sources=hw_tags,
    )

    # -- quantitative results ----------------------------------------------------
    bench: list[BenchmarkResult] = []
    metric_rx = re.compile(
        r"\b(" + "|".join(re.escape(m) for m in _METRIC_LEXICON) + r")\b", re.I
    )
    for sec_name in ("Results", "Abstract", "Discussion", "Introduction"):
        for p in plist_of(by_section, sec_name):
            txt = p.get("text", "")
            mm = metric_rx.search(txt)
            if not mm:
                continue
            metric = mm.group(1).upper() if len(mm.group(1)) <= 4 else mm.group(1)
            nums = [float(x) for x in _FLOAT_RE.findall(txt)]
            if not nums:
                continue
            ft = _FROM_TO_RE.search(txt)
            if ft:
                base, score = float(ft.group(1)), float(ft.group(2))
            else:
                # number nearest the metric token wins
                scored = sorted(nums, key=lambda v: abs(txt.find(str(mm.group(1))) - txt.find(_num_token(txt, v))))
                score = scored[0]
                base = 0.0
            on_task = re.search(r"\bon\s+([A-Z][\w\-]*(?:\s+[a-z\d]+)?)", txt)
            task = on_task.group(1) if on_task else f"{sec_name} evaluation"
            bench.append(
                BenchmarkResult(
                    task=task,
                    metric_name=metric,
                    baseline_score=base,
                    paper_score=score,
                    sota_delta=round(score - base, 6),
                    higher_is_better=metric.lower() not in _LOWER_BETTER,
                    score_context=_sentences(txt)[0][:200] if _sentences(txt) else "",
                    sources=[_tag(p, sec_name)],
                )
            )
            if len(bench) >= 10:
                break

    extraction = PaperExtraction(
        paper_id=paper_id,
        summary=summary,
        methodology=methodology,
        quantitative_results=QuantitativeResults(results=bench),
        extraction_source="heuristic-fallback",
    )
    return repair_sources(extraction, data)


def plist_of(by_section: dict[str, list[dict[str, Any]]], name: str):
    return by_section.get(name, [])


def _num_token(txt: str, value: float) -> str:
    iv = int(value)
    return str(iv) if float(iv) == value and str(iv) in txt else str(value)


# ---------------------------------------------------------------------------
# Source repair + validation
# ---------------------------------------------------------------------------


def repair_sources(extraction: PaperExtraction, paper: Any) -> PaperExtraction:
    """Drop hallucinated tags; backfill empty source lists with anchors."""
    data = _as_dict(paper)
    valid: set[tuple[str, int]] = set()
    section_first: dict[str, SourceTag] = {}
    overall_first: Optional[SourceTag] = None
    for sec in data.get("sections", []) or []:
        for p in sec.get("paragraphs", []) or []:
            pid = str(p.get("paragraph_id", ""))
            pg = int(p.get("page_number", 1) or 1)
            if pid:
                valid.add((pid, pg))
            tag = SourceTag(page_number=pg, paragraph_id=pid, section=sec.get("name", ""))
            section_first.setdefault(sec.get("name", ""), tag)
            if overall_first is None and pid:
                overall_first = tag

    def fix(tags: list[SourceTag], section: str = "") -> list[SourceTag]:
        kept = [t for t in tags if (t.paragraph_id, t.page_number) in valid]
        if kept:
            return kept
        anchor = section_first.get(section) or overall_first
        return [anchor] if anchor else []

    dump = extraction.model_dump()
    for b in dump["summary"]["executive_summary"]:
        b["sources"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in fix(_to_tags(b.get("sources", [])))]
    for e in dump["summary"]["sectional_summary"]:
        e["sources"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in fix(_to_tags(e.get("sources", [])), e.get("section", ""))]
    dump["summary"]["technical_tldr_sources"] = [
        t.model_dump() if hasattr(t, "model_dump") else t
        for t in fix(_to_tags(dump["summary"].get("technical_tldr_sources", [])))
    ]
    m = dump["methodology"]
    m["architecture_sources"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in fix(_to_tags(m.get("architecture_sources", [])), "Methodology")]
    for h in m.get("hyperparameters", []):
        h["sources"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in fix(_to_tags(h.get("sources", [])))]
    for lf in m.get("loss_functions", []):
        lf["sources"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in fix(_to_tags(lf.get("sources", [])))]
    for d in m.get("datasets", []):
        d["sources"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in fix(_to_tags(d.get("sources", [])))]
    m["hardware_sources"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in fix(_to_tags(m.get("hardware_sources", [])))]
    for r in dump["quantitative_results"]["results"]:
        r["sources"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in fix(_to_tags(r.get("sources", [])), "Results")]
    return PaperExtraction.model_validate(dump)


def _to_tags(raw: Any) -> list[SourceTag]:
    out: list[SourceTag] = []
    for item in raw or []:
        if isinstance(item, SourceTag):
            out.append(item)
        elif isinstance(item, dict):
            try:
                out.append(SourceTag.model_validate(item))
            except Exception:
                continue
    return out


def validate_citations(extraction: PaperExtraction, paper: Any) -> list[str]:
    """Return human-readable integrity issues (empty list = clean)."""
    data = _as_dict(paper)
    valid: set[tuple[str, int]] = set()
    for sec in data.get("sections", []) or []:
        for p in sec.get("paragraphs", []) or []:
            valid.add((str(p.get("paragraph_id", "")), int(p.get("page_number", 1) or 1)))
    issues: list[str] = []
    dump = extraction.model_dump()

    def check(tags: Any, where: str) -> None:
        tags = tags or []
        if not tags:
            issues.append(f"{where}: missing source tags")
            return
        for t in tags:
            if (str(t.get("paragraph_id", "")), int(t.get("page_number", 1) or 1)) not in valid:
                issues.append(f"{where}: unknown tag {t.get('paragraph_id')}/p.{t.get('page_number')}")

    for i, b in enumerate(dump["summary"]["executive_summary"]):
        check(b.get("sources"), f"executive[{i}]")
    if len(dump["summary"]["executive_summary"]) != 3:
        issues.append("executive_summary must have exactly 3 bullets")
    for e in dump["summary"]["sectional_summary"]:
        check(e.get("sources"), f"sectional[{e.get('section')}]")
    if not (dump["summary"].get("technical_tldr") or "").strip():
        issues.append("technical_tldr empty")
    for r in dump["quantitative_results"]["results"]:
        check(r.get("sources"), f"result[{r.get('metric_name')}]")
    return issues


# ---------------------------------------------------------------------------
# OpenAI Structured Outputs path + public entrypoint
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are ScholarSense, a precise research-paper analyst. Extract a \
hierarchical summary, methodology, and quantitative results from the paper text. \
Rules: executive_summary has EXACTLY 3 bullets; technical_tldr is ONE sentence; \
every claim MUST cite sources using ONLY the [paragraph_id|page] markers shown \
(do not invent ids); math MUST use standard LaTeX ($...$); scores MUST be numeric \
floats; sota_delta = paper_score - baseline_score."""


def openai_extract(
    paper: Any, model: str = EXTRACTION_MODEL, timeout: float = 120.0
) -> PaperExtraction:
    """Extract via OpenAI Structured Outputs (JSON Schema forcing)."""
    data = _as_dict(paper)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openai package is not installed (pip install openai)"
        ) from exc
    client = OpenAI(timeout=timeout)
    context = build_context(paper)
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"PAPER_ID: {data.get('paper_id', 'unknown')}\n\n{context}\n\n"
                    "Return the PaperExtraction JSON for this paper."
                ),
            },
        ],
        response_format=PaperExtraction,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(f"OpenAI returned no parsed content (refusal: {completion.choices[0].message.refusal})")
    parsed.paper_id = str(data.get("paper_id", parsed.paper_id or "unknown"))
    parsed.extraction_source = "openai-structured"
    return repair_sources(parsed, data)


def extract_paper(
    paper: Any,
    prefer: str = "auto",
    model: str = EXTRACTION_MODEL,
) -> PaperExtraction:
    """Extract structured data from a parsed paper.

    ``prefer``: ``"auto"`` (OpenAI when configured, else heuristic),
    ``"openai"`` (raise on failure), ``"heuristic"`` (deterministic).
    """
    if prefer == "heuristic":
        return heuristic_extract(paper)
    if prefer == "openai":
        return openai_extract(paper, model=model)
    # auto
    if os.getenv("OPENAI_API_KEY"):
        try:
            return openai_extract(paper, model=model)
        except Exception as exc:
            logger.warning("openai extraction failed (%s); heuristic fallback", exc)
    return heuristic_extract(paper)
