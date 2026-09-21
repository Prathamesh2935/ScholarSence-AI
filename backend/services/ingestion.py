"""Document ingestion & math parsing pipeline (Backend Module A).

Pipeline:
1. Extract per-page text with PyMuPDF (primary; reliable page numbers).
2. Optionally refine heading hints with Unstructured (best-effort, never fatal).
3. Split into paragraphs with ``page_number`` / ``paragraph_id`` attribution.
4. Segment into canonical sections via heading detection.
5. Preserve math as standard LaTeX (``$...$`` / ``$$...$$`` kept, unicode
   math symbols mapped to LaTeX commands, e.g. ``λ`` -> ``\\lambda``).
6. Extract metadata (title / authors / year) heuristically, then enrich via
   the Semantic Scholar API (fallback, best-effort).
"""

from __future__ import annotations

import io
import logging
import re
import tempfile
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"

# ---------------------------------------------------------------------------
# Pydantic models (parse-stage representation; extraction schemas land in
# backend/schemas/paper_schema.py in Step 4 and build on top of this output).
# ---------------------------------------------------------------------------


class ParsedParagraph(BaseModel):
    paragraph_id: str
    page_number: int
    text: str


class PaperSection(BaseModel):
    name: str
    text: str = ""
    paragraphs: list[ParsedParagraph] = Field(default_factory=list)


class PaperMetadata(BaseModel):
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    enrichment_source: str = "pdf-heuristic"  # or "semantic-scholar"


class ParsedPaper(BaseModel):
    paper_id: str
    filename: str
    metadata: PaperMetadata
    sections: list[PaperSection]
    page_count: int
    paragraph_count: int


# ---------------------------------------------------------------------------
# Section segmentation
# ---------------------------------------------------------------------------

CANONICAL_SECTIONS = [
    "Abstract",
    "Introduction",
    "Methodology",
    "Results",
    "Discussion",
    "Limitations",
    "Future Work",
]

# Canonical section -> heading aliases seen in the wild.
SECTION_ALIASES: dict[str, list[str]] = {
    "Abstract": ["abstract", "executive summary", "summary"],
    "Introduction": [
        "introduction",
        "background",
        "motivation",
        "related work",
        "related works",
        "literature review",
    ],
    "Methodology": [
        "methodology",
        "method",
        "methods",
        "materials and methods",
        "approach",
        "proposed approach",
        "model",
        "model architecture",
        "architecture",
        "system",
        "framework",
        "implementation",
        "experimental setup",
        "experiment setup",
    ],
    "Results": [
        "results",
        "result",
        "experiments",
        "experiment",
        "experimental results",
        "experimental evaluation",
        "evaluation",
        "findings",
        "benchmarks",
        "benchmark results",
        "performance",
        "performance evaluation",
    ],
    "Discussion": [
        "discussion",
        "analysis",
        "conclusion",
        "conclusions",
        "concluding remarks",
        "interpretation",
        "remarks",
    ],
    "Limitations": [
        "limitations",
        "limitation",
        "threats to validity",
        "weaknesses",
        "constraints",
    ],
    "Future Work": [
        "future work",
        "future works",
        "future directions",
        "future research",
        "outlook",
        "next steps",
        "conclusion and future work",
        "conclusions and future work",
    ],
    # Non-canonical bucket: keeps bibliographies from polluting Future Work.
    "References": ["references", "bibliography", "works cited"],
}

_NUMBER_PREFIX = r"(?:\d+(?:\.\d+)*\.?|[ivxlcdm]+\.?|[a-z]\.?)?\s*"
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _aliases in SECTION_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canon
# Longest aliases first so "conclusion and future work" wins over "conclusion".
_SORTED_ALIASES = sorted(_ALIAS_TO_CANONICAL, key=len, reverse=True)
_HEADING_RE = re.compile(
    r"^\s*" + _NUMBER_PREFIX + r"("
    + "|".join(re.escape(a) for a in _SORTED_ALIASES)
    + r")\s*[:\-–—.]?\s*$",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_ARXIV_RE = re.compile(r"arXiv:\s*(\d{4})\.\d{4,5}", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)


def detect_section_heading(line: str) -> Optional[str]:
    """Return the canonical section name if *line* is a section heading."""
    text = line.strip()
    if not text or len(text) > 100:
        return None
    # Must look like a heading: short, no sentence-ending period mid-text,
    # no more than ~8 words.
    if len(text.split()) > 8 or (text.count(".") > 1):
        return None
    match = _HEADING_RE.match(text)
    if match:
        return _ALIAS_TO_CANONICAL[match.group(1).lower()]
    return None


# ---------------------------------------------------------------------------
# Math preservation (LaTeX)
# ---------------------------------------------------------------------------

# Unicode math symbols -> LaTeX commands. Applied to running text so equations
# survive as standard LaTeX, e.g. "L_loss = λ · L_rec" -> "L_loss = \lambda
# \cdot L_rec" (inline/block `$` delimiters already in the text are kept).
UNICODE_MATH_MAP: dict[str, str] = {
    "λ": r"\lambda",
    "β": r"\beta",
    "α": r"\alpha",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "θ": r"\theta",
    "μ": r"\mu",
    "σ": r"\sigma",
    "Σ": r"\Sigma",
    "π": r"\pi",
    "φ": r"\phi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Δ": r"\Delta",
    "∞": r"\infty",
    "×": r"\times",
    "·": r"\cdot",
    "÷": r"\div",
    "±": r"\pm",
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "∈": r"\in",
    "∉": r"\notin",
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "∂": r"\partial",
    "√": r"\sqrt",
    "→": r"\rightarrow",
    "←": r"\leftarrow",
    "⇒": r"\Rightarrow",
}

_LATEX_DELIM_RE = re.compile(
    r"(\$\$.+?\$\$|\$.+?\$|\\\[.+?\\\]|\\\(.+?\\\))", re.DOTALL
)


def _replace_unicode_math(chunk: str) -> str:
    for uni, latex in UNICODE_MATH_MAP.items():
        if uni in chunk:
            # Pad command-like replacements with spaces when glued to alnum.
            chunk = chunk.replace(uni, f" {latex} ")
    return chunk


def preserve_math(text: str) -> str:
    """Normalize math in *text* to standard LaTeX.

    - Existing ``$...$``, ``$$...$$``, ``\\(...\\)``, ``\\[...\\]`` spans are
      kept verbatim (delimiters preserved) while their *contents* get unicode
      -> LaTeX normalization.
    - Running text outside delimiters gets unicode math mapped to LaTeX.
    """
    if not text:
        return text
    parts = _LATEX_DELIM_RE.split(text)
    joined = "".join(_replace_unicode_math(part) for part in parts)
    # Collapse padding introduced around substituted commands (spaces only;
    # paragraph newlines are joined upstream in segment_pages).
    return re.sub(r"[ \t]+", " ", joined).strip()


# ---------------------------------------------------------------------------
# PyMuPDF extraction (primary)
# ---------------------------------------------------------------------------


def extract_pages_with_pymupdf(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Extract ``[{page_number, text}]`` with PyMuPDF. Raises on corrupt PDF."""
    import pymupdf  # PyMuPDF; ImportError surfaces to the caller as 500 detail

    pages: list[dict[str, Any]] = []
    try:
        doc_cm = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        # Corrupt / unparseable uploads are client errors (mapped to 422).
        raise ValueError(f"Could not open PDF: {exc}") from exc
    with doc_cm as doc:
        if len(doc) == 0:
            raise ValueError("PDF has no pages.")
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append({"page_number": i, "text": text})
    return pages


def extract_title_candidate_pymupdf(pdf_bytes: bytes) -> Optional[str]:
    """Heuristic title: largest font span in the top third of page 1."""
    try:
        import pymupdf

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            page = doc[0]
            d = page.get_text("dict")
            height = page.rect.height
            best: tuple[float, str] = (0.0, "")
            for block in d.get("blocks", []):
                for line in block.get("lines", []):
                    top = line.get("bbox", [0, 0])[1]
                    if top > height / 2.5:  # top region only
                        continue
                    for span in line.get("spans", []):
                        txt = (span.get("text") or "").strip()
                        size = float(span.get("size", 0))
                        if len(txt) >= 10 and size > best[0]:
                            best = (size, txt)
            return best[1] or None
    except Exception:  # never let title heuristics break ingestion
        logger.debug("pymupdf title heuristic failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Unstructured (best-effort heading hints)
# ---------------------------------------------------------------------------


def try_unstructured_titles(pdf_bytes: bytes) -> list[str]:
    """Return Title-element texts via Unstructured, or [] if unavailable.

    Unstructured is optional: missing dependency, missing system libs, or
    parse errors all degrade gracefully to PyMuPDF-only extraction.
    """
    try:
        from unstructured.partition.pdf import partition_pdf  # type: ignore
    except Exception as exc:
        logger.debug("unstructured unavailable, using PyMuPDF only: %s", exc)
        return []
    tmp = Path(tempfile.mkstemp(suffix=".pdf")[1])
    try:
        tmp.write_bytes(pdf_bytes)
        try:
            elements = partition_pdf(filename=str(tmp), strategy="fast")
        except Exception:
            elements = partition_pdf(filename=str(tmp))
        titles = [
            str(getattr(el, "text", "")).strip()
            for el in elements
            if el.__class__.__name__ == "Title" and str(getattr(el, "text", "")).strip()
        ]
        return titles[:50]
    except Exception as exc:
        logger.debug("unstructured partition failed: %s", exc)
        return []
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Line-based section segmentation
#
# PDFs rarely mark paragraphs reliably (PyMuPDF may emit single newlines
# with no blank-line separators), so segmentation works line-by-line:
# each line is tested as a potential heading, body lines accumulate into
# paragraphs (flushed on blank lines), and every paragraph keeps its
# page_number / paragraph_id attribution.
# ---------------------------------------------------------------------------

_RUN_TOGETHER_RE = re.compile(
    r"^\s*(abstract|introduction|methodology|methods?|results?|"
    r"discussion|conclusions?|limitations?|future work|references)\b"
    r"[:.\-–—]?\s+(.{40,})$",
    re.IGNORECASE,
)


def segment_pages(
    pages: list[dict[str, Any]],
    unstructured_titles: Optional[list[str]] = None,
) -> list[PaperSection]:
    """Group page lines into paragraphs under canonical section headings."""
    hint_set = {t.strip().lower() for t in (unstructured_titles or []) if t.strip()}
    order = CANONICAL_SECTIONS + ["References"]
    sections: dict[str, PaperSection] = {name: PaperSection(name=name) for name in order}
    preamble = PaperSection(name="Preamble")
    current: PaperSection = preamble
    seen_any_heading = False

    buf: list[str] = []
    buf_page = 1
    counter = 0

    def flush() -> None:
        nonlocal counter, buf
        text = re.sub(r"[ \t]+", " ", " ".join(buf)).strip()
        buf = []
        if len(text) < 2:
            return
        counter += 1
        current.paragraphs.append(
            ParsedParagraph(
                paragraph_id=f"p{counter}",
                page_number=buf_page,
                text=preserve_math(text),
            )
        )

    def hint_heading(candidate: str) -> Optional[str]:
        norm = candidate.strip().lower()
        if not norm:
            return None
        for hint in hint_set:
            if norm == hint or SequenceMatcher(None, norm, hint).ratio() > 0.92:
                resolved = detect_section_heading(hint)
                if resolved:
                    return resolved
        return None

    for page in pages:
        page_no = int(page.get("page_number", 1))
        for raw_line in (page.get("text", "") or "").splitlines():
            line = raw_line.strip()
            if not line:
                flush()
                continue
            heading = detect_section_heading(line) or hint_heading(line)
            remainder: Optional[str] = None
            if heading is None and len(line) > 45:
                m = _RUN_TOGETHER_RE.match(line)
                if m:
                    heading = (
                        detect_section_heading(m.group(1))
                        or _ALIAS_TO_CANONICAL.get(m.group(1).lower())
                    )
                    remainder = m.group(2).strip()
            if heading:
                flush()
                seen_any_heading = True
                current = sections[heading]
                buf_page = page_no
                if remainder:
                    buf = [remainder]
                continue
            if not buf:
                buf_page = page_no
            buf.append(line)
    flush()

    # Orphan title block: fold into Introduction only when the PDF has no
    # headings at all; otherwise keep it as a separate Preamble section.
    result: list[PaperSection] = []
    if preamble.paragraphs:
        if seen_any_heading:
            preamble.text = "\n\n".join(p.text for p in preamble.paragraphs)
            result.append(preamble)
        else:
            sections["Introduction"].paragraphs = (
                preamble.paragraphs + sections["Introduction"].paragraphs
            )
    for name in order:
        sec = sections[name]
        sec.text = "\n\n".join(p.text for p in sec.paragraphs)
        if name == "References" and not sec.paragraphs:
            continue  # non-canonical bucket, omit when empty
        result.append(sec)
    return result


def split_into_paragraphs(pages: list[dict[str, Any]]) -> list[ParsedParagraph]:
    """All paragraphs across *pages* (ungrouped). Kept for tests/diagnostics."""
    paras: list[ParsedParagraph] = []
    for sec in segment_pages(pages):
        paras.extend(sec.paragraphs)
    # Re-number sequentially for a flat view.
    for i, p in enumerate(paras, start=1):
        p.paragraph_id = f"p{i}"
    return paras


# ---------------------------------------------------------------------------
# Metadata: heuristics + Semantic Scholar fallback
# ---------------------------------------------------------------------------


def _heuristic_metadata(
    pdf_bytes: bytes,
    pages: list[dict[str, Any]],
    paragraphs: list[ParsedParagraph],
    filename: str,
) -> PaperMetadata:
    first_text = (pages[0]["text"] if pages else "") or ""
    lines = [ln.strip() for ln in first_text.splitlines() if ln.strip()]

    title = extract_title_candidate_pymupdf(pdf_bytes)
    if not title:
        title = next((ln for ln in lines if len(ln) >= 15), None) or Path(
            filename
        ).stem.replace("_", " ")

    # Authors: lines between title and 'abstract' containing person-like cues.
    authors: list[str] = []
    try:
        title_idx = next(
            i for i, ln in enumerate(lines) if title[:25].lower() in ln.lower()
        )
    except StopIteration:
        title_idx = 0
    for ln in lines[title_idx + 1 : title_idx + 5]:
        if re.search(r"abstract", ln, re.IGNORECASE):
            break
        if re.search(r"@|university|institute|college|school|department|laboratory|lab\b", ln, re.IGNORECASE):
            continue
        if "," in ln or re.search(r"\band\b", ln, re.IGNORECASE) or len(ln.split()) in (2, 3, 4):
            if len(ln) < 200 and not re.search(r"\d", ln):
                parts = re.split(r",|\band\b|;|·|\|", ln)
                for part in parts:
                    name = part.strip()
                    if 1 <= len(name.split()) <= 4 and len(name) >= 3 and not re.search(
                        r"proceedings|conference|journal|arxiv|doi|http", name, re.IGNORECASE
                    ):
                        authors.append(name)
                if authors:
                    break

    # Year: arXiv stamp > explicit years on first pages > filename.
    year: Optional[int] = None
    head = "\n".join(p.get("text", "") for p in pages[:2])
    m = _ARXIV_RE.search(head)
    if m:
        yy = int(m.group(1)[:2])
        year = 2000 + yy if yy < 50 else 1900 + yy
    if year is None:
        years = [int(y) for y in _YEAR_RE.findall(head + "\n" + filename)]
        years = [y for y in years if 1990 <= y <= 2100]
        year = max(years) if years else None

    doi_m = _DOI_RE.search(head)
    return PaperMetadata(
        title=title.strip(),
        authors=authors[:20],
        year=year,
        doi=doi_m.group(0) if doi_m else None,
    )


def enrich_via_semantic_scholar(meta: PaperMetadata, timeout: float = 10.0) -> PaperMetadata:
    """Fill missing title/authors/year/abstract via Semantic Scholar search.

    Best-effort: any network/API error leaves *meta* unchanged.
    """
    if not meta.title:
        return meta
    try:
        resp = httpx.get(
            f"{SEMANTIC_SCHOLAR_API_URL}/paper/search",
            params={
                "query": meta.title,
                "limit": 5,
                "fields": "title,authors,year,abstract,venue,url,externalIds",
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.debug("semantic scholar status=%s", resp.status_code)
            return meta
        items = (resp.json() or {}).get("data", []) or []
        if not items:
            return meta

        def norm(s: str) -> str:
            return re.sub(r"\W+", "", (s or "").lower())

        target = norm(meta.title)
        best: Optional[dict[str, Any]] = None
        for item in items:
            cand = norm(str(item.get("title", "")))
            if cand and (cand == target or cand.startswith(target[:40]) or target.startswith(cand[:40])):
                best = item
                break
        best = best or items[0]
        # Only adopt the match if titles genuinely resemble each other.
        ratio = SequenceMatcher(
            None, target, norm(str(best.get("title", "")))
        ).ratio()
        if ratio < 0.6:
            return meta
        if best.get("title"):
            meta.title = str(best["title"])
        api_authors = [
            a.get("name") for a in (best.get("authors") or []) if a.get("name")
        ]
        if api_authors and (not meta.authors or len(meta.authors) <= 1):
            meta.authors = api_authors[:20]
        if best.get("year") and not meta.year:
            meta.year = int(best["year"])
        if best.get("abstract") and not meta.abstract:
            meta.abstract = str(best["abstract"])
        if best.get("venue") and not meta.venue:
            meta.venue = str(best["venue"])
        ext = best.get("externalIds") or {}
        if ext.get("DOI") and not meta.doi:
            meta.doi = str(ext["DOI"])
        if best.get("url") and not meta.url:
            meta.url = str(best["url"])
        meta.enrichment_source = "semantic-scholar"
        return meta
    except Exception as exc:
        logger.debug("semantic scholar enrichment failed: %s", exc)
        return meta


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def parse_pdf_bytes(
    pdf_bytes: bytes,
    filename: str = "paper.pdf",
    enrich: bool = True,
) -> ParsedPaper:
    """Parse PDF bytes into a structured :class:`ParsedPaper`.

    Raises:
        ValueError: empty file, missing ``%PDF`` magic, or zero pages.
    """
    if not pdf_bytes:
        raise ValueError("Empty file.")
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Not a PDF file (missing %PDF header).")

    pages = extract_pages_with_pymupdf(pdf_bytes)
    titles_hint = try_unstructured_titles(pdf_bytes)
    sections = segment_pages(pages, unstructured_titles=titles_hint)
    paragraphs = [p for sec in sections for p in sec.paragraphs]
    if not paragraphs:
        raise ValueError("No extractable text found in PDF.")
    meta = _heuristic_metadata(pdf_bytes, pages, paragraphs, filename)
    if enrich:
        meta = enrich_via_semantic_scholar(meta)

    # Surface abstract text on metadata when the Abstract section exists.
    if not meta.abstract:
        for sec in sections:
            if sec.name == "Abstract" and sec.text.strip():
                meta.abstract = sec.text.strip()[:2000]
                break

    return ParsedPaper(
        paper_id=str(uuid.uuid4()),
        filename=filename,
        metadata=meta,
        sections=sections,
        page_count=len(pages),
        paragraph_count=len(paragraphs),
    )


def get_section(paper: ParsedPaper, name: str) -> Optional[PaperSection]:
    for sec in paper.sections:
        if sec.name.lower() == name.lower():
            return sec
    return None
