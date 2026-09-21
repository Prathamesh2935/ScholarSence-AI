"""ScholarSense AI pipeline tests (Step 7).

Covers: PDF parsing into JSON structures, schema validation on extraction
functions, and source-citation attribution integrity — plus vector-store,
synthesis, and API endpoint tests. Offline by design: no API keys, no
network (enrich=False, heuristic extractors, local fallbacks).
"""

from __future__ import annotations

import os
import sys

# Make `backend.*` imports work whether pytest runs from the repo root or
# from /backend.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "backend") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "backend"))

import pytest
import pymupdf
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.main import app
from backend.schemas.paper_schema import (
    HierarchicalSummary,
    PaperExtraction,
    SummaryBullet,
)
from backend.services.extractor import (
    build_context,
    heuristic_extract,
    repair_sources,
    validate_citations,
)
from backend.services.ingestion import (
    CANONICAL_SECTIONS,
    parse_pdf_bytes,
    preserve_math,
)
from backend.services.synthesis import compare_papers, identify_research_gaps
from backend.services.vector_store import (
    HashEmbeddingBackend,
    chunk_paper,
    reset_store,
)

BOX = pymupdf.Rect(72, 72, 523, 770)

TRANSLATION_LINES = [
    "Loss-Weighted Transformers for Translation",
    "Alice Researcher, Bob Scientist",
    "MIT",
    "",
    "Abstract",
    "We present a loss-weighted transformer. Our objective is $L_{loss} = \\lambda \\cdot L_{rec} + \\beta \\cdot D_{KL}$.",
    "Greek symbols inline: \u03bb \u03b2 \u00b7 evaluated.",
    "",
    "1. Introduction",
    "Neural translation quality depends on training objectives.",
    "",
    "2. Methodology",
    "We use a 12-layer transformer with 8 attention heads and hidden size 512.",
    "We optimize with Adam at learning rate 3e-4, batch size 32, for 100 epochs with dropout 0.1.",
    "Training minimizes $L_{loss}$ on the WMT14 English-German dataset.",
    "Experiments run on 8xV100 GPUs.",
    "",
    "3. Results",
    "BLEU score improves from 24.1 to 28.4 on WMT14 newstest.",
    "",
    "4. Discussion",
    "Loss weighting stabilizes early training dynamics.",
    "",
    "5. Limitations",
    "Only English-German evaluated; long documents untested.",
    "",
    "6. Future Work",
    "Extend to multilingual benchmarks.",
]

RECURRENT_LINES = [
    "Recurrent Baselines Without Attention",
    "Bob Scientist",
    "Stanford",
    "",
    "Abstract",
    "We revisit recurrent models for translation on WMT14.",
    "",
    "Methodology",
    "We use a 4-layer LSTM with SGD at learning rate 1e-2 on the WMT14 dataset.",
    "",
    "Results",
    "BLEU score drops from 26.0 to 24.5 on WMT14 newstest without attention.",
    "",
    "Limitations",
    "The model fails on long documents; only WMT14 newstest evaluated.",
    "",
    "Future Work",
    "Multilingual extension and ablation of the gating mechanism.",
    "We assume fixed hyperparameters and report no sensitivity analysis.",
]


def pdf_bytes(lines: list[str], n_pages: int = 1) -> bytes:
    doc = pymupdf.open()
    paras = "\n".join(lines).split("\n\n")
    per_page = max(1, (len(paras) + n_pages - 1) // n_pages)
    for i in range(n_pages):
        doc.new_page().insert_textbox(
            BOX, "\n\n".join(paras[i * per_page : (i + 1) * per_page]), fontsize=11
        )
    return doc.tobytes()


@pytest.fixture()
def translation_pdf() -> bytes:
    return pdf_bytes(TRANSLATION_LINES)


@pytest.fixture()
def paper_a(translation_pdf: bytes):
    return parse_pdf_bytes(translation_pdf, "translation.pdf", enrich=False)


@pytest.fixture()
def paper_b():
    return parse_pdf_bytes(pdf_bytes(RECURRENT_LINES), "recurrent.pdf", enrich=False)


# ---------------------------------------------------------------------------
# Module A: ingestion
# ---------------------------------------------------------------------------


class TestIngestion:
    def test_parse_pdf_into_json_structures(self, paper_a):
        assert paper_a.page_count == 1
        assert paper_a.paragraph_count > 0
        names = [s.name for s in paper_a.sections]
        for canon in CANONICAL_SECTIONS:
            assert canon in names, f"missing section: {canon}"
        # JSON-serializable structured representation
        data = paper_a.model_dump(mode="json")
        assert data["metadata"]["title"].startswith("Loss-Weighted")
        assert data["metadata"]["authors"] == ["Alice Researcher", "Bob Scientist"]
        assert isinstance(data["sections"], list)

    def test_metadata_title_authors(self, paper_a):
        assert "Loss-Weighted" in (paper_a.metadata.title or "")
        assert len(paper_a.metadata.authors) >= 1

    def test_numbered_headings_segmented(self, paper_a):
        by_name = {s.name: s for s in paper_a.sections}
        assert "BLEU" in by_name["Results"].text
        assert "V100" in by_name["Methodology"].text
        assert "untested" in by_name["Limitations"].text

    def test_math_preserved_as_latex(self, paper_a):
        abstract = next(s for s in paper_a.sections if s.name == "Abstract").text
        assert "$L_{loss}" in abstract
        assert "\\lambda" in abstract and "\\beta" in abstract

    def test_preserve_math_unit(self):
        assert "\\lambda" in preserve_math("loss = \u03bb \u00b7 x")
        kept = "$L_{loss} = \\lambda \\cdot L_{rec}$"
        assert preserve_math(f"a {kept} b") == f"a {kept} b"

    def test_reject_empty_non_pdf_corrupt(self):
        with pytest.raises(ValueError):
            parse_pdf_bytes(b"", "e.pdf", enrich=False)
        with pytest.raises(ValueError):
            parse_pdf_bytes(b"hello world", "n.pdf", enrich=False)
        with pytest.raises(ValueError):
            parse_pdf_bytes(b"%PDF-1.4 garbage", "f.pdf", enrich=False)


# ---------------------------------------------------------------------------
# Core: vector store & hybrid retrieval
# ---------------------------------------------------------------------------


class TestVectorStore:
    def test_chunking_preserves_attribution(self, paper_a):
        chunks = chunk_paper(paper_a, target_chars=200)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.paper_id == paper_a.paper_id
            assert c.page_number >= 1
            assert c.paragraph_ids

    def test_chunking_multi_page(self):
        paper = parse_pdf_bytes(
            pdf_bytes(TRANSLATION_LINES, n_pages=2), "m.pdf", enrich=False
        )
        pages = {pg for c in chunk_paper(paper, target_chars=200) for pg in c.pages}
        assert pages == {1, 2}

    def test_hybrid_search_returns_cited_hits(self, paper_a, paper_b):
        store = reset_store(embedding_backend=HashEmbeddingBackend())
        store.upsert_paper(paper_a, target_chars=200)
        store.upsert_paper(paper_b, target_chars=200)
        hits = store.hybrid_search("BLEU score WMT14", top_k=3)
        assert hits
        top = hits[0]
        assert top.citation.paper_id == paper_a.paper_id
        assert top.citation.section == "Results"
        assert top.citation.page_number >= 1
        assert top.citation.paragraph_ids
        assert top.bm25_score > 0

    def test_search_scoping_and_empty_query(self, paper_a, paper_b):
        store = reset_store(embedding_backend=HashEmbeddingBackend())
        store.upsert_paper(paper_a, target_chars=200)
        store.upsert_paper(paper_b, target_chars=200)
        scoped = store.hybrid_search("BLEU", paper_ids=[paper_b.paper_id])
        assert scoped
        assert all(h.citation.paper_id == paper_b.paper_id for h in scoped)
        assert store.hybrid_search("") == []


# ---------------------------------------------------------------------------
# Module B: schemas + extraction
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_extraction_json_schema(self):
        schema = PaperExtraction.model_json_schema()
        assert schema["type"] == "object"
        assert "summary" in schema["properties"]

    def test_executive_summary_requires_three_bullets(self):
        with pytest.raises(ValidationError):
            HierarchicalSummary(
                executive_summary=[SummaryBullet(text="only", sources=[])],
                technical_tldr="Short.",
            )

    def test_heuristic_extraction_validates(self, paper_a):
        ext = heuristic_extract(paper_a)
        assert isinstance(ext, PaperExtraction)
        assert len(ext.summary.executive_summary) == 3
        assert ext.summary.technical_tldr.strip()
        hp = {h.name: h.value for h in ext.methodology.hyperparameters}
        assert hp.get("learning_rate") == "3e-4"
        assert hp.get("optimizer") == "Adam"
        assert ext.methodology.loss_functions
        assert ext.methodology.loss_functions[0].expression_latex.startswith("$")
        assert any(d.name == "WMT14" for d in ext.methodology.datasets)
        assert "V100" in ext.methodology.hardware_requirements
        bleu = next(
            r for r in ext.quantitative_results.results if r.metric_name == "BLEU"
        )
        assert (bleu.baseline_score, bleu.paper_score) == (24.1, 28.4)
        assert abs(bleu.sota_delta - 4.3) < 1e-6

    def test_build_context_tags_paragraphs(self, paper_a):
        ctx = build_context(paper_a)
        assert "TITLE:" in ctx and "## Results" in ctx and "[p" in ctx


# ---------------------------------------------------------------------------
# Citation attribution integrity
# ---------------------------------------------------------------------------


def _all_tags(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("page_number"), int) and "paragraph_id" in obj:
            yield obj
        else:
            for v in obj.values():
                yield from _all_tags(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_tags(v)


class TestCitations:
    def test_extraction_citations_clean(self, paper_a):
        ext = heuristic_extract(paper_a)
        assert validate_citations(ext, paper_a) == []

    def test_every_tag_grounded(self, paper_a):
        ext = heuristic_extract(paper_a)
        valid = {
            p["paragraph_id"]
            for s in paper_a.model_dump()["sections"]
            for p in s["paragraphs"]
        }
        tags = list(_all_tags(ext.model_dump()))
        assert tags
        assert all(t["paragraph_id"] in valid for t in tags)

    def test_repair_drops_hallucinated_tags(self, paper_a):
        ext = heuristic_extract(paper_a)
        bad = ext.model_copy(deep=True)
        bogus = bad.summary.executive_summary[0].sources[0].model_copy(
            update={"paragraph_id": "p9999"}
        )
        bad.summary.executive_summary[0].sources.append(bogus)
        fixed = repair_sources(bad, paper_a)
        assert validate_citations(fixed, paper_a) == []
        assert not [
            t for t in _all_tags(fixed.model_dump()) if t["paragraph_id"] == "p9999"
        ]

    def test_compare_cells_and_gaps_sourced(self, paper_a, paper_b):
        pa, pb = paper_a.model_dump(), paper_b.model_dump()
        comp = compare_papers([pa, pb])
        assert any(r.verdict == "contradiction" for r in comp.rows)
        by_id = {pa["paper_id"]: pa, pb["paper_id"]: pb}
        for r in comp.rows:
            for c in r.cells:
                if c.value == "—":
                    continue
                valid = {
                    p["paragraph_id"]
                    for s in by_id[c.paper_id]["sections"]
                    for p in s["paragraphs"]
                }
                assert c.sources
                assert all(t.paragraph_id in valid for t in c.sources)
        gaps = identify_research_gaps([pa, pb])
        assert gaps.failure_modes and gaps.dataset_shortfalls
        assert gaps.methodological_blinds
        assert any(len(g.paper_ids) >= 2 for g in gaps.cross_cutting)
        for bucket in (
            gaps.failure_modes,
            gaps.dataset_shortfalls,
            gaps.methodological_blinds,
            gaps.cross_cutting,
        ):
            for g in bucket:
                assert g.sources and g.paper_ids


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def uploaded_ids(client: TestClient) -> list[str]:
    ids = []
    for name, lines in (
        ("translation.pdf", TRANSLATION_LINES),
        ("recurrent.pdf", RECURRENT_LINES),
    ):
        resp = client.post(
            "/api/v1/papers/upload?enrich=false",
            files={"file": (name, pdf_bytes(lines), "application/pdf")},
        )
        assert resp.status_code == 201, resp.text
        ids.append(resp.json()["paper"]["paper_id"])
    return ids


class TestAPI:
    def test_upload_list_detail_search(self, client, translation_pdf):
        up = client.post(
            "/api/v1/papers/upload?enrich=false",
            files={"file": ("t.pdf", translation_pdf, "application/pdf")},
        )
        assert up.status_code == 201
        pid = up.json()["paper"]["paper_id"]
        assert client.get("/api/v1/papers").json()["count"] >= 1
        assert client.get(f"/api/v1/papers/{pid}").status_code == 200
        res = client.get(
            "/api/v1/papers/search", params={"q": "BLEU WMT14", "top_k": 3}
        )
        assert res.status_code == 200
        assert res.json()["results"]

    def test_extract_and_citation_check(self, client, uploaded_ids):
        pid = uploaded_ids[0]
        r = client.post(f"/api/v1/papers/{pid}/extract", params={"prefer": "heuristic"})
        assert r.status_code == 200, r.text
        assert len(r.json()["summary"]["executive_summary"]) == 3
        g = client.get(f"/api/v1/papers/{pid}/extraction")
        assert g.status_code == 200
        c = client.get(f"/api/v1/papers/{pid}/extraction/citations")
        assert c.json()["clean"] is True, c.json()

    def test_compare_and_gaps_endpoints(self, client, uploaded_ids):
        cr = client.post("/api/v1/papers/compare", json={"paper_ids": uploaded_ids})
        assert cr.status_code == 200, cr.text
        assert len(cr.json()["contradictions"]) == 1
        gr = client.post("/api/v1/papers/gaps", json={"paper_ids": uploaded_ids})
        assert gr.status_code == 200, gr.text
        gj = gr.json()
        assert gj["failure_modes"] and gj["dataset_shortfalls"]
        assert gj["cross_cutting"]

    def test_error_paths(self, client, uploaded_ids):
        bad = client.post(
            "/api/v1/papers/upload",
            files={"file": ("n.txt", b"hello", "text/plain")},
        )
        assert bad.status_code == 400
        assert client.get("/api/v1/papers/missing/extraction").status_code == 404
        assert (
            client.post(
                "/api/v1/papers/compare", json={"paper_ids": [uploaded_ids[0], "nope"]}
            ).status_code
            == 404
        )
        assert client.post("/api/v1/papers/gaps", json={"paper_ids": []}).status_code == 422
