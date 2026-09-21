"""ScholarSense AI — FastAPI entrypoint."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ScholarSense AI",
    description="AI Research Paper Assistant: summarize, extract, compare, and find gaps.",
    version="0.1.0",
)

# Comma-separated extra origins, e.g. ALLOWED_ORIGINS=https://a.vercel.app,https://b.example.com
_extra_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
_allow_origins = [
    "http://localhost:3000",
    "https://scholar-sence-ai.vercel.app",
    *_extra_origins,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "scholarsense-backend"}


@app.get("/api/v1/health")
def api_health_check() -> dict:
    return {"status": "ok", "api_version": "v1"}


# Routers for Steps 2-5.
try:  # `uvicorn backend.main:app` from repo root
    from backend.api.v1.papers import router as papers_router
    from backend.api.v1.extractions import router as extractions_router
    from backend.api.v1.synthesis import router as synthesis_router
except ImportError:  # `uvicorn main:app` from /backend
    from api.v1.papers import router as papers_router  # type: ignore[no-redef]
    from api.v1.extractions import router as extractions_router  # type: ignore[no-redef]
    from api.v1.synthesis import router as synthesis_router  # type: ignore[no-redef]

app.include_router(papers_router, prefix="/api/v1/papers", tags=["papers"])
app.include_router(extractions_router, prefix="/api/v1/papers", tags=["extractions"])
app.include_router(synthesis_router, prefix="/api/v1/papers", tags=["synthesis"])
