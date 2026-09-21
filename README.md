# ScholarSense AI

AI Research Paper Assistant — summarize papers, extract methodology/results into
structured schemas, compare multiple papers in a side-by-side matrix, and
identify research gaps.

## Monorepo layout

- `/backend` — Python FastAPI application.
- `/frontend` — Next.js 14+ (App Router, TypeScript, Tailwind CSS, Lucide Icons).

## Quickstart

### Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```
