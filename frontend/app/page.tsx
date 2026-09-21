"use client";

import {
  BookOpenText,
  FlaskConical,
  GitCompareArrows,
  Loader2,
  ScanSearch,
  Sparkles,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import GapsTab from "@/components/GapsTab";
import MatrixTab from "@/components/MatrixTab";
import PaperSidebar from "@/components/PaperSidebar";
import PaperViewer from "@/components/PaperViewer";
import SummaryTab from "@/components/SummaryTab";
import UploadPanel from "@/components/UploadPanel";
import {
  comparePapers,
  extractPaper,
  fetchGaps,
  getPaper,
  listPapers,
  type PaperComparison,
  type PaperExtraction,
  type PaperListItem,
  type ParsedPaper,
  type ResearchGaps,
} from "@/lib/api";

type Tab = "viewer" | "summary" | "matrix" | "gaps";

const TABS: { id: Tab; label: string; icon: typeof BookOpenText }[] = [
  { id: "viewer", label: "Paper Viewer", icon: BookOpenText },
  { id: "summary", label: "Structured Summary", icon: FlaskConical },
  { id: "matrix", label: "Comparison Matrix", icon: GitCompareArrows },
  { id: "gaps", label: "Research Gaps", icon: ScanSearch },
];

export default function Home() {
  const [papers, setPapers] = useState<PaperListItem[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [activePaper, setActivePaper] = useState<ParsedPaper | null>(null);
  const [checkedIds, setCheckedIds] = useState<string[]>([]);
  const [tab, setTab] = useState<Tab>("viewer");
  const [extractions, setExtractions] = useState<Record<string, PaperExtraction>>({});
  const [comparison, setComparison] = useState<PaperComparison | null>(null);
  const [gaps, setGaps] = useState<ResearchGaps | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshList = useCallback(async () => {
    try {
      const data = await listPapers();
      setPapers(data.papers);
    } catch (e) {
      setError(
        e instanceof Error
          ? `Backend unreachable: ${e.message}`
          : "Backend unreachable.",
      );
    }
  }, []);

  useEffect(() => {
    void refreshList();
  }, [refreshList]);

  const selectPaper = useCallback(
    async (id: string) => {
      setActiveId(id);
      setError(null);
      try {
        const paper = await getPaper(id);
        setActivePaper(paper);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load paper.");
      }
    },
    [],
  );

  function handleUploaded(paper: ParsedPaper) {
    void refreshList();
    setActiveId(paper.paper_id);
    setActivePaper(paper);
    setCheckedIds((prev) =>
      prev.includes(paper.paper_id) ? prev : [...prev, paper.paper_id],
    );
    setTab("viewer");
  }

  async function handleExtract() {
    if (!activeId) return;
    setBusy("extract");
    setError(null);
    try {
      const ext = await extractPaper(activeId, "auto");
      setExtractions((prev) => ({ ...prev, [activeId]: ext }));
      setTab("summary");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Extraction failed.");
    } finally {
      setBusy(null);
    }
  }

  async function handleCompare() {
    if (checkedIds.length === 0) {
      setError("Select at least one paper (checkbox) to compare.");
      return;
    }
    setBusy("compare");
    setError(null);
    try {
      setComparison(await comparePapers(checkedIds));
      setTab("matrix");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Comparison failed.");
    } finally {
      setBusy(null);
    }
  }

  async function handleGaps() {
    if (checkedIds.length === 0) {
      setError("Select at least one paper (checkbox) for gap analysis.");
      return;
    }
    setBusy("gaps");
    setError(null);
    try {
      setGaps(await fetchGaps(checkedIds));
      setTab("gaps");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gap analysis failed.");
    } finally {
      setBusy(null);
    }
  }

  const activeExtraction = activeId ? extractions[activeId] : undefined;

  return (
    <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <header className="mb-6">
        <p className="text-sm font-semibold uppercase tracking-widest text-indigo-600">
          ScholarSense AI
        </p>
        <h1 className="mt-1 text-3xl font-bold tracking-tight">
          AI Research Paper Assistant
        </h1>
        <p className="mt-1 text-sm text-slate-600">
          Summarize papers, extract methodology and results, compare side by
          side, and discover research gaps — every claim cited to its source.
        </p>
      </header>

      {error && (
        <div className="mb-4 rounded-2xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
          {error}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
        {/* Upload & workspace panel */}
        <aside className="space-y-4">
          <UploadPanel
            onUploaded={handleUploaded}
            onError={(m) => setError(m)}
          />
          <div>
            <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Workspace ({papers.length})
            </h2>
            <PaperSidebar
              papers={papers}
              activeId={activeId}
              checkedIds={checkedIds}
              onSelect={(id) => void selectPaper(id)}
              onToggleCheck={(id) =>
                setCheckedIds((prev) =>
                  prev.includes(id)
                    ? prev.filter((x) => x !== id)
                    : [...prev, id],
                )
              }
            />
          </div>
        </aside>

        {/* Main panel */}
        <div>
          <div className="mb-4 flex flex-wrap items-center gap-2">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-sm font-medium transition ${
                  tab === t.id
                    ? "bg-slate-900 text-white"
                    : "bg-white text-slate-600 ring-1 ring-inset ring-slate-200 hover:ring-indigo-300"
                }`}
              >
                <t.icon className="h-4 w-4" />
                {t.label}
              </button>
            ))}
            <div className="ml-auto flex gap-2">
              <button
                onClick={() => void handleExtract()}
                disabled={!activeId || busy === "extract"}
                className="inline-flex items-center gap-1.5 rounded-full bg-indigo-600 px-3.5 py-1.5 text-sm font-medium text-white disabled:opacity-50"
              >
                {busy === "extract" ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
                Extract
              </button>
              <button
                onClick={() => void handleCompare()}
                disabled={busy === "compare"}
                className="inline-flex items-center gap-1.5 rounded-full bg-white px-3.5 py-1.5 text-sm font-medium text-slate-700 ring-1 ring-inset ring-slate-200 disabled:opacity-50"
              >
                {busy === "compare" ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <GitCompareArrows className="h-4 w-4" />
                )}
                Compare ({checkedIds.length})
              </button>
              <button
                onClick={() => void handleGaps()}
                disabled={busy === "gaps"}
                className="inline-flex items-center gap-1.5 rounded-full bg-white px-3.5 py-1.5 text-sm font-medium text-slate-700 ring-1 ring-inset ring-slate-200 disabled:opacity-50"
              >
                {busy === "gaps" ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <ScanSearch className="h-4 w-4" />
                )}
                Gaps
              </button>
            </div>
          </div>

          {tab === "viewer" &&
            (activePaper ? (
              <PaperViewer paper={activePaper} />
            ) : (
              <EmptyState text="Select or upload a paper to view its sections with rendered math." />
            ))}

          {tab === "summary" &&
            (activeExtraction ? (
              <SummaryTab extraction={activeExtraction} />
            ) : (
              <EmptyState
                text="No extraction yet. Open a paper and press Extract."
                action={
                  activeId ? (
                    <button
                      onClick={() => void handleExtract()}
                      className="mt-3 rounded-full bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white"
                    >
                      Extract structured data
                    </button>
                  ) : undefined
                }
              />
            ))}

          {tab === "matrix" &&
            (comparison ? (
              <MatrixTab comparison={comparison} />
            ) : (
              <EmptyState text="Select papers with the checkboxes, then press Compare." />
            ))}

          {tab === "gaps" &&
            (gaps ? (
              <GapsTab gaps={gaps} />
            ) : (
              <EmptyState text="Select papers with the checkboxes, then press Gaps." />
            ))}
        </div>
      </div>
    </main>
  );
}

function EmptyState({ text, action }: { text: string; action?: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-10 text-center text-sm text-slate-500">
      {text}
      {action}
    </div>
  );
}
