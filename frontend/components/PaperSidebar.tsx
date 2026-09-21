"use client";

import { BookOpenText } from "lucide-react";
import { shortTitle, type PaperListItem } from "@/lib/api";

export default function PaperSidebar({
  papers,
  activeId,
  checkedIds,
  onSelect,
  onToggleCheck,
}: {
  papers: PaperListItem[];
  activeId: string | null;
  checkedIds: string[];
  onSelect: (id: string) => void;
  onToggleCheck: (id: string) => void;
}) {
  if (papers.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-500">
        No papers yet. Upload a PDF to start your workspace.
      </div>
    );
  }
  return (
    <ul className="space-y-2">
      {papers.map((p) => {
        const active = p.paper_id === activeId;
        const checked = checkedIds.includes(p.paper_id);
        return (
          <li
            key={p.paper_id}
            className={`rounded-2xl border p-3 transition ${
              active
                ? "border-indigo-500 bg-indigo-50/50"
                : "border-slate-200 bg-white hover:border-indigo-300"
            }`}
          >
            <div className="flex items-start gap-2">
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onToggleCheck(p.paper_id)}
                title="Select for comparison / gaps"
                className="mt-1 h-4 w-4 accent-indigo-600"
              />
              <button onClick={() => onSelect(p.paper_id)} className="text-left">
                <div className="flex items-center gap-1.5 text-sm font-semibold">
                  <BookOpenText className="h-4 w-4 shrink-0 text-indigo-600" />
                  <span className="line-clamp-2">
                    {shortTitle(p.title, p.filename)}
                  </span>
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {p.year ?? "—"} · {p.page_count} pages
                </div>
              </button>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
