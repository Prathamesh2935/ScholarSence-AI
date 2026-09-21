"use client";

import type { ParsedPaper } from "@/lib/api";
import MathText from "./MathText";

/** Full paper viewer: sections with KaTeX math + paragraph citations. */
export default function PaperViewer({ paper }: { paper: ParsedPaper }) {
  const meta = paper.metadata;
  return (
    <div className="space-y-6">
      <header className="rounded-2xl border border-slate-200 bg-white p-5">
        <h2 className="text-xl font-bold">{meta.title ?? paper.filename}</h2>
        <p className="mt-1 text-sm text-slate-600">
          {(meta.authors ?? []).join(", ") || "Unknown authors"}
          {meta.year ? ` · ${meta.year}` : ""}
          {meta.venue ? ` · ${meta.venue}` : ""}
        </p>
        {meta.abstract && (
          <p className="mt-3 text-sm text-slate-700">
            <MathText text={meta.abstract} />
          </p>
        )}
      </header>

      {paper.sections.map((sec) => (
        <section
          key={sec.name}
          className="rounded-2xl border border-slate-200 bg-white p-5"
        >
          <h3 className="mb-3 text-base font-semibold text-indigo-700">
            {sec.name}
          </h3>
          {sec.paragraphs.length === 0 ? (
            <p className="text-sm italic text-slate-400">
              Not present in this PDF.
            </p>
          ) : (
            <div className="space-y-3">
              {sec.paragraphs.map((p) => (
                <p key={p.paragraph_id} className="text-sm leading-relaxed">
                  <MathText text={p.text} />
                  <span
                    title={`Section ${sec.name}, page ${p.page_number}`}
                    className="ml-1.5 inline-flex items-center rounded-full bg-slate-100 px-1.5 py-px text-[10px] font-medium text-slate-500"
                  >
                    {p.paragraph_id} · p.{p.page_number}
                  </span>
                </p>
              ))}
            </div>
          )}
        </section>
      ))}
    </div>
  );
}
