"use client";

import { GitCompareArrows } from "lucide-react";
import type { PaperComparison } from "@/lib/api";
import { Cite } from "./Citation";
import MathText from "./MathText";

const VERDICT_STYLE: Record<string, string> = {
  consensus: "bg-emerald-100 text-emerald-800",
  contradiction: "bg-rose-100 text-rose-800",
  divergent: "bg-amber-100 text-amber-800",
  "single-source": "bg-slate-100 text-slate-600",
  "n/a": "bg-slate-100 text-slate-500",
};

export default function MatrixTab({
  comparison,
}: {
  comparison: PaperComparison;
}) {
  return (
    <div className="space-y-6">
      {(comparison.contradictions.length > 0 ||
        comparison.consensus_points.length > 0) && (
        <section className="grid gap-3 md:grid-cols-2">
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
            <h4 className="text-sm font-semibold text-emerald-800">
              Consensus ({comparison.consensus_points.length})
            </h4>
            <ul className="mt-2 space-y-1.5 text-sm text-emerald-900">
              {comparison.consensus_points.map((f, i) => (
                <li key={i}>
                  <MathText text={f.statement} />
                  <Cite tags={f.sources} />
                </li>
              ))}
            </ul>
          </div>
          <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4">
            <h4 className="text-sm font-semibold text-rose-800">
              Contradictions ({comparison.contradictions.length})
            </h4>
            <ul className="mt-2 space-y-1.5 text-sm text-rose-900">
              {comparison.contradictions.map((f, i) => (
                <li key={i}>
                  <MathText text={f.statement} />
                  <Cite tags={f.sources} />
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      <section className="rounded-2xl border border-slate-200 bg-white p-5">
        <h3 className="flex items-center gap-2 text-base font-semibold">
          <GitCompareArrows className="h-4 w-4 text-indigo-600" />
          Comparison matrix
        </h3>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="border-b text-left">
                <th className="py-2 pr-3 text-xs uppercase text-slate-500">
                  Dimension
                </th>
                {comparison.papers.map((p) => (
                  <th key={p.paper_id} className="py-2 pr-3 text-xs">
                    <span className="line-clamp-3 font-semibold">{p.title}</span>
                    <span className="font-normal text-slate-500">
                      {p.year ?? ""}
                    </span>
                  </th>
                ))}
                <th className="py-2 text-xs uppercase text-slate-500">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {comparison.rows.map((r) => (
                <tr key={r.dimension} className="border-b align-top last:border-0">
                  <td className="py-2 pr-3 font-mono text-xs text-slate-600">
                    {r.dimension}
                    {r.note && (
                      <span className="block font-sans text-[11px] text-slate-400">
                        {r.note}
                      </span>
                    )}
                  </td>
                  {r.cells.map((c) => (
                    <td key={c.paper_id} className="py-2 pr-3">
                      <MathText text={c.value} />
                      <Cite tags={c.sources} />
                    </td>
                  ))}
                  <td className="py-2">
                    <span
                      className={`inline-flex rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                        VERDICT_STYLE[r.verdict] ?? VERDICT_STYLE["n/a"]
                      }`}
                    >
                      {r.verdict}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
