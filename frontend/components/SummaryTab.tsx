"use client";

import { FlaskConical, ListChecks, Table2, Zap } from "lucide-react";
import type { PaperExtraction } from "@/lib/api";
import { Cite } from "./Citation";
import MathText from "./MathText";

export default function SummaryTab({
  extraction,
}: {
  extraction: PaperExtraction;
}) {
  const { summary, methodology, quantitative_results } = extraction;
  return (
    <div className="space-y-6">
      {/* Executive summary + TL;DR */}
      <section className="rounded-2xl border border-slate-200 bg-white p-5">
        <h3 className="flex items-center gap-2 text-base font-semibold">
          <Zap className="h-4 w-4 text-indigo-600" /> Executive summary
        </h3>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-sm">
          {summary.executive_summary.map((b, i) => (
            <li key={i}>
              <MathText text={b.text} />
              <Cite tags={b.sources} />
            </li>
          ))}
        </ul>
        <p className="mt-4 rounded-xl bg-indigo-50 p-3 text-sm font-medium text-indigo-900">
          TL;DR: <MathText text={summary.technical_tldr} />
          <Cite tags={summary.technical_tldr_sources} />
        </p>
        <p className="mt-2 text-[11px] text-slate-400">
          Source: {extraction.extraction_source}
        </p>
      </section>

      {/* Sectional summary */}
      <section className="rounded-2xl border border-slate-200 bg-white p-5">
        <h3 className="flex items-center gap-2 text-base font-semibold">
          <ListChecks className="h-4 w-4 text-indigo-600" /> Sectional summary
        </h3>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          {summary.sectional_summary.map((e) => (
            <div key={e.section} className="rounded-xl bg-slate-50 p-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-indigo-600">
                {e.section}
              </p>
              <p className="mt-1 text-sm">
                <MathText text={e.summary} />
                <Cite tags={e.sources} />
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Methodology cards */}
      <section className="rounded-2xl border border-slate-200 bg-white p-5">
        <h3 className="flex items-center gap-2 text-base font-semibold">
          <FlaskConical className="h-4 w-4 text-indigo-600" /> Methodology
        </h3>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <div className="rounded-xl bg-slate-50 p-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Architecture
            </p>
            <p className="mt-1 text-sm">
              <MathText text={methodology.architecture || "—"} />
              <Cite tags={methodology.architecture_sources} />
            </p>
          </div>
          <div className="rounded-xl bg-slate-50 p-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Hardware
            </p>
            <p className="mt-1 text-sm">
              {methodology.hardware_requirements || "—"}
              <Cite tags={methodology.hardware_sources} />
            </p>
          </div>
          <div className="rounded-xl bg-slate-50 p-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Hyperparameters
            </p>
            {methodology.hyperparameters.length === 0 ? (
              <p className="mt-1 text-sm text-slate-400">—</p>
            ) : (
              <dl className="mt-1 space-y-1 text-sm">
                {methodology.hyperparameters.map((h) => (
                  <div key={h.name} className="flex gap-2">
                    <dt className="font-mono text-xs text-slate-500">{h.name}</dt>
                    <dd>
                      {h.value}
                      <Cite tags={h.sources} />
                    </dd>
                  </div>
                ))}
              </dl>
            )}
          </div>
          <div className="rounded-xl bg-slate-50 p-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Datasets
            </p>
            {methodology.datasets.length === 0 ? (
              <p className="mt-1 text-sm text-slate-400">—</p>
            ) : (
              <ul className="mt-1 space-y-1 text-sm">
                {methodology.datasets.map((d) => (
                  <li key={d.name}>
                    <span className="font-semibold">{d.name}</span>
                    {d.details ? ` — ${d.details}` : ""}
                    <Cite tags={d.sources} />
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
        {methodology.loss_functions.length > 0 && (
          <div className="mt-3 rounded-xl bg-slate-50 p-3">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Loss functions
            </p>
            {methodology.loss_functions.map((l, i) => (
              <p key={i} className="mt-1 text-sm">
                {l.name && <span className="font-semibold">{l.name}: </span>}
                <MathText text={l.expression_latex} />
                {l.description && (
                  <span className="text-slate-600"> — {l.description}</span>
                )}
                <Cite tags={l.sources} />
              </p>
            ))}
          </div>
        )}
      </section>

      {/* Benchmark results */}
      <section className="rounded-2xl border border-slate-200 bg-white p-5">
        <h3 className="flex items-center gap-2 text-base font-semibold">
          <Table2 className="h-4 w-4 text-indigo-600" /> Benchmark results
        </h3>
        {quantitative_results.results.length === 0 ? (
          <p className="mt-2 text-sm text-slate-400">
            No quantitative results extracted.
          </p>
        ) : (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs uppercase text-slate-500">
                  <th className="py-2 pr-3">Task</th>
                  <th className="py-2 pr-3">Metric</th>
                  <th className="py-2 pr-3 text-right">Baseline</th>
                  <th className="py-2 pr-3 text-right">Paper</th>
                  <th className="py-2 pr-3 text-right">Δ SOTA</th>
                </tr>
              </thead>
              <tbody>
                {quantitative_results.results.map((r, i) => (
                  <tr key={i} className="border-b last:border-0">
                    <td className="py-2 pr-3">
                      {r.task}
                      <Cite tags={r.sources} />
                    </td>
                    <td className="py-2 pr-3">{r.metric_name}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">
                      {r.baseline_score}
                      {r.baseline_label ? ` (${r.baseline_label})` : ""}
                    </td>
                    <td className="py-2 pr-3 text-right font-semibold tabular-nums">
                      {r.paper_score}
                    </td>
                    <td
                      className={`py-2 pr-3 text-right font-semibold tabular-nums ${
                        r.sota_delta > 0
                          ? "text-emerald-600"
                          : r.sota_delta < 0
                            ? "text-rose-600"
                            : ""
                      }`}
                    >
                      {r.sota_delta > 0 ? "+" : ""}
                      {r.sota_delta}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
