"use client";

import { ScanSearch } from "lucide-react";
import type { GapItem, ResearchGaps } from "@/lib/api";
import { Cite } from "./Citation";
import MathText from "./MathText";

function Bucket({
  title,
  items,
  accent,
}: {
  title: string;
  items: GapItem[];
  accent: string;
}) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5">
      <h4 className={`text-sm font-semibold ${accent}`}>
        {title} ({items.length})
      </h4>
      {items.length === 0 ? (
        <p className="mt-2 text-sm italic text-slate-400">
          No gaps identified in this category.
        </p>
      ) : (
        <ul className="mt-2 space-y-2 text-sm">
          {items.map((g, i) => (
            <li key={i} className="rounded-xl bg-slate-50 p-2.5">
              <MathText text={g.statement} />
              <Cite tags={g.sources} />
              <span className="ml-2 text-[11px] text-slate-400">
                {g.paper_ids.length} paper{g.paper_ids.length === 1 ? "" : "s"}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default function GapsTab({ gaps }: { gaps: ResearchGaps }) {
  return (
    <div className="space-y-4">
      <section className="rounded-2xl border border-violet-200 bg-violet-50 p-5">
        <h3 className="flex items-center gap-2 text-base font-semibold text-violet-900">
          <ScanSearch className="h-4 w-4" /> Cross-cutting themes (
          {gaps.cross_cutting.length})
        </h3>
        {gaps.cross_cutting.length === 0 ? (
          <p className="mt-1 text-sm text-violet-700">
            No themes recur across multiple papers yet.
          </p>
        ) : (
          <ul className="mt-2 space-y-2 text-sm text-violet-950">
            {gaps.cross_cutting.map((g, i) => (
              <li key={i} className="rounded-xl bg-white/70 p-2.5">
                <MathText text={g.statement} />
                <Cite tags={g.sources} />
              </li>
            ))}
          </ul>
        )}
      </section>

      <Bucket
        title="1. Unresolved benchmark failure modes"
        items={gaps.failure_modes}
        accent="text-rose-700"
      />
      <Bucket
        title="2. Dataset & evaluation shortfalls"
        items={gaps.dataset_shortfalls}
        accent="text-amber-700"
      />
      <Bucket
        title="3. Missing ablations & methodological blinds"
        items={gaps.methodological_blinds}
        accent="text-indigo-700"
      />
    </div>
  );
}
