"use client";

import type { SourceTag } from "@/lib/api";

function tagLabel(t: SourceTag): string {
  return `${t.paragraph_id} · p.${t.page_number}`;
}

/** Inline citation badges linking a claim back to its PDF location. */
export function Cite({ tags }: { tags: SourceTag[] }) {
  if (!tags || tags.length === 0) return null;
  return (
    <span className="ml-1 inline-flex flex-wrap gap-1 align-middle">
      {tags.slice(0, 4).map((t, i) => (
        <span
          key={`${t.paragraph_id}-${t.page_number}-${i}`}
          title={`${t.section} — ${t.paragraph_id}, page ${t.page_number}`}
          className="inline-flex items-center rounded-full bg-indigo-50 px-1.5 py-px text-[10px] font-medium text-indigo-700 ring-1 ring-inset ring-indigo-200"
        >
          {tagLabel(t)}
        </span>
      ))}
      {tags.length > 4 && (
        <span className="text-[10px] text-slate-400">+{tags.length - 4}</span>
      )}
    </span>
  );
}
