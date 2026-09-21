"use client";

import katex from "katex";
import { useMemo } from "react";

const MATH_RE =
  /(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$|\\\[[\s\S]+?\\\]|\\\(.+?\\\))/g;

function stripDelims(raw: string): { math: string; block: boolean } {
  if (raw.startsWith("$$")) return { math: raw.slice(2, -2), block: true };
  if (raw.startsWith("$")) return { math: raw.slice(1, -1), block: false };
  if (raw.startsWith("\\[")) return { math: raw.slice(2, -2), block: true };
  return { math: raw.slice(2, -2), block: false };
}

function Tex({ math, block }: { math: string; block: boolean }) {
  let html: string;
  try {
    html = katex.renderToString(math, {
      displayMode: block,
      throwOnError: false,
    });
  } catch {
    return <code className="rounded bg-slate-100 px-1 text-sm">{math}</code>;
  }
  if (block) {
    return (
      <div
        className="overflow-x-auto py-1"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Render running text with $…$ / $$…$$ LaTeX rendered via KaTeX. */
export default function MathText({ text }: { text: string }) {
  const parts = useMemo(() => {
    const out: { key: number; node: React.ReactNode }[] = [];
    let last = 0;
    let m: RegExpExecArray | null;
    MATH_RE.lastIndex = 0;
    let key = 0;
    while ((m = MATH_RE.exec(text)) !== null) {
      if (m.index > last) {
        out.push({
          key: key++,
          node: <span key={key}>{text.slice(last, m.index)}</span>,
        });
      }
      const { math, block } = stripDelims(m[0]);
      out.push({ key: key++, node: <Tex key={key} math={math} block={block} /> });
      last = m.index + m[0].length;
    }
    if (last < text.length) {
      out.push({ key: key++, node: <span key={key}>{text.slice(last)}</span> });
    }
    return out;
  }, [text]);

  return (
    <>
      {parts.map((p) => (
        <span key={p.key}>{p.node}</span>
      ))}
    </>
  );
}
