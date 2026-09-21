/* ScholarSense AI — typed backend client + shared DTOs. */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/* -- ingestion DTOs ------------------------------------------------------ */

export interface SourceTag {
  page_number: number;
  paragraph_id: string;
  section: string;
}

export interface ParsedParagraph {
  paragraph_id: string;
  page_number: number;
  text: string;
}

export interface PaperSection {
  name: string;
  text: string;
  paragraphs: ParsedParagraph[];
}

export interface PaperMetadata {
  title: string | null;
  authors: string[];
  year: number | null;
  venue: string | null;
  abstract: string | null;
  doi: string | null;
  url: string | null;
  enrichment_source: string;
}

export interface ParsedPaper {
  paper_id: string;
  filename: string;
  metadata: PaperMetadata;
  sections: PaperSection[];
  page_count: number;
  paragraph_count: number;
}

export interface PaperListItem {
  paper_id: string;
  filename: string;
  title: string | null;
  year: number | null;
  page_count: number;
}

/* -- extraction DTOs ----------------------------------------------------- */

export interface SummaryBullet {
  text: string;
  sources: SourceTag[];
}

export interface SectionalEntry {
  section: string;
  summary: string;
  sources: SourceTag[];
}

export interface HierarchicalSummary {
  executive_summary: SummaryBullet[];
  sectional_summary: SectionalEntry[];
  technical_tldr: string;
  technical_tldr_sources: SourceTag[];
}

export interface Hyperparameter {
  name: string;
  value: string;
  sources: SourceTag[];
}

export interface LossFunction {
  name: string;
  expression_latex: string;
  description: string;
  sources: SourceTag[];
}

export interface DatasetInfo {
  name: string;
  details: string;
  sources: SourceTag[];
}

export interface Methodology {
  architecture: string;
  architecture_sources: SourceTag[];
  hyperparameters: Hyperparameter[];
  loss_functions: LossFunction[];
  datasets: DatasetInfo[];
  hardware_requirements: string;
  hardware_sources: SourceTag[];
}

export interface BenchmarkResult {
  task: string;
  metric_name: string;
  baseline_score: number;
  baseline_label: string;
  paper_score: number;
  sota_delta: number;
  higher_is_better: boolean;
  score_context: string;
  sources: SourceTag[];
}

export interface PaperExtraction {
  paper_id: string;
  summary: HierarchicalSummary;
  methodology: Methodology;
  quantitative_results: { results: BenchmarkResult[] };
  extraction_source: "openai-structured" | "heuristic-fallback";
}

/* -- synthesis DTOs ------------------------------------------------------ */

export interface ComparisonCell {
  paper_id: string;
  value: string;
  sources: SourceTag[];
}

export interface ComparisonRow {
  dimension: string;
  kind: string;
  cells: ComparisonCell[];
  verdict: string;
  note: string;
}

export interface Finding {
  statement: string;
  paper_ids: string[];
  sources: SourceTag[];
}

export interface PaperComparison {
  paper_ids: string[];
  papers: { paper_id: string; title: string; year: number | null }[];
  rows: ComparisonRow[];
  consensus_points: Finding[];
  contradictions: Finding[];
  generated_by: string;
}

export interface GapItem {
  statement: string;
  category: string;
  paper_ids: string[];
  sources: SourceTag[];
}

export interface ResearchGaps {
  paper_ids: string[];
  failure_modes: GapItem[];
  dataset_shortfalls: GapItem[];
  methodological_blinds: GapItem[];
  cross_cutting: GapItem[];
  generated_by: string;
}

/* -- HTTP helpers -------------------------------------------------------- */

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status} ${path}: ${detail}`);
  }
  return (await res.json()) as T;
}

export function listPapers(): Promise<{ count: number; papers: PaperListItem[] }> {
  return request("/api/v1/papers");
}

export function getPaper(paperId: string): Promise<ParsedPaper> {
  return request(`/api/v1/papers/${paperId}`);
}

export async function uploadPaper(file: File): Promise<ParsedPaper> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/api/v1/papers/upload?enrich=false`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`Upload failed (${res.status}): ${detail}`);
  }
  const data = (await res.json()) as { paper: ParsedPaper };
  return data.paper;
}

export function extractPaper(
  paperId: string,
  prefer: "auto" | "openai" | "heuristic" = "auto",
): Promise<PaperExtraction> {
  return request(`/api/v1/papers/${paperId}/extract?prefer=${prefer}`, {
    method: "POST",
  });
}

export function getExtraction(paperId: string): Promise<PaperExtraction> {
  return request(`/api/v1/papers/${paperId}/extraction`);
}

export function comparePapers(paperIds: string[]): Promise<PaperComparison> {
  return request("/api/v1/papers/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paper_ids: paperIds }),
  });
}

export function fetchGaps(paperIds: string[]): Promise<ResearchGaps> {
  return request("/api/v1/papers/gaps", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paper_ids: paperIds }),
  });
}

export function shortTitle(title: string | null, fallback: string): string {
  if (title && title.trim()) return title;
  return fallback;
}
