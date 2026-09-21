"use client";

import { FileUp, Loader2 } from "lucide-react";
import { useRef, useState } from "react";
import { uploadPaper, type ParsedPaper } from "@/lib/api";

export default function UploadPanel({
  onUploaded,
  onError,
}: {
  onUploaded: (paper: ParsedPaper) => void;
  onError: (msg: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      onError("Only .pdf files are accepted.");
      return;
    }
    setBusy(true);
    try {
      const paper = await uploadPaper(file);
      onUploaded(paper);
    } catch (e) {
      onError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        void handleFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      className={`cursor-pointer rounded-2xl border-2 border-dashed p-5 text-center transition ${
        dragOver
          ? "border-indigo-500 bg-indigo-50"
          : "border-slate-300 bg-white hover:border-indigo-400"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,application/pdf"
        className="hidden"
        onChange={(e) => void handleFiles(e.target.files)}
      />
      <div className="flex flex-col items-center gap-2">
        {busy ? (
          <Loader2 className="h-6 w-6 animate-spin text-indigo-600" />
        ) : (
          <FileUp className="h-6 w-6 text-indigo-600" />
        )}
        <p className="text-sm font-medium">
          {busy ? "Parsing PDF…" : "Drop a paper PDF here or click to upload"}
        </p>
        <p className="text-xs text-slate-500">
          Sections, math, and metadata are extracted automatically.
        </p>
      </div>
    </div>
  );
}
