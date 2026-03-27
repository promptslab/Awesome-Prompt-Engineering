"use client";

import { useEffect, useRef, useState } from "react";
import Fuse from "fuse.js";
import { SearchItem } from "@/lib/types";

const categoryColors: Record<string, string> = {
  Papers: "bg-violet-50 text-violet-700 dark:bg-violet-500/15 dark:text-violet-300",
  Tools: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300",
  Models: "bg-indigo-50 text-indigo-700 dark:bg-indigo-500/15 dark:text-indigo-300",
  APIs: "bg-orange-50 text-orange-700 dark:bg-orange-500/15 dark:text-orange-300",
  Datasets: "bg-amber-50 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300",
  Courses: "bg-pink-50 text-pink-700 dark:bg-pink-500/15 dark:text-pink-300",
  Tutorials: "bg-cyan-50 text-cyan-700 dark:bg-cyan-500/15 dark:text-cyan-300",
  Videos: "bg-red-50 text-red-700 dark:bg-red-500/15 dark:text-red-300",
  Books: "bg-indigo-50 text-indigo-700 dark:bg-indigo-500/15 dark:text-indigo-300",
  Communities: "bg-teal-50 text-teal-700 dark:bg-teal-500/15 dark:text-teal-300",
  Detectors: "bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300",
};

export function SearchDialog({
  items,
  open,
  onClose,
}: {
  items: SearchItem[];
  open: boolean;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const fuseRef = useRef<Fuse<SearchItem> | null>(null);

  useEffect(() => {
    if (items.length > 0) {
      fuseRef.current = new Fuse(items, {
        keys: [
          { name: "name", weight: 0.6 },
          { name: "description", weight: 0.3 },
          { name: "category", weight: 0.1 },
        ],
        threshold: 0.4,
        includeScore: true,
        minMatchCharLength: 2,
      });
    }
  }, [items]);

  useEffect(() => {
    if (open) {
      inputRef.current?.focus();
      setQuery("");
    }
  }, [open]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        if (open) onClose();
      }
      if (e.key === "Escape" && open) {
        onClose();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  const results = query.length >= 2 && fuseRef.current
    ? fuseRef.current.search(query).slice(0, 20)
    : [];

  const grouped = results.reduce(
    (acc, r) => {
      const cat = r.item.category;
      if (!acc[cat]) acc[cat] = [];
      acc[cat].push(r.item);
      return acc;
    },
    {} as Record<string, SearchItem[]>
  );

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]">
      <div className="fixed inset-0 bg-black/40 backdrop-blur-sm dark:bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-full max-w-lg rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700/50 dark:bg-[#1e293b] dark:shadow-indigo-500/5">
        <div className="flex items-center border-b border-slate-200 px-4 dark:border-slate-700/50">
          <svg className="mr-3 h-4 w-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search resources..."
            className="w-full bg-transparent py-3.5 text-sm text-slate-900 outline-none placeholder:text-slate-400 dark:text-slate-100 dark:placeholder:text-slate-500"
          />
          <kbd className="rounded-md bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] text-slate-400 dark:bg-slate-700 dark:text-slate-500">
            Esc
          </kbd>
        </div>

        <div className="max-h-[60vh] overflow-y-auto p-2">
          {query.length < 2 ? (
            <p className="px-3 py-8 text-center text-sm text-slate-400 dark:text-slate-500">
              Type at least 2 characters to search...
            </p>
          ) : results.length === 0 ? (
            <p className="px-3 py-8 text-center text-sm text-slate-400 dark:text-slate-500">
              No results found for &quot;{query}&quot;
            </p>
          ) : (
            Object.entries(grouped).map(([category, items]) => (
              <div key={category} className="mb-2">
                <div className="px-3 py-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                  {category}
                </div>
                {items.map((item, i) => (
                  <a
                    key={i}
                    href={item.url || "#"}
                    target={item.url ? "_blank" : undefined}
                    rel="noopener noreferrer"
                    onClick={onClose}
                    className="flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors hover:bg-indigo-50 dark:hover:bg-slate-700/50"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-slate-800 dark:text-slate-200">
                        {item.name}
                      </div>
                      {item.description && (
                        <div className="truncate text-xs text-slate-500 dark:text-slate-400">
                          {item.description}
                        </div>
                      )}
                    </div>
                    <span
                      className={`shrink-0 rounded-full px-2.5 py-0.5 text-[10px] font-semibold ${
                        categoryColors[category] || "bg-slate-100 text-slate-600 dark:bg-slate-700/50 dark:text-slate-400"
                      }`}
                    >
                      {item.subcategory}
                    </span>
                  </a>
                ))}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
