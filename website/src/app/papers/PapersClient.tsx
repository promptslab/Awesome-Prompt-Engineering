"use client";

import { useState, useMemo } from "react";
import { Paper } from "@/lib/types";
import { PaperCard } from "@/components/ResourceCard";
import { FilterBar } from "@/components/FilterBar";

export function PapersClient({
  papers,
  subcategories,
}: {
  papers: Paper[];
  subcategories: string[];
}) {
  const [selected, setSelected] = useState("All");
  const [yearFilter, setYearFilter] = useState("All");

  const years = useMemo(() => {
    const set = new Set(papers.map((p) => p.year).filter(Boolean) as string[]);
    return Array.from(set).sort().reverse();
  }, [papers]);

  const filtered = useMemo(() => {
    return papers.filter((p) => {
      if (selected !== "All" && p.subcategory !== selected) return false;
      if (yearFilter !== "All" && p.year !== yearFilter) return false;
      return true;
    });
  }, [papers, selected, yearFilter]);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-4 py-4">
        <FilterBar
          categories={subcategories}
          selected={selected}
          onSelect={setSelected}
        />
      </div>
      <div className="flex flex-wrap gap-2 pb-4">
        <span className="self-center text-sm text-slate-500 dark:text-slate-400">Year:</span>
        <button
          onClick={() => setYearFilter("All")}
          className={`rounded-full px-2.5 py-1 text-xs transition-colors ${
            yearFilter === "All"
              ? "bg-indigo-600 text-white shadow-sm shadow-indigo-500/25 dark:bg-indigo-500"
              : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
          }`}
        >
          All
        </button>
        {years.map((y) => (
          <button
            key={y}
            onClick={() => setYearFilter(y)}
            className={`rounded-full px-2.5 py-1 text-xs transition-colors ${
              yearFilter === y
                ? "bg-indigo-600 text-white shadow-sm shadow-indigo-500/25 dark:bg-indigo-500"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
            }`}
          >
            {y}
          </button>
        ))}
      </div>
      <p className="pb-4 text-sm text-slate-500 dark:text-slate-400">
        Showing {filtered.length} of {papers.length} papers
      </p>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {filtered.map((paper, i) => (
          <PaperCard key={i} paper={paper} />
        ))}
      </div>
    </div>
  );
}
