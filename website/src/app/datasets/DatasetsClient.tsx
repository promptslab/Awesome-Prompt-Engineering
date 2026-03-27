"use client";

import { useState, useMemo } from "react";
import { Resource } from "@/lib/types";
import { ResourceTable } from "@/components/ResourceTable";
import { FilterBar } from "@/components/FilterBar";

export function DatasetsClient({
  datasets,
  detectors,
  datasetSubcategories,
  detectorSubcategories,
}: {
  datasets: Resource[];
  detectors: Resource[];
  datasetSubcategories: string[];
  detectorSubcategories: string[];
}) {
  const [tab, setTab] = useState<"datasets" | "detectors">("datasets");
  const [selected, setSelected] = useState("All");

  const categories = tab === "datasets" ? datasetSubcategories : detectorSubcategories;
  const items = tab === "datasets" ? datasets : detectors;

  const filtered = useMemo(() => {
    if (selected === "All") return items;
    return items.filter((d) => d.subcategory === selected);
  }, [items, selected]);

  return (
    <div>
      <div className="flex gap-1 border-b border-slate-200 dark:border-slate-700/50">
        <button
          onClick={() => { setTab("datasets"); setSelected("All"); }}
          className={`px-4 py-2.5 text-sm font-medium transition-colors ${
            tab === "datasets"
              ? "border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400"
              : "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
          }`}
        >
          Datasets & Benchmarks ({datasets.length})
        </button>
        <button
          onClick={() => { setTab("detectors"); setSelected("All"); }}
          className={`px-4 py-2.5 text-sm font-medium transition-colors ${
            tab === "detectors"
              ? "border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400"
              : "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
          }`}
        >
          AI Detectors ({detectors.length})
        </button>
      </div>

      <div className="mt-4">
        <FilterBar
          categories={categories}
          selected={selected}
          onSelect={setSelected}
        />
      </div>
      <p className="py-4 text-sm text-slate-500 dark:text-slate-400">
        Showing {filtered.length} results
      </p>
      <ResourceTable resources={filtered} />
    </div>
  );
}
