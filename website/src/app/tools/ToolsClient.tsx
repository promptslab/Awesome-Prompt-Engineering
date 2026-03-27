"use client";

import { useState, useMemo } from "react";
import { Resource } from "@/lib/types";
import { ResourceCard } from "@/components/ResourceCard";
import { ResourceTable } from "@/components/ResourceTable";
import { FilterBar } from "@/components/FilterBar";
import { ViewToggle } from "@/components/ViewToggle";

export function ToolsClient({
  tools,
  subcategories,
}: {
  tools: Resource[];
  subcategories: string[];
}) {
  const [selected, setSelected] = useState("All");
  const [view, setView] = useState<"cards" | "table">("cards");

  const filtered = useMemo(() => {
    if (selected === "All") return tools;
    return tools.filter((t) => t.subcategory === selected);
  }, [tools, selected]);

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4 py-4">
        <FilterBar
          categories={subcategories}
          selected={selected}
          onSelect={setSelected}
        />
        <ViewToggle view={view} onToggle={setView} />
      </div>
      <p className="pb-4 text-sm text-slate-500 dark:text-slate-400">
        Showing {filtered.length} of {tools.length} tools
      </p>
      {view === "cards" ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((tool, i) => (
            <ResourceCard key={i} resource={tool} />
          ))}
        </div>
      ) : (
        <ResourceTable resources={filtered} />
      )}
    </div>
  );
}
