import { fetchSiteData } from "@/lib/github";
import { buildSearchIndex } from "@/lib/parser";
import { ClientShell } from "@/components/ClientShell";
import { PapersClient } from "./PapersClient";
import type { Metadata } from "next";

export const revalidate = 300;

export const metadata: Metadata = {
  title: "Papers",
  description: "Research papers on prompt engineering, reasoning, and LLM techniques.",
};

export default async function PapersPage() {
  const data = await fetchSiteData();
  const searchItems = buildSearchIndex(data);

  return (
    <ClientShell searchItems={searchItems}>
      <div className="py-6">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
          Papers
        </h1>
        <p className="mt-2 text-slate-500 dark:text-slate-400">
          {data.papers.length} research papers on prompt engineering techniques.
        </p>
      </div>
      <PapersClient
        papers={data.papers}
        subcategories={data.paperSubcategories}
      />
    </ClientShell>
  );
}
