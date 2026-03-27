import { fetchSiteData } from "@/lib/github";
import { buildSearchIndex } from "@/lib/parser";
import { ClientShell } from "@/components/ClientShell";
import { DatasetsClient } from "./DatasetsClient";
import type { Metadata } from "next";

export const revalidate = 300;

export const metadata: Metadata = {
  title: "Datasets & Benchmarks",
  description: "Benchmarks, datasets, and red teaming resources for LLM evaluation.",
};

export default async function DatasetsPage() {
  const data = await fetchSiteData();
  const searchItems = buildSearchIndex(data);

  return (
    <ClientShell searchItems={searchItems}>
      <div className="py-6">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
          Datasets & Benchmarks
        </h1>
        <p className="mt-2 text-slate-500 dark:text-slate-400">
          {data.datasets.length} datasets and benchmarks for LLM evaluation.
          {data.detectors.length > 0 && ` Plus ${data.detectors.length} AI content detectors.`}
        </p>
      </div>
      <DatasetsClient
        datasets={data.datasets}
        detectors={data.detectors}
        datasetSubcategories={data.datasetSubcategories}
        detectorSubcategories={data.detectorSubcategories}
      />
    </ClientShell>
  );
}
