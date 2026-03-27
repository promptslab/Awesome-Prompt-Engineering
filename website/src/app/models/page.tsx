import { fetchSiteData } from "@/lib/github";
import { buildSearchIndex } from "@/lib/parser";
import { ClientShell } from "@/components/ClientShell";
import { ModelsClient } from "./ModelsClient";
import type { Metadata } from "next";

export const revalidate = 300;

export const metadata: Metadata = {
  title: "Models & APIs",
  description: "Frontier models, reasoning models, and API providers for LLMs.",
};

export default async function ModelsPage() {
  const data = await fetchSiteData();
  const searchItems = buildSearchIndex(data);

  return (
    <ClientShell searchItems={searchItems}>
      <div className="py-6">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
          Models & APIs
        </h1>
        <p className="mt-2 text-slate-500 dark:text-slate-400">
          Frontier models, reasoning models, and API providers.
        </p>
      </div>
      <ModelsClient models={data.models} apis={data.apis} />
    </ClientShell>
  );
}
