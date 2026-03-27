import { fetchSiteData } from "@/lib/github";
import { buildSearchIndex } from "@/lib/parser";
import { ClientShell } from "@/components/ClientShell";
import { ToolsClient } from "./ToolsClient";
import type { Metadata } from "next";

export const revalidate = 300;

export const metadata: Metadata = {
  title: "Tools",
  description: "Prompt management, evaluation, agent frameworks, and developer tools.",
};

export default async function ToolsPage() {
  const data = await fetchSiteData();
  const searchItems = buildSearchIndex(data);

  return (
    <ClientShell searchItems={searchItems}>
      <div className="py-6">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
          Tools & Code
        </h1>
        <p className="mt-2 text-slate-500 dark:text-slate-400">
          {data.tools.length} tools for prompt management, evaluation, and development.
        </p>
      </div>
      <ToolsClient
        tools={data.tools}
        subcategories={data.toolSubcategories}
      />
    </ClientShell>
  );
}
