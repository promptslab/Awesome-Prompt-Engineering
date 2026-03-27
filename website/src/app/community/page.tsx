import { fetchSiteData } from "@/lib/github";
import { buildSearchIndex } from "@/lib/parser";
import { ClientShell } from "@/components/ClientShell";
import type { Metadata } from "next";

export const revalidate = 300;

export const metadata: Metadata = {
  title: "Community",
  description: "Discord servers, Reddit communities, forums, and GitHub organizations.",
};

export default async function CommunityPage() {
  const data = await fetchSiteData();
  const searchItems = buildSearchIndex(data);

  // Group communities by subcategory
  const grouped = data.communitySubcategories.map((sub) => ({
    name: sub,
    items: data.communities.filter((c) => c.subcategory === sub),
  }));

  return (
    <ClientShell searchItems={searchItems}>
      <div className="py-6">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
          Community
        </h1>
        <p className="mt-2 text-slate-500 dark:text-slate-400">
          {data.communities.length} communities for prompt engineering discussions.
        </p>
      </div>

      {/* PromptsLab Discord — highlighted */}
      <section className="mb-8">
        <a
          href="https://discord.gg/m88xfYMbK6"
          target="_blank"
          rel="noopener noreferrer"
          className="group relative block overflow-hidden rounded-2xl border-2 border-[#5865F2]/30 bg-gradient-to-r from-[#5865F2]/5 via-indigo-50 to-[#5865F2]/5 p-6 transition-all hover:border-[#5865F2]/50 hover:shadow-xl hover:shadow-[#5865F2]/10 dark:from-[#5865F2]/10 dark:via-indigo-950/30 dark:to-[#5865F2]/10"
        >
          <div className="pointer-events-none absolute -right-10 -top-10 h-32 w-32 rounded-full bg-[#5865F2]/10 blur-3xl" />
          <div className="relative flex flex-col items-center gap-4 sm:flex-row">
            <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-[#5865F2] shadow-lg shadow-[#5865F2]/25">
              <svg className="h-7 w-7 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.095 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.095 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" />
              </svg>
            </div>
            <div className="flex-1 text-center sm:text-left">
              <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                Join the PromptsLab Discord
              </h2>
              <p className="mt-1 text-slate-600 dark:text-slate-300">
                Connect with prompt engineers, share techniques, get help with your projects, and stay updated on our latest tools and research.
              </p>
            </div>
            <span className="inline-flex items-center gap-2 rounded-full bg-[#5865F2] px-6 py-2.5 text-sm font-semibold text-white shadow-lg shadow-[#5865F2]/25 transition-all group-hover:shadow-xl group-hover:shadow-[#5865F2]/30">
              Join Server
              <svg className="h-4 w-4 transition-transform group-hover:translate-x-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </span>
          </div>
        </a>
      </section>

      <div className="grid gap-8">
        {grouped.map((group) => (
          <section key={group.name}>
            <h2 className="mb-4 text-xl font-semibold text-slate-900 dark:text-white">
              {group.name}
            </h2>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {group.items.map((community, i) => (
                <a
                  key={i}
                  href={community.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group flex flex-col rounded-xl border border-slate-200 bg-white p-4 transition-all hover:border-indigo-300 hover:shadow-md dark:border-slate-700/50 dark:bg-slate-800/50 dark:hover:border-indigo-500/40 dark:hover:shadow-indigo-500/10"
                >
                  <h3 className="font-medium text-slate-900 group-hover:text-indigo-600 dark:text-slate-100 dark:group-hover:text-indigo-400">
                    {community.name}
                  </h3>
                  {community.description && (
                    <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                      {community.description}
                    </p>
                  )}
                </a>
              ))}
            </div>
          </section>
        ))}
      </div>
    </ClientShell>
  );
}
