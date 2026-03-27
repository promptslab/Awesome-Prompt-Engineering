import { fetchSiteData, fetchStarCount } from "@/lib/github";
import { buildSearchIndex } from "@/lib/parser";
import { ClientShell } from "@/components/ClientShell";
import { HeroSection } from "@/components/HeroSection";
import { StartHerePath } from "@/components/StartHerePath";
import { SectionGrid } from "@/components/SectionGrid";
import Link from "next/link";

export const revalidate = 300;

export default async function HomePage() {
  const [data, stars] = await Promise.all([fetchSiteData(), fetchStarCount()]);
  const searchItems = buildSearchIndex(data);

  const apiModelCount = data.apis.reduce((acc, a) => acc + a.models.length, 0);
  const modelCount = data.models.reduce((acc, m) => acc + m.models.length, 0);

  const sections = [
    {
      title: "Papers",
      description: "Research papers on prompt engineering techniques",
      href: "/papers",
      count: data.papers.length,
      icon: "📄",
    },
    {
      title: "Tools",
      description: "Frameworks, libraries, and platforms",
      href: "/tools",
      count: data.tools.length,
      icon: "🔧",
    },
    {
      title: "Models & APIs",
      description: "Frontier models and API providers",
      href: "/models",
      count: modelCount + apiModelCount,
      icon: "🧠",
    },
    {
      title: "Datasets",
      description: "Benchmarks, datasets, and evaluations",
      href: "/datasets",
      count: data.datasets.length,
      icon: "💾",
    },
    {
      title: "Learn",
      description: "Books, courses, tutorials, and videos",
      href: "/learn",
      count:
        data.books.length +
        data.courses.length +
        data.tutorials.length +
        data.videos.length,
      icon: "📚",
    },
    {
      title: "Community",
      description: "Discord, Reddit, forums, and GitHub orgs",
      href: "/community",
      count: data.communities.length,
      icon: "🤝",
    },
    {
      title: "AI Detectors",
      description: "AI content detection tools and watermarking",
      href: "/datasets",
      count: data.detectors.length,
      icon: "🔎",
    },
  ];

  return (
    <ClientShell searchItems={searchItems}>
      <HeroSection stars={stars} />

      <section className="py-8">
        <h2 className="mb-6 text-2xl font-bold text-slate-900 dark:text-white">
          Start Here
        </h2>
        <StartHerePath steps={data.startHere} />
      </section>

      <section className="py-8">
        <h2 className="mb-6 text-2xl font-bold text-slate-900 dark:text-white">
          Explore Resources
        </h2>
        <SectionGrid sections={sections} />
      </section>

      {/* About PromptsLab */}
      <section className="py-10">
        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white dark:border-slate-700/50 dark:bg-slate-800/50">
          <div className="border-b border-slate-200 bg-slate-50 px-8 py-5 dark:border-slate-700/50 dark:bg-slate-800/30">
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
              About PromptsLab
            </h2>
          </div>
          <div className="p-8">
            <div className="flex flex-col gap-8 md:flex-row md:items-start">
              <div className="shrink-0">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="https://avatars.githubusercontent.com/u/120981762?s=400&u=0f99838e8b43f32da445e3ed913681b481416d8a&v=4"
                  alt="PromptsLab"
                  className="h-20 w-20 rounded-full shadow-lg shadow-indigo-500/10 ring-2 ring-indigo-500/20 dark:ring-indigo-400/30"
                />
              </div>
              <div className="space-y-4">
                <p className="text-lg text-slate-600 dark:text-slate-300">
                  From the team behind{" "}
                  <span className="font-semibold text-indigo-600 dark:text-indigo-400">
                    Promptify (4.6k stars)
                  </span>,{" "}
                  <span className="font-semibold text-indigo-600 dark:text-indigo-400">
                    LLMtuner (200+ stars)
                  </span>, and{" "}
                  <span className="font-semibold text-indigo-600 dark:text-indigo-400">
                    Awesome-Prompt-Engineering (5.4k+ stars)
                  </span>.
                  We develop open-source tools and libraries that support
                  developers in creating robust pipelines for using LLMs APIs
                  in production.
                </p>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  <a href="https://github.com/promptslab/Promptify" target="_blank" rel="noopener noreferrer" className="rounded-xl border border-slate-200 bg-slate-50 p-4 transition-all hover:border-indigo-300 dark:border-slate-700/50 dark:bg-slate-800/40 dark:hover:border-indigo-500/40">
                    <h4 className="font-semibold text-slate-900 dark:text-white">Promptify</h4>
                    <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">NLP pipeline with LLMs</p>
                    <span className="mt-2 inline-block text-xs font-medium text-amber-600 dark:text-amber-400">4.6k stars</span>
                  </a>
                  <a href="https://github.com/promptslab/Awesome-Prompt-Engineering" target="_blank" rel="noopener noreferrer" className="rounded-xl border border-slate-200 bg-slate-50 p-4 transition-all hover:border-indigo-300 dark:border-slate-700/50 dark:bg-slate-800/40 dark:hover:border-indigo-500/40">
                    <h4 className="font-semibold text-slate-900 dark:text-white">Awesome Prompt Engineering</h4>
                    <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Curated PE resource collection</p>
                    <span className="mt-2 inline-block text-xs font-medium text-amber-600 dark:text-amber-400">5.4k+ stars</span>
                  </a>
                  <a href="https://github.com/promptslab/LLMtuner" target="_blank" rel="noopener noreferrer" className="rounded-xl border border-slate-200 bg-slate-50 p-4 transition-all hover:border-indigo-300 dark:border-slate-700/50 dark:bg-slate-800/40 dark:hover:border-indigo-500/40">
                    <h4 className="font-semibold text-slate-900 dark:text-white">LLMtuner</h4>
                    <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Fine-tune LLMs easily</p>
                    <span className="mt-2 inline-block text-xs font-medium text-amber-600 dark:text-amber-400">200+ stars</span>
                  </a>
                </div>
                <div className="flex gap-3 pt-2">
                  <a
                    href="https://github.com/promptslab"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition-all hover:border-indigo-300 hover:text-indigo-600 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-300 dark:hover:border-indigo-500/40 dark:hover:text-indigo-400"
                  >
                    <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                    </svg>
                    GitHub
                  </a>
                  <a
                    href="https://promptslab.github.io"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 rounded-full bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-all hover:bg-indigo-500 dark:bg-indigo-500 dark:hover:bg-indigo-400"
                  >
                    Learn More
                    <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Bottom Course CTA */}
      <section className="py-10">
        <Link
          href="/course"
          className="group relative block overflow-hidden rounded-2xl border border-indigo-200 bg-gradient-to-br from-indigo-50 via-violet-50 to-purple-50 p-10 text-center transition-all hover:shadow-xl dark:border-indigo-500/20 dark:from-indigo-950/40 dark:via-violet-950/30 dark:to-purple-950/20"
        >
          <div className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full bg-gradient-to-br from-indigo-400/20 to-violet-400/20 blur-3xl dark:from-indigo-400/10 dark:to-violet-400/10" />
          <div className="pointer-events-none absolute -bottom-16 -left-16 h-48 w-48 rounded-full bg-gradient-to-br from-violet-400/20 to-pink-400/15 blur-3xl dark:from-violet-400/10 dark:to-pink-400/5" />

          <div className="relative mx-auto max-w-2xl">
            <span className="inline-block rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 px-3 py-1 text-xs font-bold uppercase tracking-wider text-white shadow-sm">
              Coming Soon
            </span>
            <h2 className="mt-4 text-3xl font-bold text-slate-900 dark:text-white">
              Stop Guessing. Start Engineering.
            </h2>
            <p className="mt-3 text-lg text-slate-600 dark:text-slate-300">
              Our upcoming course takes you from writing basic prompts to building
              production-grade LLM systems with confidence. Built by the team behind
              10K+ GitHub stars in the prompt engineering space.
            </p>
            <div className="mt-6 flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
              <span className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-violet-600 via-indigo-600 to-violet-600 px-8 py-3 text-base font-semibold text-white shadow-lg shadow-indigo-500/25 transition-all group-hover:shadow-xl group-hover:shadow-indigo-500/30">
                Reserve Your Spot for Free
                <svg className="h-4 w-4 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </span>
              <span className="text-sm text-slate-500 dark:text-slate-400">
                No credit card required
              </span>
            </div>
          </div>
        </Link>
      </section>
    </ClientShell>
  );
}
