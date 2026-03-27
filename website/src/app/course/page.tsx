import { fetchStarCount } from "@/lib/github";
import { ClientShell } from "@/components/ClientShell";
import { EmailCapture } from "@/components/EmailCapture";
import type { Metadata } from "next";

export const revalidate = 300;

export const metadata: Metadata = {
  title: "Prompt Engineering Course",
  description:
    "A comprehensive prompt engineering course from PromptsLab — coming soon.",
};

const outcomes = [
  "Write production-grade prompts that work reliably at scale",
  "Build multi-step agent systems with tool use and guardrails",
  "Evaluate and optimize prompts systematically using DSPy and TextGrad",
  "Defend against prompt injection and implement security best practices",
  "Design context engineering pipelines for real-world applications",
];

const faqs = [
  {
    q: "When will the course launch?",
    a: "We're targeting Q2 2026. Sign up above to get notified the moment it's available.",
  },
  {
    q: "Is it free?",
    a: "We're planning a mix of free foundational content and premium advanced modules. Early subscribers will get a discount.",
  },
  {
    q: "What makes this different from existing courses?",
    a: "Built by the team behind the 5K+ star Awesome Prompt Engineering repo. We cover the full spectrum from basics to production — including context engineering, agent design, and security.",
  },
  {
    q: "What prerequisites do I need?",
    a: "Basic programming knowledge and familiarity with using LLMs (ChatGPT, Claude, etc.). No ML background required.",
  },
];

export default async function CoursePage() {
  const stars = await fetchStarCount();

  return (
    <ClientShell searchItems={[]} hideCourseBar>
      {/* Hero with gradient background */}
      <section className="relative py-20 text-center">
        {/* Background glows */}
        <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
          <div className="absolute left-1/4 top-0 h-[500px] w-[500px] rounded-full bg-gradient-to-br from-indigo-500/15 to-violet-500/10 blur-3xl dark:from-indigo-500/20 dark:to-violet-500/10" />
          <div className="absolute right-1/4 top-20 h-[400px] w-[400px] rounded-full bg-gradient-to-br from-violet-500/10 to-pink-500/10 blur-3xl dark:from-violet-500/15 dark:to-pink-500/5" />
        </div>

        <div className="mx-auto max-w-2xl">
          <span className="inline-block rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 px-4 py-1.5 text-sm font-bold uppercase tracking-wider text-white shadow-lg shadow-indigo-500/25">
            Coming Soon
          </span>
          <h1 className="mt-6 bg-gradient-to-r from-indigo-600 via-violet-600 to-indigo-600 bg-clip-text text-4xl font-bold text-transparent dark:from-indigo-400 dark:via-violet-300 dark:to-indigo-400 md:text-5xl lg:text-6xl">
            Prompt Engineering Course
          </h1>
          <p className="mt-5 text-lg text-slate-600 dark:text-slate-300">
            From the team behind the{" "}
            <span className="font-semibold text-indigo-600 dark:text-indigo-400">
              {stars.toLocaleString()}+ star
            </span>{" "}
            Awesome Prompt Engineering collection at PromptsLab.
          </p>

          <div className="mt-8 flex items-center justify-center gap-3">
            <div className="flex items-center gap-1.5 rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1.5 text-sm dark:border-indigo-500/20 dark:bg-indigo-500/10">
              <svg className="h-4 w-4 text-indigo-500" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              <span className="font-semibold text-indigo-700 dark:text-indigo-300">
                {stars.toLocaleString()}+ GitHub stars
              </span>
            </div>
            <div className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-600 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-400">
              394+ curated resources
            </div>
          </div>
        </div>
      </section>

      {/* Gap section */}
      <section className="py-8">
        <div className="mx-auto max-w-2xl rounded-2xl border border-indigo-200/50 bg-gradient-to-br from-indigo-50 to-violet-50 p-8 dark:border-indigo-500/15 dark:from-indigo-950/30 dark:to-violet-950/20">
          <h2 className="text-xl font-bold text-slate-900 dark:text-white">
            You&apos;ve read the guides, watched the videos, bookmarked the papers...
          </h2>
          <p className="mt-3 text-slate-600 dark:text-slate-300">
            But when it&apos;s time to build production systems that use LLMs
            reliably, you hit walls. Prompts that work in the playground break in
            production. Agents that demo well fail on edge cases. Security feels
            like an afterthought.
          </p>
          <p className="mt-3 font-semibold text-indigo-700 dark:text-indigo-300">
            This course bridges the gap between knowing about prompt engineering
            and doing it professionally.
          </p>
        </div>
      </section>

      {/* Outcomes */}
      <section className="py-10">
        <div className="mx-auto max-w-2xl">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
            What you&apos;ll be able to do
          </h2>
          <ul className="mt-6 space-y-4">
            {outcomes.map((outcome, i) => (
              <li key={i} className="flex gap-3 rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-700/50 dark:bg-slate-800/50">
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 text-white shadow-sm shadow-emerald-500/25">
                  <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <span className="text-slate-700 dark:text-slate-200">
                  {outcome}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* Email Capture */}
      <section className="py-14">
        <div className="relative mx-auto max-w-2xl overflow-hidden rounded-2xl border border-indigo-200/50 bg-gradient-to-br from-indigo-50 via-violet-50 to-indigo-50 p-10 text-center dark:border-indigo-500/15 dark:from-indigo-950/30 dark:via-violet-950/20 dark:to-indigo-950/30">
          {/* Decorative glow */}
          <div className="pointer-events-none absolute -right-20 -top-20 h-60 w-60 rounded-full bg-gradient-to-br from-indigo-400/20 to-transparent blur-3xl dark:from-indigo-400/10" />
          <div className="pointer-events-none absolute -bottom-20 -left-20 h-60 w-60 rounded-full bg-gradient-to-br from-violet-400/20 to-transparent blur-3xl dark:from-violet-400/10" />

          <div className="relative">
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
              Get notified when we launch
            </h2>
            <p className="mt-2 text-slate-600 dark:text-slate-300">
              Be the first to know. Early subscribers get priority access and a
              discount.
            </p>
            <div className="mt-8">
              <EmailCapture />
            </div>
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section className="py-8">
        <div className="mx-auto max-w-2xl">
          <h2 className="mb-6 text-2xl font-bold text-slate-900 dark:text-white">
            FAQ
          </h2>
          <div className="space-y-3">
            {faqs.map((faq, i) => (
              <details
                key={i}
                className="group rounded-xl border border-slate-200 bg-white dark:border-slate-700/50 dark:bg-slate-800/50"
              >
                <summary className="cursor-pointer px-5 py-4 font-medium text-slate-800 dark:text-slate-200">
                  {faq.q}
                </summary>
                <p className="px-5 pb-4 text-sm text-slate-500 dark:text-slate-400">
                  {faq.a}
                </p>
              </details>
            ))}
          </div>
        </div>
      </section>
    </ClientShell>
  );
}
