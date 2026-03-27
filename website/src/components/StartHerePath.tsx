import { StartHereStep } from "@/lib/types";

export function StartHerePath({ steps }: { steps: StartHereStep[] }) {
  return (
    <div className="relative">
      {/* Vertical connector line (mobile) */}
      <div className="absolute left-5 top-8 bottom-8 w-px bg-gradient-to-b from-indigo-500 via-violet-500 to-emerald-500 md:hidden" />

      <div className="grid gap-4 md:grid-cols-5">
        {steps.map((step, i) => (
          <a
            key={step.step}
            href={step.url}
            target="_blank"
            rel="noopener noreferrer"
            className="group relative flex gap-4 rounded-xl border border-slate-200 bg-white p-4 transition-all hover:border-indigo-300 hover:shadow-md dark:border-slate-700/50 dark:bg-slate-800/50 dark:hover:border-indigo-500/40 dark:hover:shadow-indigo-500/10 md:flex-col md:gap-2"
          >
            <div className="relative z-10 flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 text-sm font-bold text-white shadow-lg shadow-indigo-500/25">
              {step.step}
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
                {step.label}
              </div>
              <div className="mt-1 text-sm font-medium text-slate-800 group-hover:text-indigo-600 dark:text-slate-200 dark:group-hover:text-indigo-400">
                {step.title}
              </div>
            </div>
            {i < steps.length - 1 && (
              <div className="absolute -right-4 top-1/2 hidden -translate-y-1/2 text-slate-300 dark:text-slate-600 md:block">
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            )}
          </a>
        ))}
      </div>
    </div>
  );
}
