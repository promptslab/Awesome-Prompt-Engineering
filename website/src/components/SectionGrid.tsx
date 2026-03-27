import Link from "next/link";

interface SectionCardData {
  title: string;
  description: string;
  href: string;
  count: number;
  icon: string;
}

export function SectionGrid({ sections }: { sections: SectionCardData[] }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {sections.map((section) => (
        <Link
          key={section.href}
          href={section.href}
          className="group flex flex-col rounded-xl border border-slate-200 bg-white p-5 transition-all hover:border-indigo-300 hover:shadow-md dark:border-slate-700/50 dark:bg-slate-800/50 dark:hover:border-indigo-500/40 dark:hover:shadow-indigo-500/10"
        >
          <span className="text-2xl">{section.icon}</span>
          <h3 className="mt-3 font-semibold text-slate-900 group-hover:text-indigo-600 dark:text-slate-100 dark:group-hover:text-indigo-400">
            {section.title}
          </h3>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            {section.description}
          </p>
          <span className="mt-3 text-sm font-semibold text-indigo-600/70 dark:text-indigo-400/70">
            {section.count} resources
          </span>
        </Link>
      ))}
    </div>
  );
}
