"use client";

export function FilterBar({
  categories,
  selected,
  onSelect,
}: {
  categories: string[];
  selected: string;
  onSelect: (cat: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      <button
        onClick={() => onSelect("All")}
        className={`rounded-full px-3.5 py-1.5 text-sm font-medium transition-all ${
          selected === "All"
            ? "bg-indigo-600 text-white shadow-sm shadow-indigo-500/25 dark:bg-indigo-500 dark:shadow-indigo-500/20"
            : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
        }`}
      >
        All
      </button>
      {categories.map((cat) => (
        <button
          key={cat}
          onClick={() => onSelect(cat)}
          className={`rounded-full px-3.5 py-1.5 text-sm font-medium transition-all ${
            selected === cat
              ? "bg-indigo-600 text-white shadow-sm shadow-indigo-500/25 dark:bg-indigo-500 dark:shadow-indigo-500/20"
              : "bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
          }`}
        >
          {cat}
        </button>
      ))}
    </div>
  );
}
