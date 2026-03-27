import { Resource } from "@/lib/types";

export function ResourceTable({ resources }: { resources: Resource[] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700/50">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50 dark:border-slate-700/50 dark:bg-slate-800/50">
            <th className="px-4 py-3 text-left font-medium text-slate-500 dark:text-slate-400">Name</th>
            <th className="px-4 py-3 text-left font-medium text-slate-500 dark:text-slate-400">Description</th>
            <th className="px-4 py-3 text-left font-medium text-slate-500 dark:text-slate-400">Category</th>
            <th className="px-4 py-3 text-left font-medium text-slate-500 dark:text-slate-400">Link</th>
          </tr>
        </thead>
        <tbody>
          {resources.map((resource, i) => (
            <tr
              key={i}
              className="border-b border-slate-100 transition-colors hover:bg-slate-50 dark:border-slate-700/30 dark:hover:bg-slate-800/40"
            >
              <td className="px-4 py-3 font-medium text-slate-900 dark:text-slate-100">
                {resource.name}
                {resource.stars && (
                  <span className="ml-2 text-xs font-medium text-amber-600 dark:text-amber-400">
                    {resource.stars} stars
                  </span>
                )}
              </td>
              <td className="max-w-md px-4 py-3 text-slate-500 dark:text-slate-400">
                {resource.description}
              </td>
              <td className="px-4 py-3">
                <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-600 dark:bg-slate-700/50 dark:text-slate-400">
                  {resource.subcategory}
                </span>
              </td>
              <td className="px-4 py-3">
                {resource.url && (
                  <a
                    href={resource.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-medium text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300"
                  >
                    View
                  </a>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
