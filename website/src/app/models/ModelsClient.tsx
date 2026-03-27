"use client";

import { useState } from "react";
import { APIProvider } from "@/lib/types";

function ModelTable({ provider }: { provider: APIProvider }) {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700/50">
      <div className="border-b border-slate-200 bg-slate-50 px-5 py-3 dark:border-slate-700/50 dark:bg-slate-800/50">
        <h3 className="font-semibold text-slate-900 dark:text-white">
          {provider.provider}
        </h3>
        {provider.features && (
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            {provider.features}
          </p>
        )}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 dark:border-slate-700/30">
              <th className="px-5 py-2.5 text-left font-medium text-slate-500 dark:text-slate-400">
                Model
              </th>
              {provider.models[0]?.context && (
                <th className="px-5 py-2.5 text-left font-medium text-slate-500 dark:text-slate-400">
                  Context
                </th>
              )}
              {provider.models[0]?.price && (
                <th className="px-5 py-2.5 text-left font-medium text-slate-500 dark:text-slate-400">
                  Price
                </th>
              )}
              <th className="px-5 py-2.5 text-left font-medium text-slate-500 dark:text-slate-400">
                {provider.models[0]?.significance ? "Significance" : "Key Feature"}
              </th>
            </tr>
          </thead>
          <tbody>
            {provider.models.map((model, i) => (
              <tr
                key={i}
                className="border-b border-slate-50 last:border-0 transition-colors hover:bg-slate-50 dark:border-slate-700/20 dark:hover:bg-slate-800/40"
              >
                <td className="px-5 py-2.5 font-medium text-slate-900 dark:text-slate-100">
                  {model.name}
                </td>
                {model.context !== undefined && (
                  <td className="px-5 py-2.5 text-slate-500 dark:text-slate-400">
                    {model.context}
                  </td>
                )}
                {model.price !== undefined && (
                  <td className="px-5 py-2.5 font-mono text-xs text-slate-500 dark:text-slate-400">
                    {model.price}
                  </td>
                )}
                <td className="px-5 py-2.5 text-slate-500 dark:text-slate-400">
                  {model.keyFeature || model.significance}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {provider.url && (
        <div className="border-t border-slate-200 px-5 py-2.5 dark:border-slate-700/50">
          <a
            href={provider.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300"
          >
            View docs
          </a>
        </div>
      )}
    </div>
  );
}

export function ModelsClient({
  models,
  apis,
}: {
  models: APIProvider[];
  apis: APIProvider[];
}) {
  const [tab, setTab] = useState<"models" | "apis">("models");

  return (
    <div>
      <div className="flex gap-1 border-b border-slate-200 dark:border-slate-700/50">
        <button
          onClick={() => setTab("models")}
          className={`px-4 py-2.5 text-sm font-medium transition-colors ${
            tab === "models"
              ? "border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400"
              : "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
          }`}
        >
          Models
        </button>
        <button
          onClick={() => setTab("apis")}
          className={`px-4 py-2.5 text-sm font-medium transition-colors ${
            tab === "apis"
              ? "border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400"
              : "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
          }`}
        >
          API Providers
        </button>
      </div>

      <div className="mt-6 grid gap-6">
        {(tab === "models" ? models : apis).map((provider, i) => (
          <ModelTable key={i} provider={provider} />
        ))}
      </div>
    </div>
  );
}
