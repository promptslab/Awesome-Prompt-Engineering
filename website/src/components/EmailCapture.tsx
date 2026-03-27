"use client";

import { useState } from "react";

const identityOptions = [
  "AI/ML Engineer",
  "Software Developer",
  "Product Manager",
  "Researcher / Academic",
];

export function EmailCapture() {
  const [email, setEmail] = useState("");
  const [identity, setIdentity] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !identity) return;

    setStatus("loading");
    try {
      const res = await fetch("/api/subscribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, identity }),
      });

      if (res.ok) {
        setStatus("success");
        setEmail("");
        setIdentity("");
      } else {
        setStatus("error");
      }
    } catch {
      setStatus("error");
    }
  };

  if (status === "success") {
    return (
      <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-6 text-center dark:border-emerald-500/20 dark:bg-emerald-500/10">
        <p className="font-semibold text-emerald-700 dark:text-emerald-300">
          You&apos;re on the list! We&apos;ll notify you when the course launches.
        </p>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="mx-auto max-w-md space-y-4">
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="your@email.com"
        required
        className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-900 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:focus:border-indigo-400 dark:focus:ring-indigo-400/20"
      />

      <div className="space-y-2">
        <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
          I&apos;m a...
        </p>
        <div className="grid grid-cols-2 gap-2">
          {identityOptions.map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => setIdentity(option)}
              className={`rounded-xl border px-3 py-2.5 text-sm font-medium transition-all ${
                identity === option
                  ? "border-indigo-500 bg-indigo-50 text-indigo-700 shadow-sm shadow-indigo-500/10 dark:border-indigo-400 dark:bg-indigo-500/15 dark:text-indigo-300"
                  : "border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-slate-50 dark:border-slate-600 dark:text-slate-400 dark:hover:border-slate-500 dark:hover:bg-slate-700/50"
              }`}
            >
              {option}
            </button>
          ))}
        </div>
      </div>

      <button
        type="submit"
        disabled={!email || !identity || status === "loading"}
        className="w-full rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-500/25 transition-all hover:from-indigo-600 hover:to-violet-600 hover:shadow-xl hover:shadow-indigo-500/30 disabled:opacity-50 disabled:shadow-none"
      >
        {status === "loading" ? "Subscribing..." : "Notify Me When It Launches"}
      </button>

      {status === "error" && (
        <p className="text-center text-sm text-red-600 dark:text-red-400">
          Something went wrong. Please try again.
        </p>
      )}
    </form>
  );
}
