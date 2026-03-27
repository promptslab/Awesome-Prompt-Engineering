"use client";

import { ReactNode } from "react";
import Link from "next/link";
import { Header } from "./Header";
import { Footer } from "./Footer";
import { SearchDialog } from "./SearchDialog";
import { useSearch } from "@/hooks/useSearch";
import { SearchItem } from "@/lib/types";

export function ClientShell({
  children,
  searchItems,
  hideCourseBar,
}: {
  children: ReactNode;
  searchItems: SearchItem[];
  hideCourseBar?: boolean;
}) {
  const { open, onOpen, onClose } = useSearch();

  return (
    <>
      <Header onSearchOpen={onOpen} />
      {!hideCourseBar && (
        <div className="mx-auto max-w-7xl px-4 pt-4 pb-2">
          <Link
            href="/course"
            className="group relative block overflow-hidden rounded-xl border border-indigo-200 bg-gradient-to-r from-indigo-50 via-violet-50 to-purple-50 px-6 py-4 transition-all hover:shadow-lg dark:border-indigo-500/20 dark:from-indigo-950/40 dark:via-violet-950/30 dark:to-purple-950/20"
          >
            <div className="relative flex flex-col items-center justify-between gap-4 sm:flex-row">
              <div className="flex items-center gap-3">
                <span className="inline-block rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white">
                  Coming Soon
                </span>
                <h2 className="text-lg font-bold text-slate-900 dark:text-white">
                  Prompt Engineering Course
                </h2>
                <p className="hidden text-sm text-slate-500 dark:text-slate-400 lg:block">
                  From prompt basics to production-grade systems
                </p>
              </div>
              <div className="flex items-center gap-3">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 px-4 py-1.5 text-sm font-semibold text-white shadow-md shadow-indigo-500/25 transition-all group-hover:shadow-lg group-hover:shadow-indigo-500/30">
                  Get Early Access
                  <svg className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </span>
                <span className="inline-flex items-center gap-1.5 rounded-full border border-indigo-200 bg-white px-4 py-1.5 text-sm font-medium text-indigo-600 transition-all group-hover:border-indigo-300 dark:border-indigo-500/20 dark:bg-indigo-500/10 dark:text-indigo-300">
                  Learn More
                </span>
              </div>
            </div>
          </Link>
        </div>
      )}
      <main className="mx-auto max-w-7xl px-4 py-6">{children}</main>
      <Footer />
      <SearchDialog items={searchItems} open={open} onClose={onClose} />
    </>
  );
}
