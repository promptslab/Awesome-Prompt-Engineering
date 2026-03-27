"use client";

import { useState } from "react";
import { Book, Paper } from "@/lib/types";
import { PaperCard } from "@/components/ResourceCard";

function BookCard({ book }: { book: Book }) {
  return (
    <div className="flex flex-col rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-700/50 dark:bg-slate-800/50">
      <h3 className="font-medium text-slate-900 dark:text-white">
        {book.title}
      </h3>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        {book.authors}
      </p>
      <div className="mt-3 flex flex-wrap gap-1.5">
        <span className="rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700 dark:bg-indigo-500/15 dark:text-indigo-300">
          {book.publisher}
        </span>
        <span className="rounded-full bg-violet-50 px-2.5 py-0.5 text-xs font-medium text-violet-700 dark:bg-violet-500/15 dark:text-violet-300">
          {book.year}
        </span>
        <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs text-slate-600 dark:bg-slate-700/50 dark:text-slate-400">
          {book.category}
        </span>
      </div>
    </div>
  );
}

export function LearnClient({
  books,
  courses,
  tutorials,
  videos,
}: {
  books: Book[];
  courses: Paper[];
  tutorials: Paper[];
  videos: Paper[];
  bookCategories: string[];
  courseSubcategories: string[];
}) {
  const [tab, setTab] = useState<"books" | "courses" | "tutorials" | "videos">("courses");

  const tabs = [
    { key: "courses" as const, label: "Courses", count: courses.length },
    { key: "tutorials" as const, label: "Tutorials", count: tutorials.length },
    { key: "books" as const, label: "Books", count: books.length },
    { key: "videos" as const, label: "Videos", count: videos.length },
  ];

  return (
    <div>
      <div className="flex gap-1 border-b border-slate-200 dark:border-slate-700/50">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2.5 text-sm font-medium transition-colors ${
              tab === t.key
                ? "border-b-2 border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400"
                : "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
            }`}
          >
            {t.label} ({t.count})
          </button>
        ))}
      </div>

      <div className="mt-6">
        {tab === "books" && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {books.map((book, i) => (
              <BookCard key={i} book={book} />
            ))}
          </div>
        )}

        {tab === "courses" && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {courses.map((course, i) => (
              <PaperCard key={i} paper={course} />
            ))}
          </div>
        )}

        {tab === "tutorials" && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {tutorials.map((tutorial, i) => (
              <PaperCard key={i} paper={tutorial} />
            ))}
          </div>
        )}

        {tab === "videos" && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {videos.map((video, i) => (
              <PaperCard key={i} paper={video} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
