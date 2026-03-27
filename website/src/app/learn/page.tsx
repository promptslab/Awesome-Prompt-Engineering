import { fetchSiteData } from "@/lib/github";
import { buildSearchIndex } from "@/lib/parser";
import { ClientShell } from "@/components/ClientShell";
import { LearnClient } from "./LearnClient";
import type { Metadata } from "next";

export const revalidate = 300;

export const metadata: Metadata = {
  title: "Learn",
  description: "Books, courses, tutorials, and videos for learning prompt engineering.",
};

export default async function LearnPage() {
  const data = await fetchSiteData();
  const searchItems = buildSearchIndex(data);

  return (
    <ClientShell searchItems={searchItems}>
      <div className="py-6">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
          Learn
        </h1>
        <p className="mt-2 text-slate-500 dark:text-slate-400">
          Books, courses, tutorials, and videos for learning prompt engineering.
        </p>
      </div>
      <LearnClient
        books={data.books}
        courses={data.courses}
        tutorials={data.tutorials}
        videos={data.videos}
        bookCategories={data.bookCategories}
        courseSubcategories={data.courseSubcategories}
      />
    </ClientShell>
  );
}
