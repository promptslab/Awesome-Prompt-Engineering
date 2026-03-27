import { SiteData } from "./types";
import { parseReadme } from "./parser";

const GITHUB_RAW_URL =
  "https://raw.githubusercontent.com/promptslab/Awesome-Prompt-Engineering/main/README.md";

const GITHUB_API_URL =
  "https://api.github.com/repos/promptslab/Awesome-Prompt-Engineering";

export async function fetchReadme(): Promise<string> {
  const res = await fetch(GITHUB_RAW_URL, {
    next: { revalidate: 300 }, // ISR: revalidate every 5 minutes
  });

  if (!res.ok) {
    throw new Error(`Failed to fetch README: ${res.status}`);
  }

  return res.text();
}

export async function fetchSiteData(): Promise<SiteData> {
  const markdown = await fetchReadme();
  return parseReadme(markdown);
}

export async function fetchStarCount(): Promise<number> {
  try {
    const res = await fetch(GITHUB_API_URL, {
      next: { revalidate: 3600 }, // Revalidate star count every hour
    });

    if (!res.ok) return 5000; // fallback

    const data = await res.json();
    return data.stargazers_count || 5000;
  } catch {
    return 5000;
  }
}
