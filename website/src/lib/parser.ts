import {
  SiteData,
  Paper,
  Resource,
  APIProvider,
  APIModel,
  Book,
  StartHereStep,
  Community,
  SearchItem,
} from "./types";

function extractLink(text: string): { label: string; url: string } | null {
  const match = text.match(/\[([^\]]*)\]\(([^)]+)\)/);
  if (match) return { label: match[1], url: match[2] };
  return null;
}

function extractAllLinks(
  text: string
): { label: string; url: string }[] {
  const re = /\[([^\]]*)\]\(([^)]+)\)/g;
  const links: { label: string; url: string }[] = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    links.push({ label: m[1], url: m[2] });
  }
  return links;
}

function stripMarkdown(text: string): string {
  return text
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]+\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .trim();
}

function extractStars(text: string): string | undefined {
  const match = text.match(/~?([\d,]+K?\+?)\s*⭐/);
  return match ? match[1] : undefined;
}

function extractYear(text: string): string | undefined {
  const match = text.match(/\[(\d{4})\]/);
  return match ? match[1] : undefined;
}

function detectLinkType(url: string): "github" | "website" | "arxiv" | "other" {
  if (url.includes("github.com")) return "github";
  if (url.includes("arxiv.org")) return "arxiv";
  if (
    url.includes("http") &&
    !url.includes("github.com") &&
    !url.includes("arxiv.org")
  )
    return "website";
  return "other";
}

function parseTableRows(
  tableText: string
): { cells: string[] }[] {
  const lines = tableText.split("\n").filter((l) => l.trim().startsWith("|"));
  if (lines.length < 2) return [];
  // Skip header and separator
  const dataLines = lines.slice(2);
  return dataLines.map((line) => {
    const cells = line
      .split("|")
      .slice(1, -1)
      .map((c) => c.trim());
    return { cells };
  });
}

function parseListItems(text: string): Paper[] {
  const lines = text.split("\n").filter((l) => l.trim().startsWith("- "));
  return lines.map((line) => {
    const content = line.replace(/^-\s*/, "").trim();
    const link = extractLink(content);
    const year = extractYear(content);

    // Extract description after the dash separator
    const descMatch = content.match(/[—–-]\s*(.+)$/);
    let description = descMatch ? descMatch[1].trim() : "";
    // Clean description of markdown links and formatting
    description = stripMarkdown(description);

    // Extract venue from brackets after year, like [ACL 2024]
    const venueMatch = content.match(
      /\[\d{4}(?:,\s*)?([A-Z][A-Za-z\s]+\d{0,4})\]/
    );
    const venue = venueMatch ? venueMatch[1].trim() : undefined;

    return {
      title: link?.label || stripMarkdown(content.split(/[—–]/)[0]),
      url: link?.url || "",
      year,
      venue,
      description,
      category: "",
      subcategory: "",
    };
  });
}

function parseStartHere(section: string): StartHereStep[] {
  const lines = section.split("\n").filter((l) => /^\d+\./.test(l.trim()));
  return lines.map((line, i) => {
    const content = line.replace(/^\d+\.\s*/, "").trim();
    // Extract the bold label
    const labelMatch = content.match(/\*\*([^*]+)\*\*/);
    const label = labelMatch ? labelMatch[1] : "";

    // Extract links
    const links = extractAllLinks(content);
    const mainLink = links[0];

    // Extract source info after parentheses
    const afterArrow = content.split("→").slice(1).join("→").trim();

    return {
      step: i + 1,
      label,
      title: mainLink?.label || stripMarkdown(afterArrow.split("·")[0]),
      url: mainLink?.url || "",
      source: undefined,
    };
  });
}

function parseToolsTable(
  tableText: string,
  subcategory: string
): Resource[] {
  const rows = parseTableRows(tableText);
  return rows
    .filter((r) => r.cells.length >= 3)
    .map((row) => {
      const nameCell = row.cells[0];
      const descCell = row.cells[1];
      const linkCell = row.cells[2];

      const boldMatch = nameCell.match(/\*\*([^*]+)\*\*/);
      const name = boldMatch ? boldMatch[1] : stripMarkdown(nameCell);
      const stars = extractStars(descCell);
      const description = stripMarkdown(descCell);
      const link = extractLink(linkCell);
      const allLinks = extractAllLinks(linkCell);
      const url = link?.url || allLinks[0]?.url || "";

      return {
        name,
        description,
        url,
        linkType: detectLinkType(url),
        stars,
        category: "Tools",
        subcategory,
      };
    });
}

function parseBooksTable(
  tableText: string,
  category: string
): Book[] {
  const rows = parseTableRows(tableText);
  return rows
    .filter((r) => r.cells.length >= 4)
    .map((row) => {
      const titleRaw = row.cells[0];
      const boldMatch = titleRaw.match(/\*\*([^*]+)\*\*/);
      return {
        title: boldMatch ? boldMatch[1] : stripMarkdown(titleRaw),
        authors: stripMarkdown(row.cells[1]),
        publisher: stripMarkdown(row.cells[2]),
        year: stripMarkdown(row.cells[3]),
        category,
      };
    });
}

function parseAPIProviderSection(sectionText: string): APIProvider {
  const lines = sectionText.split("\n");
  // First line should be the provider name
  const providerLine = lines[0]?.trim() || "";
  const provider = providerLine.replace(/^###?\s*/, "").trim();

  // Find table in section
  const tableStart = lines.findIndex((l) => l.trim().startsWith("|"));
  let models: APIModel[] = [];

  if (tableStart >= 0) {
    const tableText = lines.slice(tableStart).join("\n");
    const rows = parseTableRows(tableText);
    models = rows.map((row) => {
      const cells = row.cells;
      // Check which kind of table based on header
      const headerLine = lines[tableStart];
      if (headerLine?.includes("Key Feature") || headerLine?.includes("Key Strength")) {
        return {
          name: stripMarkdown(cells[0] || ""),
          context: stripMarkdown(cells[cells.length >= 4 ? (headerLine.includes("Provider") ? 2 : 1) : 1] || ""),
          price: cells.length >= 4 ? stripMarkdown(cells[2] || "") : undefined,
          keyFeature: stripMarkdown(cells[cells.length - 1] || ""),
        };
      }
      if (headerLine?.includes("Key Detail")) {
        return {
          name: stripMarkdown(cells[0] || ""),
          keyFeature: stripMarkdown(cells[1] || ""),
        };
      }
      if (headerLine?.includes("Significance")) {
        return {
          name: stripMarkdown(cells[0] || ""),
          context: stripMarkdown(cells[1] || ""),
          significance: stripMarkdown(cells[2] || ""),
        };
      }
      // generic
      return {
        name: stripMarkdown(cells[0] || ""),
        context: cells[1] ? stripMarkdown(cells[1]) : undefined,
        price: cells[2] ? stripMarkdown(cells[2]) : undefined,
        keyFeature: cells[3] ? stripMarkdown(cells[3]) : undefined,
      };
    });
  }

  // Find features text (paragraph after table)
  const featuresMatch = sectionText.match(
    new RegExp("Key features?:\\s*(.+?)(?:\\n\\n|\\n###|$)", "s")
  );
  const features = featuresMatch
    ? stripMarkdown(featuresMatch[1].trim())
    : undefined;

  // Find URL
  const allLinks = extractAllLinks(sectionText);
  const url = allLinks[allLinks.length - 1]?.url;

  return {
    provider,
    models,
    features,
    url,
  };
}

function parseDetectorsTable(
  tableText: string,
  subcategory: string
): Resource[] {
  const rows = parseTableRows(tableText);
  return rows
    .filter((r) => r.cells.length >= 3)
    .map((row) => {
      const nameCell = row.cells[0];
      const boldMatch = nameCell.match(/\*\*([^*]+)\*\*/);
      const name = boldMatch ? boldMatch[1] : stripMarkdown(nameCell);

      // For 4-column detector tables (name, accuracy, feature, link)
      // For 3-column tables (name, description, link)
      let description: string;
      let url: string;
      if (row.cells.length >= 4) {
        description = `${stripMarkdown(row.cells[1])} — ${stripMarkdown(row.cells[2])}`;
        const link = extractLink(row.cells[3]);
        url = link?.url || "";
      } else {
        description = stripMarkdown(row.cells[1]);
        const link = extractLink(row.cells[2]);
        url = link?.url || "";
      }

      return {
        name,
        description,
        url,
        linkType: detectLinkType(url),
        category: "Detectors",
        subcategory,
      };
    });
}

function parseDatasetsTable(
  tableText: string,
  subcategory: string
): Resource[] {
  const rows = parseTableRows(tableText);
  return rows
    .filter((r) => r.cells.length >= 3)
    .map((row) => {
      const nameCell = row.cells[0];
      const boldMatch = nameCell.match(/\*\*([^*]+)\*\*/);
      const name = boldMatch ? boldMatch[1] : stripMarkdown(nameCell);
      const description = stripMarkdown(row.cells[1]);
      const linkCell = row.cells[2] || row.cells[row.cells.length - 1];
      const link = extractLink(linkCell);
      const url = link?.url || "";

      return {
        name,
        description,
        url,
        linkType: detectLinkType(url),
        category: "Datasets",
        subcategory,
      };
    });
}

function parseCommunities(section: string): { communities: Community[]; subcategories: string[] } {
  const communities: Community[] = [];
  const subcategories: string[] = [];

  const subsections = section.split(/^### /m).filter(Boolean);
  for (const sub of subsections) {
    const lines = sub.split("\n");
    const subcategoryName = lines[0]?.trim() || "";
    if (!subcategoryName) continue;
    subcategories.push(subcategoryName);

    const listItems = lines.filter((l) => l.trim().startsWith("- "));
    for (const item of listItems) {
      const content = item.replace(/^-\s*/, "").trim();
      const link = extractLink(content);
      const descMatch = content.match(/[—–-]\s*(.+)$/);
      let description = descMatch ? descMatch[1].trim() : "";
      description = stripMarkdown(description);

      if (!description) {
        // Remove the link part and use remaining text
        const cleaned = content
          .replace(/\[([^\]]*)\]\([^)]+\)/, "")
          .replace(/^[—–-]\s*/, "")
          .trim();
        description = stripMarkdown(cleaned);
      }

      communities.push({
        name: link?.label || stripMarkdown(content.split(/[—–]/)[0]),
        description,
        url: link?.url || "",
        subcategory: subcategoryName,
      });
    }
  }

  return { communities, subcategories };
}

export function parseReadme(markdown: string): SiteData {
  // Split by ## headings
  const sections: { title: string; content: string }[] = [];
  const parts = markdown.split(/^## /m);

  for (const part of parts.slice(1)) {
    const newlineIdx = part.indexOf("\n");
    const title = part.substring(0, newlineIdx).trim();
    const content = part.substring(newlineIdx + 1);
    sections.push({ title, content });
  }

  const findSection = (keyword: string) =>
    sections.find((s) =>
      s.title.toLowerCase().includes(keyword.toLowerCase())
    );

  // Parse Start Here
  const startHereSection = findSection("Start Here");
  const startHere = startHereSection
    ? parseStartHere(startHereSection.content)
    : [];

  // Parse Papers
  const papersSection = findSection("Papers");
  const papers: Paper[] = [];
  const paperSubcategories: string[] = [];

  if (papersSection) {
    const subs = papersSection.content.split(/^### /m).filter(Boolean);
    for (const sub of subs) {
      const lines = sub.split("\n");
      const subcategory = lines[0]?.trim().replace(/[📄🔧💻💾🧠🔎📖👩‍🏫📚🎥🤝]/g, "").trim() || "";
      if (!subcategory) continue;
      paperSubcategories.push(subcategory);

      const items = parseListItems(lines.slice(1).join("\n"));
      items.forEach((item) => {
        item.category = "Papers";
        item.subcategory = subcategory;
      });
      papers.push(...items);
    }
  }

  // Parse Tools
  const toolsSection = findSection("Tools and Code");
  const tools: Resource[] = [];
  const toolSubcategories: string[] = [];

  if (toolsSection) {
    const subs = toolsSection.content.split(/^### /m).filter(Boolean);
    for (const sub of subs) {
      const lines = sub.split("\n");
      const subcategory = lines[0]?.trim().replace(/[📄🔧💻💾🧠🔎📖👩‍🏫📚🎥🤝]/g, "").trim() || "";
      if (!subcategory) continue;
      toolSubcategories.push(subcategory);

      const tableContent = lines.slice(1).join("\n");
      tools.push(...parseToolsTable(tableContent, subcategory));
    }
  }

  // Parse APIs
  const apisSection = findSection("APIs");
  const apis: APIProvider[] = [];

  if (apisSection) {
    const subs = apisSection.content.split(/^### /m).filter(Boolean);
    for (const sub of subs) {
      const provider = parseAPIProviderSection("### " + sub);
      if (provider.provider && provider.provider !== "APIs") {
        apis.push(provider);
      }
    }
  }

  // Parse Datasets
  const datasetsSection = findSection("Datasets and Benchmarks");
  const datasets: Resource[] = [];
  const datasetSubcategories: string[] = [];

  if (datasetsSection) {
    const subs = datasetsSection.content.split(/^### /m).filter(Boolean);
    for (const sub of subs) {
      const lines = sub.split("\n");
      const subcategory = lines[0]?.trim().replace(/[📄🔧💻💾🧠🔎📖👩‍🏫📚🎥🤝]/g, "").trim() || "";
      if (!subcategory) continue;
      datasetSubcategories.push(subcategory);

      const tableContent = lines.slice(1).join("\n");
      datasets.push(...parseDatasetsTable(tableContent, subcategory));
    }
  }

  // Parse Models
  const modelsSection = findSection("Models");
  const models: APIProvider[] = [];
  const modelSubcategories: string[] = [];

  if (modelsSection) {
    const subs = modelsSection.content.split(/^### /m).filter(Boolean);
    for (const sub of subs) {
      const lines = sub.split("\n");
      const subcategory = lines[0]?.trim().replace(/[📄🔧💻💾🧠🔎📖👩‍🏫📚🎥🤝]/g, "").trim() || "";
      if (!subcategory) continue;
      modelSubcategories.push(subcategory);

      const provider = parseAPIProviderSection("### " + sub);
      provider.provider = subcategory;
      models.push(provider);
    }
  }

  // Parse Detectors
  const detectorsSection = findSection("AI Content Detectors");
  const detectors: Resource[] = [];
  const detectorSubcategories: string[] = [];

  if (detectorsSection) {
    const subs = detectorsSection.content.split(/^### /m).filter(Boolean);
    for (const sub of subs) {
      const lines = sub.split("\n");
      const subcategory = lines[0]?.trim().replace(/[📄🔧💻💾🧠🔎📖👩‍🏫📚🎥🤝]/g, "").trim() || "";
      if (!subcategory) continue;
      detectorSubcategories.push(subcategory);

      const tableContent = lines.slice(1).join("\n");
      detectors.push(...parseDetectorsTable(tableContent, subcategory));
    }
  }

  // Parse Books
  const booksSection = findSection("Books");
  const books: Book[] = [];
  const bookCategories: string[] = [];

  if (booksSection) {
    const subs = booksSection.content.split(/^### /m).filter(Boolean);
    for (const sub of subs) {
      const lines = sub.split("\n");
      const category = lines[0]?.trim().replace(/[📄🔧💻💾🧠🔎📖👩‍🏫📚🎥🤝]/g, "").trim() || "";
      if (!category) continue;
      bookCategories.push(category);

      const tableContent = lines.slice(1).join("\n");
      books.push(...parseBooksTable(tableContent, category));
    }
  }

  // Parse Courses
  const coursesSection = findSection("Courses");
  const courses: Paper[] = [];
  const courseSubcategories: string[] = [];

  if (coursesSection) {
    const subs = coursesSection.content.split(/^### /m).filter(Boolean);
    for (const sub of subs) {
      const lines = sub.split("\n");
      const subcategory = lines[0]?.trim().replace(/[📄🔧💻💾🧠🔎📖👩‍🏫📚🎥🤝]/g, "").trim() || "";
      if (!subcategory) continue;
      courseSubcategories.push(subcategory);

      const items = parseListItems(lines.slice(1).join("\n"));
      items.forEach((item) => {
        item.category = "Courses";
        item.subcategory = subcategory;
      });
      courses.push(...items);
    }
  }

  // Parse Tutorials
  const tutorialsSection = findSection("Tutorials and Guides");
  const tutorials: Paper[] = [];

  if (tutorialsSection) {
    const subs = tutorialsSection.content.split(/^### /m).filter(Boolean);
    for (const sub of subs) {
      const lines = sub.split("\n");
      const subcategory = lines[0]?.trim().replace(/[📄🔧💻💾🧠🔎📖👩‍🏫📚🎥🤝]/g, "").trim() || "";

      const items = parseListItems(lines.slice(1).join("\n"));
      items.forEach((item) => {
        item.category = "Tutorials";
        item.subcategory = subcategory || "Tutorials";
      });
      tutorials.push(...items);
    }
  }

  // Parse Videos
  const videosSection = findSection("Videos");
  const videos: Paper[] = [];

  if (videosSection) {
    const items = parseListItems(videosSection.content);
    items.forEach((item) => {
      item.category = "Videos";
      item.subcategory = "Videos";
    });
    videos.push(...items);
  }

  // Parse Communities
  const communitiesSection = findSection("Communities");
  let communities: Community[] = [];
  let communitySubcategories: string[] = [];

  if (communitiesSection) {
    const result = parseCommunities(communitiesSection.content);
    communities = result.communities;
    communitySubcategories = result.subcategories;
  }

  return {
    papers,
    tools,
    apis,
    datasets,
    models,
    detectors,
    books,
    courses,
    tutorials,
    videos,
    communities,
    startHere,
    paperSubcategories,
    toolSubcategories,
    datasetSubcategories,
    modelSubcategories,
    detectorSubcategories,
    bookCategories,
    courseSubcategories,
    communitySubcategories,
  };
}

export function buildSearchIndex(data: SiteData): SearchItem[] {
  const items: SearchItem[] = [];

  data.papers.forEach((p) =>
    items.push({
      name: p.title,
      description: p.description,
      url: p.url,
      category: "Papers",
      subcategory: p.subcategory,
    })
  );

  data.tools.forEach((t) =>
    items.push({
      name: t.name,
      description: t.description,
      url: t.url,
      category: "Tools",
      subcategory: t.subcategory,
    })
  );

  data.datasets.forEach((d) =>
    items.push({
      name: d.name,
      description: d.description,
      url: d.url,
      category: "Datasets",
      subcategory: d.subcategory,
    })
  );

  data.detectors.forEach((d) =>
    items.push({
      name: d.name,
      description: d.description,
      url: d.url,
      category: "Detectors",
      subcategory: d.subcategory,
    })
  );

  data.courses.forEach((c) =>
    items.push({
      name: c.title,
      description: c.description,
      url: c.url,
      category: "Courses",
      subcategory: c.subcategory,
    })
  );

  data.tutorials.forEach((t) =>
    items.push({
      name: t.title,
      description: t.description,
      url: t.url,
      category: "Tutorials",
      subcategory: t.subcategory,
    })
  );

  data.videos.forEach((v) =>
    items.push({
      name: v.title,
      description: v.description,
      url: v.url,
      category: "Videos",
      subcategory: v.subcategory,
    })
  );

  data.books.forEach((b) =>
    items.push({
      name: b.title,
      description: `${b.authors} — ${b.publisher} (${b.year})`,
      url: "",
      category: "Books",
      subcategory: b.category,
    })
  );

  data.communities.forEach((c) =>
    items.push({
      name: c.name,
      description: c.description,
      url: c.url,
      category: "Communities",
      subcategory: c.subcategory,
    })
  );

  // Add API models
  data.apis.forEach((a) =>
    a.models.forEach((m) =>
      items.push({
        name: `${m.name} (${a.provider})`,
        description: m.keyFeature || m.price || "",
        url: a.url || "",
        category: "APIs",
        subcategory: a.provider,
      })
    )
  );

  // Add model entries
  data.models.forEach((group) =>
    group.models.forEach((m) =>
      items.push({
        name: m.name,
        description: m.keyFeature || m.significance || "",
        url: "",
        category: "Models",
        subcategory: group.provider,
      })
    )
  );

  return items;
}
