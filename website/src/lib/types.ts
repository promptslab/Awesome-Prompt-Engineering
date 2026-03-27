export interface Resource {
  name: string;
  description: string;
  url: string;
  linkType?: "github" | "website" | "arxiv" | "other";
  stars?: string;
  category: string;
  subcategory: string;
}

export interface Paper {
  title: string;
  url: string;
  year?: string;
  venue?: string;
  description: string;
  category: string;
  subcategory: string;
}

export interface APIModel {
  name: string;
  context?: string;
  price?: string;
  keyFeature?: string;
  architecture?: string;
  significance?: string;
}

export interface APIProvider {
  provider: string;
  description?: string;
  url?: string;
  models: APIModel[];
  features?: string;
}

export interface Book {
  title: string;
  authors: string;
  publisher: string;
  year: string;
  category: string;
}

export interface StartHereStep {
  step: number;
  label: string;
  title: string;
  url: string;
  source?: string;
}

export interface Community {
  name: string;
  description: string;
  url: string;
  subcategory: string;
}

export interface SiteData {
  papers: Paper[];
  tools: Resource[];
  apis: APIProvider[];
  datasets: Resource[];
  models: APIProvider[];
  detectors: Resource[];
  books: Book[];
  courses: Paper[];
  tutorials: Paper[];
  videos: Paper[];
  communities: Community[];
  startHere: StartHereStep[];
  paperSubcategories: string[];
  toolSubcategories: string[];
  datasetSubcategories: string[];
  modelSubcategories: string[];
  detectorSubcategories: string[];
  bookCategories: string[];
  courseSubcategories: string[];
  communitySubcategories: string[];
}

export interface SearchItem {
  name: string;
  description: string;
  url: string;
  category: string;
  subcategory: string;
}
