import Fuse from "fuse.js";
import { SearchItem } from "./types";

export function createSearchIndex(items: SearchItem[]) {
  return new Fuse(items, {
    keys: [
      { name: "name", weight: 0.6 },
      { name: "description", weight: 0.3 },
      { name: "category", weight: 0.1 },
    ],
    threshold: 0.4,
    includeScore: true,
    minMatchCharLength: 2,
  });
}
