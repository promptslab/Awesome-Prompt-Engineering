#!/usr/bin/env python3
"""Sync the Autonomous Research section from awesome-autoresearch."""

import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://raw.githubusercontent.com/alvinunreal/awesome-autoresearch/main/README.md"
README_PATH = Path(__file__).resolve().parent.parent / "README.md"

START_MARKER = "<!-- AUTORESEARCH-START -->"
END_MARKER = "<!-- AUTORESEARCH-END -->"

# Sections to extract from the upstream README (in order)
SECTIONS = [
    ("General-Purpose Descendants", "🛠️ General-purpose descendants"),
    ("Research-Agent Systems", "🔬 Research-agent systems"),
    ("Platform Ports & Hardware Forks", "💻 Platform ports and hardware forks"),
    ("Domain-Specific Adaptations", "🎯 Domain-specific adaptations"),
    ("Evaluation & Benchmarks", "📊 Evaluation & benchmarks"),
    ("Related Resources", "📚 Related resources"),
]


def fetch_upstream() -> str:
    """Fetch the upstream awesome-autoresearch README."""
    req = urllib.request.Request(REPO_URL, headers={"User-Agent": "sync-autoresearch/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def extract_section(content: str, heading: str) -> list[str]:
    """Extract list items under a given ## heading."""
    # Match the heading (with or without emoji prefix)
    pattern = rf"^##\s+(?:.*?){re.escape(heading.split(' ', 1)[-1] if ' ' in heading else heading)}"
    lines = content.split("\n")
    items = []
    in_section = False

    for line in lines:
        if re.match(pattern, line, re.IGNORECASE):
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            # Capture list items, strip star badge images for cleaner display
            if line.startswith("- "):
                cleaned = re.sub(
                    r"\s*!\[GitHub stars\]\([^)]+\)\s*", " ", line
                ).strip()
                # Normalize " - " separators to " — "
                cleaned = re.sub(r"^(- \[[^\]]+\]\([^)]+\))\s+-\s+", r"\1 — ", cleaned)
                items.append(cleaned)
    return items


def build_section(upstream: str) -> str:
    """Build the replacement content for between the markers."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = [
        START_MARKER,
        "## 🔬 Autonomous Research & Self-Improving Agents",
        f"> Auto-synced from [awesome-autoresearch](https://github.com/alvinunreal/awesome-autoresearch) · Last synced: {today}",
        "",
    ]

    for local_heading, upstream_heading in SECTIONS:
        items = extract_section(upstream, upstream_heading)
        if not items:
            continue
        parts.append(f"### {local_heading}")
        parts.append("")
        for item in items:
            parts.append(item)
        parts.append("")

    parts.append(END_MARKER)
    return "\n".join(parts)


def update_readme(new_section: str) -> bool:
    """Replace content between markers in README.md. Returns True if changed."""
    readme = README_PATH.read_text(encoding="utf-8")

    pattern = re.compile(
        rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        re.DOTALL,
    )

    if not pattern.search(readme):
        print("ERROR: Could not find AUTORESEARCH markers in README.md")
        return False

    updated = pattern.sub(new_section, readme)

    if updated == readme:
        print("No changes detected.")
        return False

    README_PATH.write_text(updated, encoding="utf-8")
    print("README.md updated successfully.")
    return True


def main():
    print(f"Fetching upstream README from {REPO_URL}...")
    upstream = fetch_upstream()
    print(f"Fetched {len(upstream)} bytes.")

    new_section = build_section(upstream)
    update_readme(new_section)


if __name__ == "__main__":
    main()
