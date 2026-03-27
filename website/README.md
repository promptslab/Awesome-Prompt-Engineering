## Awesome Prompt Engineering — Website

This is the website for the [Awesome Prompt Engineering](https://github.com/promptslab/Awesome-Prompt-Engineering) collection.

### How it works

- Built with Next.js (App Router) + Tailwind CSS
- Fetches README.md from GitHub at runtime (ISR, 5-minute revalidation)
- Parses markdown into structured typed data
- No manual data sync needed — README is the single source of truth

### Development

```bash
cd website
npm install
npm run dev
```

### Deploy

Set Vercel root directory to `website/`. No additional config needed.
