# vLLM Semantic Router Documentation

This directory contains the Docusaurus-based documentation website for the vLLM Semantic Router project.

## Quick Start

### Prerequisites

- Node.js 18+
- npm or yarn

### Development

Start the development server with hot reload:

```bash
# From project root
make docs-dev

# Or manually
cd website && npm start
```

The site will be available at <http://localhost:3000>

### Production Build

Build the static site for production:

```bash
# From project root
make docs-build

# Or manually
cd website && npm run build
```

### Preview Production Build

Serve the production build locally:

```bash
# From project root
make docs-serve

# Or manually
cd website && npm run serve
```

## Features

### Current Design System

- **Dark-only shell** built around a monochrome editorial system
- **Shared tokens and CSS layers** for homepage, docs, blog, and custom pages
- **Fixed chrome and route-aware wrappers** so docs/blog/community pages read as one site
- **Responsive layouts** tuned for mobile and desktop

### UI Contract

Treat the current website redesign as the default design contract for all public routes, not as a one-off homepage polish.

- **Dark-only, monochrome editorial language:** keep the black/graphite surfaces, bright neutral typography, thin borders, and restrained highlights. Do not reintroduce colorful default Docusaurus styling or a light-mode fork.
- **Shared system before page-local styling:** extend `src/css/tokens.css`, `src/css/base.css`, `src/css/shell.css`, and shared components under `src/components/site/` before adding bespoke per-page styles.
- **Homepage and custom pages stay bold; docs and blog stay readable:** landing routes can use stronger composition, diagrams, and motion, but docs/blog routes must preserve reading comfort, sidebar/TOC clarity, and code/table legibility.
- **Motion and effects stay restrained:** dither fields, hover lifts, and interactive figures should support the content hierarchy. Decorative effects must stay subtle and must not compete with the text.
- **Diagram language should match the shell:** use monochrome SVGs, line-art panels, thin strokes, centered compositions where appropriate, and card surfaces that feel like part of the same system.
- **Copy should stay high-signal and system-level:** prefer concise, technical language such as encoder, Shannon signals, entropy folding, neural-symbolic routing, and system intelligence. Avoid low-status product phrasing that breaks the tone.
- **Responsive behavior is part of the contract:** desktop and mobile are both first-class. New UI must avoid horizontal overflow and keep fixed header, docs navigation, tables, code blocks, and visual panels usable on narrow screens.
- **Route structure and docs affordances remain intact:** redesign work should preserve URLs, docs versions, locale routing, sidebar taxonomy, pagination, and article metadata unless a deliberate product change is being made.

### Website Features

- **Mermaid and code block styling** integrated into the docs theme
- **Custom landing, publications, community, and white-paper routes**
- **Theme overrides** for docs and blog shells
- **Local docs search** in the navbar, with no external search service (see below)

### Search

Search is provided by [`@easyops-cn/docusaurus-search-local`](https://github.com/easyops-cn/docusaurus-search-local), registered in the `themes` array of `docusaurus.config.ts`.

- **Local and offline.** The Lunr index is compiled during the build and served as a static asset from our own domain. There is no account, no API key, no crawler, no quota, and no network call at search time.
- **Keyboard shortcut.** `Ctrl+K` (Linux/Windows) or `Cmd+K` (macOS) opens and focuses the search box. `Escape` closes it. `/search` renders the standalone results page.
- **What is indexed.** The current documentation version (`docs/`) and the blog. Archived versions (`v0.1`-`v0.3`) are excluded through `ignoreFiles`, and `src/pages` marketing routes are excluded through `indexPages: false`. Searching from an archived-version page therefore returns current-docs results. To make archived versions searchable, drop `ignoreFiles` and add `searchContextByPaths` — there is a comment in the config explaining how.
- **The index is a build-time artefact.** Only `npm run build` (or `make docs-build`) regenerates it, so `npm run start` will not reflect fresh content in search results. Use a production build plus `npm run serve` when verifying search changes.
- **Both locales are searchable.** `language: ['en', 'zh']` enables the Chinese tokenizer, which is needed because Chinese is written without spaces; it is backed by `@node-rs/jieba` (a native module shipping prebuilt binaries, so nothing is compiled at install time). The Chinese search UI strings live in `i18n/zh-Hans/code.json` under `theme.SearchBar.*` and `theme.SearchPage.*`.
- **Styling.** All search overrides live in `src/css/search.css` so they can be removed cleanly if the search theme is ever swapped out. Note that the plugin's own `[data-theme="dark"]` rules never apply here — the site renders as `html[data-theme="light"]` and is dark through its own tokens — so the overrides target the vendor's light-mode rules.

### UX Goals

- **Fast loading** with optimized builds
- **Accessible design** following WCAG guidelines
- **Mobile-first** responsive layout
- **SEO optimized** with proper meta tags

## 📁 Project Structure

```text
website/
├── docs/                   # Documentation content (Markdown files)
├── src/
│   ├── components/        # Custom React components
│   ├── css/              # Global styles and theme
│   └── pages/            # Custom pages (homepage, etc.)
├── static/               # Static assets (images, icons, etc.)
├── docusaurus.config.ts  # Main configuration
├── sidebars.ts          # Navigation structure
└── package.json         # Dependencies and scripts
```

## Customization

### Styling

Use `src/css/custom.css` as the entrypoint. The real design layers live in:

- `src/css/tokens.css` for site tokens
- `src/css/base.css` for shared layout primitives
- `src/css/shell.css` for chrome, navbar, and footer
- `src/css/docs.css` for docs-specific styling
- `src/css/blog.css` for blog-specific styling

### Navigation

Update `sidebars.ts` to modify:

- Documentation structure
- Category organization
- Page ordering

### Site Configuration

Modify `docusaurus.config.ts` for:

- Site metadata
- Plugin configuration
- Theme settings
- Build options

## Available Commands

| Command | Description |
| ------- | ----------- |
| `make docs-dev` | Start development server |
| `make docs-build` | Build for production |
| `make docs-serve` | Preview production build |
| `make docs-clean` | Clear build cache |
| `make docs-check-translations` | Audit Chinese translation coverage, metadata, and source drift |
| `make docs-test-translation-sync` | Test translation status synchronization behavior |
| `make docs-fix-translation-status` | Update unambiguous Chinese translation `outdated` flags |

## Links

- **Live Preview**: <http://localhost:3000> (when running)
- **Docusaurus Docs**: <https://docusaurus.io/docs>
- **Main Project**: ../README.md
