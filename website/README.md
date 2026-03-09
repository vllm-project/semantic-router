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

### Website Features

- **Mermaid and code block styling** integrated into the docs theme
- **Custom landing, publications, community, and white-paper routes**
- **Theme overrides** for docs and blog shells
- **Search-ready Docusaurus foundation**

### UX Goals

- **Fast loading** with optimized builds
- **Accessible design** following WCAG guidelines
- **Mobile-first** responsive layout
- **SEO optimized** with proper meta tags

## 📁 Project Structure

```
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
|---------|-------------|
| `make docs-dev` | Start development server |
| `make docs-build` | Build for production |
| `make docs-serve` | Preview production build |
| `make docs-clean` | Clear build cache |

## Links

- **Live Preview**: <http://localhost:3000> (when running)
- **Docusaurus Docs**: <https://docusaurus.io/docs>
- **Main Project**: ../README.md
