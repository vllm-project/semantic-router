# vLLM Semantic Router Documentation

This directory contains the Docusaurus-based documentation website for the vLLM Semantic Router project.

## 🚀 Quick Start

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

The build runs a lightweight public-docs structure check before invoking Docusaurus.

### Preview Production Build

Serve the production build locally:

```bash
# From project root
make docs-serve

# Or manually
cd website && npm run serve
```

## 🎨 Features

### ✨ Modern Tech-Inspired Design

- **Dark theme by default** with neon blue/green accents
- **Glassmorphism effects** with backdrop blur and transparency
- **Gradient backgrounds** and animated hover effects
- **Responsive design** optimized for all devices

### 🔧 Enhanced Functionality

- **Mermaid diagram support** with dark theme optimization
- **Advanced code highlighting** with multiple language support
- **Interactive navigation** with smooth animations
- **Search functionality** (ready for Algolia integration)

### 📱 User Experience

- **Fast loading** with optimized builds
- **Accessible design** following WCAG guidelines
- **Mobile-first** responsive layout
- **SEO optimized** with proper meta tags

## 📁 Project Structure

```text
website/
├── docs/                     # English public documentation
├── i18n/                     # Localized docs and UI strings
├── scripts/                  # Website-specific validation helpers
├── src/
│   ├── components/           # Custom React components
│   ├── css/                  # Global styles and theme
│   └── pages/                # Homepage and custom pages
├── static/                   # Static assets
├── docusaurus.config.ts      # Site configuration
├── sidebars.ts               # Canonical public docs navigation
└── package.json              # Website scripts and dependencies
```

## 🛠️ Customization

### Themes and Colors

Edit `src/css/custom.css` to modify:

- Color scheme and gradients
- Typography and spacing
- Component styling
- Animations and effects

### Navigation

Update `sidebars.ts` to modify:

- Documentation structure
- Category organization
- Page ordering
- Public doc reachability

### Site Configuration

Modify `docusaurus.config.ts` for:

- Site metadata
- Plugin configuration
- Theme settings
- Build options

### Documentation Governance

Use the structure check before landing navigation changes:

```bash
# From project root
make docs-check-structure

# Or manually
cd website && npm run check:structure
```

The structure check verifies that:

- every public English markdown page is reachable from `sidebars.ts`
- locale-only top-level sections are explicitly allowlisted
- the public docs tree does not drift silently away from the canonical navigation

## 📚 Available Commands

| Command | Description |
|---------|-------------|
| `make docs-dev` | Start development server |
| `make docs-check-structure` | Validate public docs reachability and locale structure |
| `make docs-lint` | Lint website source files |
| `make docs-build` | Build for production |
| `make docs-serve` | Preview production build |
| `make docs-clean` | Clear build cache |

## 🔗 Links

- **Live Preview**: <http://localhost:3000> (when running)
- **Docusaurus Docs**: <https://docusaurus.io/docs>
- **Main Project**: ../README.md
