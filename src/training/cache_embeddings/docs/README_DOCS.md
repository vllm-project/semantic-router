# Cache Embeddings Training Documentation

This directory contains comprehensive documentation for the domain-specific cache embeddings training pipeline.

## 📚 Documentation Files

### Interactive HTML Documentation
- **[index.html](index.html)** - Main interactive documentation page with tooltips, diagrams, and visualizations

### Technical Documentation
- **[README.md](README.md)** - Complete technical deep-dive with methodology, architecture, and usage
- **[QUICK_START_AWS.md](QUICK_START_AWS.md)** - AWS deployment guide with one-command setup
- **[blog.md](blog.md)** - Validation methodology and margin-based testing explanation

## 🚀 Viewing the Documentation

### Option 1: Open Directly in Browser
```bash
# From the repository root
open src/training/cache_embeddings/docs/index.html

# Or using your preferred browser
chrome src/training/cache_embeddings/docs/index.html
firefox src/training/cache_embeddings/docs/index.html
```

### Option 2: Local Web Server (Recommended)
```bash
# Navigate to the docs directory
cd src/training/cache_embeddings/docs

# Start a simple HTTP server
python3 -m http.server 8000

# Then visit in your browser:
# http://localhost:8000
```

### Option 3: VS Code Live Server
If you have the Live Server extension in VS Code:
1. Open `index.html` in VS Code
2. Right-click and select "Open with Live Server"

## ✨ Features

The interactive HTML documentation includes:

- **📊 Interactive Tooltips** - Hover over technical terms for plain-English explanations
- **📈 Visual Metrics** - Key performance indicators prominently displayed
- **🎨 Modern UI** - Dark theme, smooth animations, responsive design
- **🔍 Complete Pipeline** - Step-by-step walkthrough from data to deployment
- **📉 Results Visualization** - Tables and charts showing performance improvements
- **💻 Code Examples** - Copy-paste ready commands and snippets
- **🎯 Manager-Friendly** - Clear business impact and ROI explanations
- **👨‍💻 Developer-Friendly** - Technical details and implementation guides

## 🎯 Target Audiences

### For Managers
- Clear ROI metrics (21.4% margin improvement)
- Business impact explanations
- Production-ready features highlighted
- Cost-efficient AWS deployment

### For Developers
- Complete technical architecture
- Step-by-step implementation guide
- Code examples and commands
- AWS deployment automation

### For Researchers
- Research paper references (arXiv:2504.02268v1)
- Methodology explanations
- Validation approach
- Future research directions

## 📝 Documentation Maintenance

When updating the documentation:

1. **HTML** - Main entry point, keep it user-friendly with tooltips
2. **README.md** - Technical reference for developers
3. **QUICK_START_AWS.md** - AWS deployment instructions
4. **blog.md** - Validation methodology and results

Keep all files in sync when making significant changes to the pipeline or results.

## 🔗 Quick Links

- [Main Project Repository](../../../../)
- [arXiv Paper: 2504.02268v1](https://arxiv.org/pdf/2504.02268v1)
- [MedQuAD Dataset](https://github.com/abachaa/MedQuAD)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Sentence-BERT Paper (MNR Loss)](https://arxiv.org/abs/1908.10084)
