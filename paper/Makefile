# Makefile for vLLM Semantic Router arXiv paper
# Usage:
#   make          - Build PDF (full compilation with bibliography)
#   make quick    - Quick build (single pdflatex pass, no bib)
#   make clean    - Remove all generated files
#   make watch    - Continuously rebuild on file changes (requires inotifywait)
#   make wc       - Word count estimate
#   make check    - Check for common LaTeX issues

MAIN     = main
TEX      = pdflatex
BIB      = bibtex
TEXFLAGS = -interaction=nonstopmode -halt-on-error -file-line-error

SECTIONS = $(wildcard sections/*.tex)
SOURCES  = $(MAIN).tex $(SECTIONS) references.bib

# Generated files to clean
AUX_EXTS = aux bbl blg log out toc lof lot fls fdb_latexmk synctex.gz nav snm vrb run.xml bcf

.PHONY: all quick clean watch wc check

# Full build: pdflatex -> bibtex -> pdflatex -> pdflatex
all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	$(TEX) $(TEXFLAGS) $(MAIN)
	$(BIB) $(MAIN) || true
	$(TEX) $(TEXFLAGS) $(MAIN)
	$(TEX) $(TEXFLAGS) $(MAIN)
	@echo ""
	@echo "=== Build complete: $(MAIN).pdf ==="

# Quick single-pass build (no bibliography update)
quick:
	$(TEX) $(TEXFLAGS) $(MAIN)
	@echo ""
	@echo "=== Quick build complete: $(MAIN).pdf ==="

# Remove all generated files
clean:
	@for ext in $(AUX_EXTS); do \
		rm -f $(MAIN).$$ext; \
	done
	rm -f $(MAIN).pdf
	@echo "=== Cleaned ==="

# Watch for changes and rebuild (requires inotifywait from inotify-tools)
watch:
	@echo "Watching for changes... (Ctrl+C to stop)"
	@while true; do \
		inotifywait -qe modify $(SOURCES) 2>/dev/null || sleep 2; \
		echo ""; \
		echo "=== Change detected, rebuilding... ==="; \
		$(MAKE) quick; \
	done

# Approximate word count (strips LaTeX commands)
wc:
	@echo "=== Approximate word count ==="
	@cat sections/*.tex | \
		sed 's/\\[a-zA-Z]*\(\[[^]]*\]\)\?{[^}]*}//g' | \
		sed 's/\\[a-zA-Z]*//g' | \
		sed 's/[{}$$\\]//g' | \
		wc -w | \
		xargs -I{} echo "  Body text: ~{} words"
	@echo "  Sections:  $$(ls sections/*.tex | wc -l) files"
	@echo "  Total TeX: $$(cat $(MAIN).tex sections/*.tex | wc -l) lines"

# Check for common issues
check:
	@echo "=== Checking for common issues ==="
	@echo "Undefined references:"
	@grep -c 'undefined' $(MAIN).log 2>/dev/null || echo "  (none or not yet compiled)"
	@echo "Missing citations:"
	@grep -c 'Citation.*undefined' $(MAIN).log 2>/dev/null || echo "  (none or not yet compiled)"
	@echo "Overfull boxes:"
	@grep -c 'Overfull' $(MAIN).log 2>/dev/null || echo "  (none or not yet compiled)"
