MAIN := main
LATEX := pdflatex
BIBTEX := bibtex

.PHONY: all rebuild artifact-check deploy clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib appendix_artifact_spec.tex math_commands.tex \
	$(wildcard sections/*.tex) $(wildcard figures/*.tex) $(wildcard figures/*.png) \
	$(wildcard *.sty) $(wildcard *.bst)
	$(LATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(LATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(LATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex

rebuild: clean
	$(MAKE) all

artifact-check:
	python3 artifact/verify.py
	python3 artifact/test_negative.py

deploy: $(MAIN).pdf
	cp $(MAIN).pdf ../../website/static/mom-paper.pdf
	@echo "=== Deployed: ../../website/static/mom-paper.pdf ==="

clean:
	rm -f $(MAIN).pdf
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk
