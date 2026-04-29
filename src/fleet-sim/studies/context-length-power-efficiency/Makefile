PYTHON ?= python3
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex

.PHONY: all tables pdf clean distclean help

all: tables pdf

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make tables     Run all table reproduction scripts' \
		'  make pdf        Build main.pdf from LaTeX sources' \
		'  make clean      Remove LaTeX auxiliary files' \
		'  make distclean  Remove auxiliary files and main.pdf'

tables:
	$(PYTHON) scripts/reproduce_all.py

pdf: main.pdf

main.pdf: main.tex refs.bib
	$(PDFLATEX) main.tex
	$(BIBTEX) main
	$(PDFLATEX) main.tex
	$(PDFLATEX) main.tex

clean:
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.synctex.gz

distclean: clean
	rm -f main.pdf
