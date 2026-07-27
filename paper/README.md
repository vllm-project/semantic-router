# Research paper sources

This directory contains the editable sources for the vLLM Semantic Router
research papers.

| Paper | Source entrypoint | Build | Published PDF |
| --- | --- | --- | --- |
| vLLM Semantic Router white paper | `main.tex` | `make -C paper` | `website/static/white-paper.pdf` |
| Mixture-of-Models position paper | `mom/main.tex` | `make -C paper/mom` | `website/static/mom-paper.pdf` |

The Mixture-of-Models source bundle also includes editable TikZ figures,
high-resolution scorecard inputs, the verified bibliography, and the proposed
portable bundle schemas and examples under `mom/artifact/`.
