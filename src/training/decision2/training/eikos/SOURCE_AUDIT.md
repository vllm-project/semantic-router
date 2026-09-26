# Eikos-4B source and rights evidence (2026-09-26)

This audit covers the pinned upstream starting point for the Eikos-native
Decision 2.0 arm. The initial `balanced_human_5824_v1` run contains
noncommercial or research-only TweetEval rows; a separate source attestation
must bind any model release to its noncommercial research scope and upstream
conditions. This source audit alone does not clear the new training lineage.

## Immutable upstream identity

- Model: [`caiovicentino1/Eikos-4B`](https://huggingface.co/caiovicentino1/Eikos-4B)
  revision `582ffb13f19a4da3f455e3db198584190bd7755b`.
- Local source `SHA256SUMS`: `8978a143290508976d8ddffb735416954cca1de779d5524ca4c59d528539d3dd`.
- `NOTICE`: `281b5c258f113cf0fd53df3bd20868eb8b3c63aef37e6496d1c172e468b3a073`.
  Eikos `LICENSE` (MIT): `2cf8c18cf63fd0e31addba3775a2bbec6f58aa31bb5bdfb549c640bd9b676611`.
  `LICENSE-Qwen` (Apache-2.0): `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`.
- Source card identifies a Qwen3.5-4B backbone, a native SemIf letter-logit
  readout, and an author-released merged checkpoint. Our exported package
  copies native readout files and all three notices verbatim.

## Author-disclosed training provenance

The [Eikos Decisions dataset card](https://huggingface.co/datasets/caiovicentino1/eikos-decisions)
reports 23,168 Eikos-4B training rows and 1,296 development rows. It describes
generated and programmatic examples, teacher probability targets, and 1,994
human-labelled FinEntity examples. The [model card](https://huggingface.co/caiovicentino1/Eikos-4B)
also describes rationale, language-view, and light JEPA auxiliary losses. These
are author statements; we did not reconstruct the original training run.

The dataset card marks the release CC BY 4.0, except row-level `upstream`
sources: FinEntity (ODC-BY 1.0), TAT-QA (CC BY 4.0), and GSM8K train (MIT).
The pinned `NOTICE` names each source and credits SemIf (MIT) for the prompt
format. The card says synthetic item writers and teachers were GLM-5.3-Flash
and Qwen models, and asserts their outputs can be redistributed. No explicit
noncommercial source is disclosed for the upstream Eikos-4B fine-tune.

## Limits of this audit

- FinEntity states contain real public financial-news text. The dataset card
  identifies the FinEntity annotation license, but it does not independently
  establish rights for every underlying news excerpt.
- Teacher-output redistribution and source dataset rights are author claims
  in the current public cards. Preserve their attribution and review pinned
  source terms before a release.
- The upstream Eikos weights being openly licensed does not clear our own
  fine-tuning, checkpoint-selection, or temperature-calibration inputs. The
  old pilot requires an explicit noncommercial-use/source attestation; the
  separate rights-clean split is a control candidate with its own manifest.

The released Decision 2.0 model card should identify this upstream lineage,
license notices, added training sources, and measured limitations separately.

The original 5,824-row pilot has an independent
`decision2-noncommercial-research-attestation/1` statement, SHA-256
`a0eb728d4c876a8d3e7f2763f94a594998629f8f9ca9edbf10afaeb8d6bb5d91`.
It binds the training-run provenance, data manifest, all three split hashes,
19 TRAIN source buckets and the SELECT/CAL source buckets. The statement names
the noncommercial or research-only terms and marks unresolved copied-text
rights for implicit-hate, coarse-discourse, and MultiNLI. It contains no raw
source rows and grants no new rights in those sources.
