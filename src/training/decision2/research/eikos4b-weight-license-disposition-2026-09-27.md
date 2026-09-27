# Decision 2.0 4B: public weight-license disposition

**Finding, 2026-09-27: conditional route exists; the exact candidate remains
HOLD for public redistribution until an accountable rights reviewer records the
artifact-specific decision and final license text.** This note is an
independent source and package review, not that release authorization. It does
not change the frozen weights, predictions, score files, or original evaluation
gate.

## Exact facts and scope

- The proposed `DEV2.0-4B` weights were initialized from
  [`caiovicentino1/Eikos-4B`](https://huggingface.co/caiovicentino1/Eikos-4B)
  revision `582ffb13f19a4da3f455e3db198584190bd7755b`, itself based on
  [`Qwen/Qwen3.5-4B-Base`](https://huggingface.co/Qwen/Qwen3.5-4B-Base).
  Eikos's card identifies MIT for its contributions and Apache-2.0 for Qwen.
  The candidate's `LICENSE` (Eikos MIT), `LICENSE-Qwen` (Apache-2.0), and
  `NOTICE` are byte-identical to the pinned Eikos snapshot according to the
  independent [package audit](decision2-4b-release-record-rights-audit-2026-09-27.md).
  All 21 entries in the package's `SHA256SUMS` verified; that list's SHA-256 is
  `7e005ef609553d973c3a6232840436d1384657a19f5765e42752ebcc04906e39`.
- The frozen rights-clean v2 TRAIN contains 7,455 records. Its rights manifest
  names 272 [SNLI](https://nlp.stanford.edu/projects/snli/) and 334
  [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) training records;
  both publishers identify CC BY-SA 4.0. The manifest excludes raw training
  rows and individual text predictions from the proposed public artifact.
  `publication_eligible` and a mechanical `rights_gate.verify_rights` pass
  attest a documented data scope, not the public weight license.
- Native SemIf inference emits option letters and probabilities. This reduces
  normal-service exposure of training text. It is **not** proof that the
  downloadable weights cannot reproduce text: the Qwen backbone retains an
  `lm_head` (used directly by
  [`training/eikos/train.py`](../training/eikos/train.py)), and downstream
  users control how weights are loaded. The 606-row count, absence of raw rows
  in the package, and zero detected TRAIN-to-evaluation near matches do not
  establish that no source expression was memorized. No targeted
  source-memorization study is part of the current rights receipt.

## What the primary terms do and do not establish

[Creative Commons' AI-training guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)
says its conservative route for publicly shared models based on ShareAlike
material is to use the same CC license. It explicitly calls that route likely
overcompliance and explains that application of copyright and exceptions is
fact-specific. The [BY-SA 4.0 legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode)
defines adapted material by copyright-relevant transformation of licensed
material; BY and SA are triggered by public sharing of the original or an
adaptation. It does **not** declare every model trained on BY-SA examples an
adaptation. Consequently:

1. **Yes, conditionally:** a named, accountable reviewer can reach and
   document a fact-specific conclusion that these weights do not share
   copyright-relevant adaptations of the SNLI/SQuAD expression. On that
   premise, a clear custom noncommercial license for **our own fine-tuning
   contribution** is a possible distribution choice. It must not imply that
   the original Qwen Apache-2.0 or Eikos MIT grants are withdrawn, or that the
   datasets themselves are sublicensed. Include dataset attribution and the
   inherited license texts and `NOTICE`. The review should identify applicable
   jurisdiction, exact package, architecture and native output, what was
   checked for memorization, and the remaining uncertainty.
2. **No present clearance:** the currently frozen evidence does not determine
   whether weights or alternative uses of their generative head contain or
   disclose protected expression. The candidate cannot be marked rights PASS
   solely from `license: other`, a private/research intent, a mechanical data
   audit, or this note.
3. **If the conservative ShareAlike route is chosen instead:** the final
   license treatment of our contributions and combined model must be reviewed
   against the inherited Apache/MIT notices. [Creative Commons lists the
   compatible licenses](https://creativecommons.org/compatible-licenses/);
   Apache-2.0 is not listed as a substitute adapter license for BY-SA 4.0.
   A noncommercial restriction must not simply be added to a BY-SA adaptation,
   because [BY-SA 4.0 section 3(b)](https://creativecommons.org/licenses/by-sa/4.0/legalcode)
   forbids restrictive additional terms on adapted material. The effect on
   this mixed-origin weight artifact requires the same artifact-specific review.

Hugging Face's [license documentation](https://huggingface.co/docs/hub/repositories-licenses)
requires a real `LICENSE` text and `license_name` for `license: other`; the
[`model-card metadata guide`](https://huggingface.co/docs/hub/model-cards)
also allows `license_link`. The frozen candidate's `LICENSE` is only Eikos's
MIT text. Therefore changing the metadata to `other` without a reviewed
composite license file would mislead downloaders. If custom terms are approved,
create a **derived publication package** with the new composite `LICENSE`,
retain a byte-identical Eikos MIT copy under `LICENSE-Eikos`, and preserve
`LICENSE-Qwen` and `NOTICE`. Recompute the file inventory and perform package
read-back and native-output parity against the frozen weights. No training or
evaluation result is changed by that documentation-only packaging step.

## Conditional model-card wording; not yet authorized for publication

> **License and provenance.** These weights are a fine-tune of Eikos-4B
> revision `582ffb13f19a4da3f455e3db198584190bd7755b`, whose contributions
> are MIT-licensed and whose Qwen3.5-4B-Base ancestry is Apache-2.0.
> `LICENSE-Eikos`, `LICENSE-Qwen`, and `NOTICE` preserve those terms and
> attributions. Our fine-tuning used the rights-clean v2 training mix,
> including 272 SNLI and 334 SQuAD 2.0 records under CC BY-SA 4.0; dataset
> text is not included in this repository. [Insert the signed, exact-artifact
> rights disposition and link to the **actual** license governing our
> contribution here.] Dataset licenses remain with the source datasets and
> are not replaced by this model card. The SemIf interface returns bounded
> decisions, but this does not establish that all possible uses of the weights
> cannot reproduce source text.

Do not publish the bracketed placeholder or advertise the combined checkpoint
as wholly MIT, wholly Apache-2.0, automatically BY-SA, or unrestricted open
source. A statement that this project conducted noncommercial research is a
description of **our use**, not a license granted to downstream users.

**One minimum external input:** a dated, signed disposition from the
accountable model-rights owner or qualified rights reviewer, bound to the
frozen package and data-manifest hashes, that chooses the no-adaptation or
ShareAlike treatment, supplies the exact public weight-license text and
attribution scope, and addresses the inherited MIT/Apache terms. If that
reviewer cannot support a license for these frozen weights, the technical
alternative is a newly trained candidate excluding the unresolved rows,
followed by new selection, calibration, package verification, and independent
evaluation under a new identity.
