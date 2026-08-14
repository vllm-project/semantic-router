# Embedding Anchor Design Principles

## Overview

An **embedding anchor pack** is the candidate phrase set that an `embedding`
signal rule matches in vector space. These principles apply to text and image
packs, including the optional image-routing example in
`config/fragments/signal/embedding/image-routing.yaml`.

Read this after the [Embedding Signal](./embedding) tutorial. That page covers the mechanics (`candidates`, `threshold`, `aggregation_method`, `query_modality`); this page covers how to make a pack that routes reliably instead of one that looks reasonable and misfires at scale.

## Key Advantages

- Turns anchor authoring into a repeatable pack-design workflow instead of trial-and-error phrase tweaking.
- Helps reduce false positives by treating the full anchor pack as the signal, not any single phrase.
- Makes threshold calibration and validation explicit so packs stay aligned with the embedding model in production.

## What Problem Does It Solve?

Embedding rules can look plausible in review and still route poorly in practice when anchors overfit to literal wording, omit benign counterexamples, or reuse thresholds from a different model. This guide explains how to build anchor packs that generalize across real traffic instead of only matching hand-picked examples.

## When to Use

Use these principles whenever you create or revise `embedding` signal rules, especially when you are:

- adding a new text or image routing category,
- recalibrating thresholds for a different embedding model, or
- debugging false positives or false negatives from an existing anchor pack.

## Configuration

Apply these principles to the same `embedding` signal fields described in the [Embedding Signal](./embedding) tutorial:

- `candidates` should cover the full category with multiple semantically distinct anchors,
- `aggregation_method` should match whether any strong anchor or broad pack agreement should trigger routing, and
- `threshold` must be calibrated against the deployed model and labeled corpus.

## Principle 1: anchors describe what the input *is*, not the words in it

An embedding anchor is matched in the model's semantic space, so it should describe the **signature** of the content, not a literal string you expect to appear.

- For image-modality rules this means describing the **visual** signature. `photograph of a passport page` works because passports have a recognizable visual structure; `the words "passport number"` does not, because the vision encoder embeds visual layout, not OCR'd text.
- For text-modality rules this means describing the **kind** of request. `a request to summarize a legal contract` generalizes; pasting one specific contract sentence as an anchor overfits to that phrasing.

A quick self-check: if an anchor only matches when a specific literal token is present, it is describing text, not signature, and it will generalize poorly.

## Principle 2: the pack defines category coverage

Cosine similarity is noisy, so one phrase rarely covers a useful category.
Author distinct anchors that represent the category from several angles; too
few make the rule brittle, while near-duplicates add little coverage.

Aggregation still matters. With `aggregation_method: max`, one highest-scoring
anchor can trigger the rule. Calibrate every anchor against negative examples
instead of assuming the pack votes as a group. Use `mean` only when broad
agreement across the pack is the behavior you want.

## Principle 3: cover the benign classes explicitly (negative-space anchors)

The most common failure mode is a benign input drifting closer to a sensitive anchor than to anything describing benign content - simply because nothing in the pack describes the benign case. The fix is **additive**: add anchors that positively describe the benign / ambient classes, rather than trying to subtract or blocklist them.

The image pack ships `ambient_office_imagery` for exactly this reason: whiteboards, conference rooms, generic office scenes, and wide factory/warehouse shots give low-sensitivity inputs something to match so they stay low-confidence instead of landing near a sensitive anchor by accident. Mirror this for any pack: for every sensitive category you route on, give the routine, non-sensitive content of the same surface its own anchors.

## Principle 4: calibrate the threshold to your model and corpus

Thresholds are not portable across embedding models or modalities.

- The checked-in image pack uses a much lower threshold than the text examples.
  Treat it as a starting point for that example model and corpus, not a portable
  default.
- A different embedding model, or the same model on a different content distribution, will have a different operating range. Always calibrate against your own labeled evaluation set rather than copying a threshold from an example pack.

## Principle 5: validate the pack as a unit before relying on it

A pack is a small classifier. Treat changes to it the way you would treat a model change:

- Keep a small labeled corpus (sensitive and benign examples for each category) and check the pack's accept/reject behavior against it before and after edits.
- When you add or remove an anchor, re-check that the benign corpus still stays below threshold - adding a sensitive anchor can pull benign inputs up with it.
- Record the model and threshold the corpus was calibrated against; both are part of the pack's contract.

## Reference: the opt-in image pack

[`config/fragments/signal/embedding/image-routing.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/embedding/image-routing.yaml)
is a worked example with three categories, multiple anchors per category,
`aggregation_method: max`, and an intentionally low image/text threshold.
Inline it under `routing.signals.embeddings`, replace the anchors with content
specific to your deployment, and recalibrate.

See the [Embedding Signal](./embedding) tutorial for the field reference and a full worked routing example.

Anchor packs are configuration, not trained safety models. Store the labeled
evaluation corpus and embedding-model version with the pack, and do not use a
semantic image rule as a substitute for OCR, PII detection, or content
moderation.
