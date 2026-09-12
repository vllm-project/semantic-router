# Reviewed task-data provenance

`source-files.json` pins the exact public text/metadata files used by the recipes.
Paths use repository IDs with `/` replaced by `--`. The MMLU-Pro and category
supplement files are historical-exposure screening inputs for Domain; they are
not claimed to be a new independent test set.

- Aya: `CohereLabs/aya_dataset`, revision
  `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`, Apache-2.0.
- Global-MMLU: `CohereLabs/Global-MMLU`, revision
  `0e619dbeb34206cd48705a1a0ea7fb21cae09993`, Apache-2.0.
- DiffusionDB captions: `poloclub/diffusiondb`, revision
  `fb620fbe49fa4420e0734bd9c0df11f51176b61f`, CC0-1.0. No images are used.
- Dolly: `databricks/databricks-dolly-15k`, revision
  `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, CC-BY-SA-3.0.
  Copyright 2023 Databricks, Inc.; Wikipedia editors and contributors for source
  passages. Retain source attribution and applicable license terms when
  distributing reconstructed data or derived annotations. The
  [official card](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/README.md)
  describes the employee-written instructions and Wikipedia context fields.

The three reviewed sidecars contain 216 original Aya requests with per-task
labels, 59 additional Aya FactCheck requests (57 retained), and 192 Dolly
FactCheck requests (177 retained). Every task judgment was made by an assistant
reading the request before model predictions; these labels are not human-expert
annotations. Excluded records preserve the review reason. Source category does
not assign the FactCheck label: for example, a Dolly brainstorming request can
still ask for factual country names, while an extraction request can be answered
entirely from its supplied context.

Aya contributors form split groups. Dolly uses the source passage, or instruction
when no passage exists, as its grouping key. Identified duplicate passage topics
and instruction-template families across partitions are excluded in the reviewed
set. These checks cannot establish that old published weights or the base
encoder never saw the source data. Earlier FactCheck recipes used some Dolly
instructions and did not preserve a complete row-level training manifest.

`hydrate_annotations` verifies the source-file hash and reconstructs the exact
reviewed text. Reconstructed v2 JSONL checksums are
`140e6f04b9ec52dd9dfaac2322f6180ffc0c292859a43d0423557940b48c7df5`
for Aya and
`b8c1a2e3badef640f08ee1a68240fed7647d0902bd4fabf833a99008c2d17cd0`
for Dolly. Both were reproduced against the pinned source files.

`task-context-backgrounds.json` contains original prose for controlled context
stress. Train, development, and test use different background narratives.
Paragraphs repeat within an input to reach measured token budgets. A source
request reused at several lengths or positions is a paired variant, not a new
independent natural example.
