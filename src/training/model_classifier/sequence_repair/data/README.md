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

Store annotations, review provenance, exclusion reasons and partition manifests
with the dataset artifacts for the run. The repository contains the source
manifest, hydration code and small contract fixtures. Source category does not
assign the FactCheck label: a brainstorming request can still ask for factual
country names, while an extraction request can be answered entirely from its
supplied context. Distinguish human source requests from assistant task labels.

Aya contributors form split groups. Dolly uses the source passage, or instruction
when no passage exists, as its grouping key. Identified duplicate passage topics
and instruction-template families across partitions are excluded in the reviewed
set. These checks cannot establish that old published weights or the base
encoder never saw the source data. Earlier FactCheck recipes used some Dolly
instructions and did not preserve a complete row-level training manifest.

`hydrate_annotations` verifies the source-file hash and reconstructs the exact
reviewed text. Record the sidecar, source and reconstructed JSONL checksums in
the run's data manifest.

`task-context-backgrounds.json` contains original prose for controlled context
stress. Train, development, and test use different background narratives.
Paragraphs repeat within an input to reach measured token budgets. A source
request reused at several lengths or positions is a paired variant, not a new
independent natural example.
