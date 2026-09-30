# PAWS-derived lexical polarity fixtures

Source: PAWS by Yuan Zhang, Jason Baldridge, and Luheng He (NAACL 2019),
Google LLC. The upstream dataset terms are retained in `LICENSE`.

`pairs.json` pins the Google Research Datasets PAWS `labeled_final` test
revision and the retrieved Dataset Viewer page hashes. IDs 39, 74, and 1437
retain both original sentences and the original human paraphrase label.
The other three pairs insert `not` or replace `active` with `inactive` in
sentence 1. They have explicitly local labels: they are not original PAWS
negative examples or human annotations. The original GCS archive was not
available during collection; provenance is the pinned Google Research
Datasets Hugging Face mirror, not an asserted archive-byte comparison.

The regression tests the English lexical floor with above-threshold candidate
vectors and with an explicitly supplied published embedding model. It does
not measure PAWS accuracy or establish coverage of cue-less, word-order-only,
or non-English contradictions. For example, original PAWS test pair 87
reverses which company acquired another without an applicable lexical cue.
That class of error still requires model/threshold calibration or an
appropriate semantic verifier.
