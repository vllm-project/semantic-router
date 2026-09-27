# Decision 2.0 package paths: native System One response parity

The [System One API reference](https://docs.typesafe.ai/api) defines three
question types. Choice returns the selected option, option probabilities and
confidence; Noul returns the probability of yes; Score returns an expected
level, level legend, probabilities and confidence. A Decision 2.0 product
package should expose these typed decisions, not a chat response.

The current full-checkpoint package source and adapter package source already
use the shared `product_answer` conversion. A legacy package source path used
the benchmark-only `normalized_answer`, which omits Choice/Score product
fields. This source change moves the legacy path to the same conversion and
rejects non-text/non-object/non-array state values before inference. A CPU
contract test sends the same structured state and Choice/Noul/Score questions
through all three package paths, including a no-truncation over-budget case,
and asserts equal response objects. Numeric logits and the benchmark scoring
path are unchanged.

This is a package-source compatibility correction, **not a trained-model
improvement**. The private 0.6B full-checkpoint package already used the
correct path; its weights, published private revision, JevArena v3 and public
JevBench scores are unchanged. Any future package made through the corrected
legacy path still needs exact-revision download, full-file hashes, native
Choice/Noul/Score example execution and benchmark-output parity before upload.
