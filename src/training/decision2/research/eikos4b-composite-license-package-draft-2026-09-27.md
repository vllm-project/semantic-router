# Decision 2.0 4B: conservative composite license package draft

This is an **additive publication proposal**, not a rights PASS or a new
evaluation result. The [independent license review](eikos4b-weight-license-disposition-2026-09-27.md)
found that a fact-specific no-adaptation determination is possible but not
established by the present evidence. Following the owner's direction, this
draft instead grants CC BY-SA 4.0 for original Decision 2.0 fine-tuning
contributions to the extent the contributors own those rights. It does not
impose a noncommercial restriction in the grant. The exact text proposed for
top-level `LICENSE` is
[`decision2-4b-composite-LICENSE.txt`](../publication/decision2-4b-composite-LICENSE.txt).

The wrapper `scripts/package_postkey_aggregate_v3.py` only packages the
reviewed `llm-semantic-router/DEV2.0-4B` release record derived from the
pinned Eikos-4B revision `582ffb13f19a4da3f455e3db198584190bd7755b`.
It fails if the record lacks a real reviewer, if rights are pending, or if
the record still asserts a noncommercial-only weight scope. It leaves the
frozen native model directory, weight tensors, calibration, predictions,
scores, pre-key HOLD, and post-key ranking diagnosis unchanged. The derived
package contains:

| Top-level file | Purpose |
| --- | --- |
| `LICENSE` | Proposed composite grant and component attribution. |
| `LICENSE-Eikos` | Byte-identical copy of native upstream MIT license. |
| `LICENSE-Qwen` | Byte-identical copy of inherited Apache-2.0 license. |
| `NOTICE` | Byte-identical copy of inherited Eikos third-party notice. |
| `README.md` | `license: other`, descriptive `license_name` and `license_link`, and model/data lineage disclosure. |

The derived package manifest hashes each added file and the updated card;
verification compares the three inherited top-level notices with the native
copies. `license: other` is a container for the **actual attached terms**, not
an independent grant. [Hugging Face's license documentation](https://huggingface.co/docs/hub/repositories-licenses)
requires the custom text in a `LICENSE` file, and its
[model-card guide](https://huggingface.co/docs/hub/model-cards) describes
`license_name` and `license_link`.

The proposed wording follows [Creative Commons' conservative AI-training
guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)
for ShareAlike source material. The guidance also says it likely overcomplies
with what copyright law requires in many situations. Applying the proposed
CC BY-SA grant only to the fine-tuning contribution, while preserving the
inherited MIT and Apache terms in one combined binary, is a question for the
independent, accountable reviewer. That reviewer must check the exact final
`LICENSE`, model card, rights record, source licenses and package hashes
before updating `rights.status`; the draft does not supply a reviewer name or
fabricate a sign-off. An unreviewed package remains HOLD even if every byte
integrity test passes.
