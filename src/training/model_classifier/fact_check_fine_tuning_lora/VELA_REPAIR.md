# Vela FactCheck repair

`Vela-1.0-Encoder-307M-FactCheck` classifies whether a request requires externally
grounded factual knowledge or verification. It does not determine whether an
answer is true. The existing contract is retained:

| ID | Label |
| --- | --- |
| 0 | NO_FACT_CHECK_NEEDED |
| 1 | FACT_CHECK_NEEDED |

A self-contained computation, fictional scene, subjective response, wording
change, or extraction from a supplied passage does not require an external check.
The passage must actually support the requested answer. Specific external facts,
current recommendations, historical comparisons, and factual explanations do
require verification. Ambiguous prompts are excluded from the reviewed set.

The Vela path initializes a fresh head and LoRA adapter from the pinned encoder.
Old FactCheck weights are baselines, not initial weights. Historical recipes
included sources that do not provide a clean, reusable provenance chain for a
new publication, and they did not retain complete row-level training receipts.

Prepare source files and hydrate all three reviewed-label sidecars using the
[shared workflow](../sequence_repair/README.md). First build the Domain corpus,
then the initial task corpus and natural extension:

```bash
python -m src.training.model_classifier.fact_check_fine_tuning_lora.prepare_vela_data \
  --domain-corpus artifacts/vela/domain --sources artifacts/vela/sources \
  --annotations artifacts/vela/annotations/aya-reviewed-labels-v1.jsonl \
  --output artifacts/vela/factcheck-v1

python -m src.training.model_classifier.fact_check_fine_tuning_lora.extend_natural \
  --corpus artifacts/vela/factcheck-v1 \
  --annotations artifacts/vela/annotations/aya-fact-reviewed-v2.jsonl \
    artifacts/vela/annotations/dolly-fact-reviewed-v2.jsonl \
  --output artifacts/vela/factcheck-v2
```

The initial corpus includes weak factual-question supervision and authored
transformation negatives. It must not be presented as independently human-labeled
fact-checking data. Aya and Dolly task labels were assigned by an assistant reading
the original request, including its full supplied context, before predictions;
source dataset categories did not mechanically assign those labels. Separate
source results expose errors hidden by an easy synthetic subset.

Dolly's source license is CC-BY-SA-3.0; its source attribution is preserved in the
manifest and annotation sidecar. The original classifier recipe used some Dolly
instructions, so historical exposure of the old baseline remains unknown. New
Vela train/dev/test groups stay separate, and identified duplicate passage or
instruction-template families are excluded before predictions. These are
independent task-training holdouts for the new run, not a claim of base-pretraining
novelty.

Initialize with `sequence_repair.initialize`, continue with
`sequence_repair.train`, and use source-balanced sampling/selection so that the
reviewed natural subset is not overwhelmed by weak examples. Development and
final results must separate source, language, and length. Report context-only
extraction failures alongside factual false negatives. Repeated-background
context tests supplement this evaluation; their score is not natural-document
accuracy. Freeze the candidate and actual Base lineage before the first final
test invocation.
