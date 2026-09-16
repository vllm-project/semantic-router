# Vela Feedback

Vela Feedback classifies the current user turn as `SAT`, `NEED_CLARIFICATION`,
`WRONG_ANSWER`, `WANT_DIFFERENT`, or `NO_FEEDBACK`. An independent request is not
feedback merely because it contains words such as “explain” or “wrong”. Quoted
text and supplied code belong to the task being discussed, not automatically to
the user's assessment of an earlier answer.

## Prepare training data

Keep training, development, and final evaluation separate by conversation,
source document, and authored family. Translations and long-context variants
retain their parent's partition. Review complete user inputs against the same
label definitions; exclude unresolved cases instead of assigning an arbitrary
class. Include natural independent requests alongside all four feedback types.

The data tools accept reviewed annotations as explicit inputs:

- `vela_no_feedback --source source.json --review review.json --output reviewed/`
  materializes exact reviewed rows and verifies their source and partition.
- `vela_natural_negatives --previous corpus/ --wildfeedback source.json
  --review review.json --output expanded/` appends reviewed training requests
  without rewriting development data.
- `vela_current_turn --previous corpus/ --wildfeedback source.json
  --review review.json --output refined/` applies exact reviewed group overrides
  and reports excluded weak supervision.
- `vela_weak_supervision_v2 --input train.jsonl --wildfeedback source.json
  --quote-policy structured-v3 --quote-review review.json --output filtered/`
  requires the review associated with that projection. Its remaining source
  labels are weak supervision, not reviewed gold.
- `vela_answer_quality_contrasts --registry reviewed-families.json
  --output contrasts/` freezes bilingual intent families and their supplied
  annotations, retaining the declared family partitions and input hashes.

Run these modules with
`python -m src.training.model_classifier.user_feedback_classifier.<module>`.
Store annotation artifacts and dataset revisions with the dataset release.
Historical review snapshots and training outputs do not belong in the source
checkout. Authored family definitions remain generator inputs; correlated
variants do not constitute independent evaluation examples.

## Train from the Vela base

Start a new release from the qualified Vela base with a fresh five-class head.
Do not initialize it from a classifier adapter trained on a previous base.
Download the base at an immutable revision and keep its file hashes with the
training run.

The contract file contains the exact mapping and native pooling choice:

```json
{
  "label2id": {
    "SAT": 0,
    "NEED_CLARIFICATION": 1,
    "WRONG_ANSWER": 2,
    "WANT_DIFFERENT": 3,
    "NO_FEEDBACK": 4
  },
  "id2label": {
    "0": "SAT",
    "1": "NEED_CLARIFICATION",
    "2": "WRONG_ANSWER",
    "3": "WANT_DIFFERENT",
    "4": "NO_FEEDBACK"
  },
  "classifier_pooling": "cls",
  "problem_type": "single_label_classification"
}
```

Each JSONL row supplies `id`, `text`, `label`, `group_id`, and `source`. Record
measured `length_bucket` for context evaluation. Freeze the data recipe and
training budget before comparing development scores.

```bash
python -m src.training.model_classifier.sequence_repair.train \
  --base /artifacts/vela-base \
  --base-id llm-semantic-router/Vela-1.0-Encoder-307M \
  --base-revision fe9ccc074b781bc0e2e13c2c8d26f2640410636a \
  --method full --fresh-head \
  --contract /artifacts/feedback-contract.json \
  --train /artifacts/feedback-train.jsonl \
  --dev /artifacts/feedback-development.jsonl \
  --output /artifacts/feedback-run \
  --steps 1600 --batch-size 16 --accumulate 1 \
  --learning-rate 0.00001 --head-learning-rate 0.0001 \
  --max-length 32768 --microbatch-token-budget 32768 \
  --balanced-sampling --length-balanced-sampling \
  --evaluation-dtype float32 --selection macro-f1 --eval-every 1600
```

These are explicit experiment settings, not a quality guarantee or the sampling
recipe of a particular published checkpoint. To replay a frozen source mixture,
replace the sampling flags with `--training-order order.json`; the shared trainer
verifies the complete order against eligible records and source hashes. See
[sequence training](../sequence_repair/README.md) for its format.

## Evaluate and export

Measure per-class precision and recall, non-feedback false positives, source and
language slices, and each supported context length. Include natural follow-up
requests, quotations, code, sarcasm, and independent questions. Keep the original
weak development data as a diagnostic; do not treat it as independent gold or
select on final evaluation.

Position capacity and task accuracy are separate. A checkpoint that accepts
32K tokens must also preserve its task decisions with relevant evidence at
different positions. Record actual token counts and truncation, and compare
native FP32 inference with each exported engine on the same inputs.

Export the qualified complete checkpoint with the shared exporter and
`--runtime-task feedback`. Regenerate ONNX graphs from its actual weights.
Never copy graphs from an older model into a new release.

Optional development-only applicability calibration is available through
`calibrate_vela_bias`. It changes the actual classifier bias, reruns native FP32
inference, and exports only a candidate satisfying its declared constraints.
Freeze any calibration before final evaluation; a softmax score is not itself
a validated probability of user intent.

## Inference

```python
from inference_feedback import FeedbackDetector

detector = FeedbackDetector(
    "llm-semantic-router/Vela-1.0-Encoder-307M-Feedback",
    max_length=32768,
)
result = detector.classify("Could you clarify your previous explanation?")
print(result.label, result.confidence, result.all_scores)
```

The helper accepts an explicit token budget up to the checkpoint's position
capacity and rejects oversized input. Its default resource budget is 512 tokens.
It supports both the legacy four-label mapping and Vela's five-label mapping;
`NO_FEEDBACK` remains a distinct result. Router confidence thresholds and the
consequence of a predicted label are separate routing policy decisions.
