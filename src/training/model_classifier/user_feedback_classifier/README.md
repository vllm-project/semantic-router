# User Feedback Classifier

For the Vela generation, see the [application recipes](../vela-applications.md).
They preserve historical recipes below while defining the new data, label,
training and independent evaluation contracts.

This pipeline trains a four-class classifier for a user's follow-up message:

| Label | Meaning |
|---|---|
| `SAT` | expresses satisfaction |
| `NEED_CLARIFICATION` | asks for clarification or more explanation |
| `WRONG_ANSWER` | says the previous answer is incorrect |
| `WANT_DIFFERENT` | requests another option or approach |

The model consumes the follow-up text only. That keeps inference simple, but it
also means ambiguous replies such as “yes” or “not that one” may require
conversation context outside this classifier.

Ordinary new questions have no valid label in this four-class taxonomy. The
caller must establish that the current turn is feedback on a preceding answer;
a high softmax confidence does not establish that applicability. In particular,
requests such as “explain photosynthesis” or “give me different dinner ideas”
can resemble clarification or revision feedback. The development-only
`vela_applicability` diagnostic reports these forced classifications without
inventing a SAT label or an accuracy score. A previous assistant turn is a
necessary routing gate, but does not by itself distinguish a new topic.

## Install and Train

```bash
pip install -r requirements.txt

python train_feedback_detector.py \
  --model_name llm-semantic-router/mmbert-32k-yarn \
  --data_source llm-semantic-router/feedback-detector-dataset \
  --output_dir models/feedback-detector \
  --max_samples 2000 \
  --epochs 1
```

Use the short run to verify data loading and output. Remove the sample cap and
tune on validation data for a full run. `--use_lora`, `--lora_rank`,
`--lora_alpha`, and `--merge_lora` control adapter training and export; see
`python train_feedback_detector.py --help` for current defaults.

## Inference

```python
from inference_feedback import FeedbackDetector

detector = FeedbackDetector("models/feedback-detector")
result = detector.classify("Could you explain that another way?")
print(result.label, result.confidence, result.all_scores)
```

The helper defaults to an explicit 512-token budget, including special tokens.
It rejects over-budget inputs instead of silently truncating them. Set
`max_length` up to the checkpoint's real position capacity when using a verified
long-context checkpoint; capacity alone does not establish feedback quality.

`classify_batch()` accepts a list of follow-up messages. Treat confidence as a
model score, not a calibrated probability, unless calibration has been measured
on the deployment distribution.

## Evaluation and Use

Before connecting feedback labels to routing or online learning, measure the
confusion matrix and per-class precision/recall on held-out conversations.
Inspect short, multilingual, sarcastic, and context-dependent replies. Routing
policy should define the consequence of each label and should not update model
experience from a low-confidence prediction without safeguards.

Published checkpoints and datasets should have their own model or dataset card
with revisions, split policy, base model, metrics, and limitations. Links in a
README are not a substitute for that evidence.

## Feedback repair v2

The historical 98.83% validation score does not establish generalization: its
SAT validation texts overlapped training templates. The historical JSONL export
also includes non-feedback statements and assistant responses labelled SAT.
The repair curriculum excludes both sources rather than treating those labels
as ground truth.

The four output IDs remain `SAT=0`, `NEED_CLARIFICATION=1`, `WRONG_ANSWER=2`, and
`WANT_DIFFERENT=3`. Classify the current user's feedback. A statement that needs
unavailable conversation context should not be forced into an unambiguous gold
label. In particular, asking for explanation differs from requesting a different
style or approach.

```bash
python -m src.training.model_classifier.user_feedback_classifier.prepare_repair_data \
  --output-dir /tmp/feedback-repair-v2
python -m src.training.model_classifier.user_feedback_classifier.train_feedback_detector \
  --model_name llm-semantic-router/mmbert-32k-yarn \
  --model_revision 72a23a6640489471eb4ff7ad3ec5bc80af8a27de \
  --data_source /tmp/feedback-repair-v2 --output_dir /tmp/feedback-v2 \
  --use_lora --merge_lora --lora_rank 32 --lora_alpha 64 \
  --epochs 8 --batch_size 16 --lr 0.0001 --max_length 2048
```

This authored curriculum separates expression families and topics between
training and validation. Translations and topic substitutions are correlated
examples, not thousands of independent user judgments. Choose checkpoints on
validation, then compare with the original checkpoint on an independently
frozen final set. Do not select a checkpoint by the old overlapping validation
score or claim production accuracy from authored tests alone.

The 32K backbone's position capacity differs from task accuracy at that length.
Record evaluated token counts and truncation. `FeedbackDetector` defaults to a
512-token resource budget and accepts an explicit `max_length` up to the actual
checkpoint capacity. It validates the output mapping instead of silently
renaming arbitrary classifier outputs. Export fresh ONNX graphs for changed
weights; never copy an old graph into the new model repository.
