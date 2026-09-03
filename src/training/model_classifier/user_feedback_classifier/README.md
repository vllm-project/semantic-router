# User Feedback Classifier

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
