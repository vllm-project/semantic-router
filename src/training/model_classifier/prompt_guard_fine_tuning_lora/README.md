# PromptGuard

Vela detects prompt injection and jailbreak attempts independently of content
safety. The current licensed-source builders are `vela_data.py` and
`vela_authored.py`; training uses the explicit shared sequence loop. See the
[Vela application recipes](../vela-applications.md) for output contracts,
fixed sources, initialization, long-context evaluation and export.

## Historical injection-specific v2

The old entry point mixed harmful content with instruction attacks through the
`toxicity` label and harmful-request templates. It now requires an explicit
`--legacy-toxic-training` flag for historical reproduction. The historical candidate recipe uses `jailbreak_data_v2.py` and `train_v2.py`;
its ToxicChat source has a noncommercial license and is excluded from Vela training.

V2 keeps IDs `benign=0` and `jailbreak=1`, but `benign` means no annotated
instruction attack; it does not mean the requested content is safe. Content
risk belongs to the separate safety classifier. Data preparation verifies the
pinned toxic-chat and SALAD files, uses toxic-chat's `jailbreaking` annotation,
and pairs each SALAD attack with its original request. Connected question and
normalized-text groups are split before balancing or train-only augmentation.
The source-question split does not claim that all attack methods are unseen.

```bash
python src/training/model_classifier/prompt_guard_fine_tuning_lora/jailbreak_data_v2.py \
  --toxic-train /data/toxic-chat_annotation_train.csv \
  --salad /data/attack_enhanced_set.json --output-dir /data/jailbreak-v2
python src/training/model_classifier/prompt_guard_fine_tuning_lora/train_v2.py \
  --data-dir /data/jailbreak-v2 --output-dir /models/jailbreak-v2 \
  --max-length 2048 --epochs 5
```

Freeze a candidate using validation before evaluating external final tests.
Report injection precision/recall, content-risk versus injection four-quadrant
results, and quoted or negated attack false positives. Do not report toxicity
accuracy as injection accuracy, or infer long-context quality from the 32K
backbone name. Changed merged weights require newly exported inference graphs.
