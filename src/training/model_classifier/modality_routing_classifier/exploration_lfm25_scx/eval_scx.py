"""Evaluate an SCX Router (GLiClass decoder-KV) checkpoint on the modality-routing
test split, using the SAME GLiClassDataset/collator path as training so the label
formatting matches what the model saw. Usage: eval_scx.py <model_path_or_repo> <out_json>"""
import json
import sys
from collections import Counter

import torch
from gliclass import GLiClassModel
from gliclass.data_processing import AugmentationConfig, DataCollatorWithPadding, GLiClassDataset
from transformers import AutoTokenizer
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_CLASSIFIER_DIR = _HERE.parent  # modality_routing_classifier/
_DATA_DIR = _CLASSIFIER_DIR / "exported_modality_routing_dataset"


model_path, out_json = sys.argv[1], sys.argv[2]
DATA = str(_DATA_DIR / "test.jsonl")
LABELS = ["AR", "DIFFUSION", "BOTH"]
LABEL_DESCRIPTIONS = {
    "AR": "a text-only response",
    "DIFFUSION": "generating an image",
    "BOTH": "both a text explanation and an image",
}

rows = [json.loads(l) for l in open(DATA) if l.strip()]
examples = [{"text": r["text"], "all_labels": LABELS, "true_labels": [r["label_name"]]} for r in rows]

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GLiClassModel.from_pretrained(model_path).to("cuda").eval()

ds = GLiClassDataset(
    examples,
    tokenizer,
    AugmentationConfig(enabled=False),
    label2description={n: f"{n}: {d}" for n, d in LABEL_DESCRIPTIONS.items()},
    max_length=512,
    problem_type="multi_label_classification",
    architecture_type="decoder-kv",
    shuffle_labels=False,
)
collator = DataCollatorWithPadding(device="cuda:0", config=model.config)

preds = []
BS = 16
with torch.no_grad():
    for start in range(0, len(ds), BS):
        batch = collator([ds[i] for i in range(start, min(start + BS, len(ds)))])
        labels_text = batch.pop("labels_text")
        batch.pop("input_texts", None)
        batch.pop("labels", None)
        batch = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        logits = model(**batch).logits.float()
        for i in range(logits.shape[0]):
            n_lab = len(labels_text[i])
            preds.append(labels_text[i][int(logits[i, :n_lab].argmax())])

truth = [r["label_name"] for r in rows]
acc = sum(p == t for p, t in zip(preds, truth)) / len(rows)
cm = Counter(zip(truth, preds))
print(f"MODEL: {model_path}")
print(f"ACCURACY: {acc:.4f} ({sum(p == t for p, t in zip(preds, truth))}/{len(rows)})")
print("CONFUSION (rows=true, cols=pred, order AR/DIFFUSION/BOTH):")
for t in LABELS:
    print(f"  {t:10s}", [cm.get((t, p), 0) for p in LABELS])
for t in LABELS:
    tp = cm.get((t, t), 0)
    pred_n = sum(cm.get((x, t), 0) for x in LABELS)
    true_n = sum(cm.get((t, x), 0) for x in LABELS)
    print(f"  {t:10s} precision={tp / max(pred_n, 1):.4f} recall={tp / max(true_n, 1):.4f}")

json.dump({"model": model_path, "accuracy": acc, "preds": preds}, open(out_json, "w"))
