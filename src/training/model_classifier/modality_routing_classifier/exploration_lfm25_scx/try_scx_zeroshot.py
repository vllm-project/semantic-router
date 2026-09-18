"""Quick curiosity probe: zero-shot SCX Router v0.1 on our modality-routing test set.
Not part of the formal decision-record pipeline -- exploratory only."""
import json
import sys

from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_CLASSIFIER_DIR = _HERE.parent  # modality_routing_classifier/
_DATA_DIR = _CLASSIFIER_DIR / "exported_modality_routing_dataset"


MODEL_ID = "scx-admin/scx-router-v0.1"
DATA = str(_DATA_DIR / "test.jsonl")

LABELS = ["AR", "DIFFUSION", "BOTH"]
LABEL_DESCRIPTIONS = {
    "AR": "a text-only response",
    "DIFFUSION": "generating an image",
    "BOTH": "both a text explanation and an image",
}

print("Loading SCX Router v0.1...", file=sys.stderr)
model = GLiClassModel.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model.to("cuda")
pipe = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type="multi-class", device="cuda", progress_bar=False
)

rows = [json.loads(l) for l in open(DATA) if l.strip()]
print(f"Loaded {len(rows)} test rows", file=sys.stderr)

subset = rows
correct = 0
label_texts = [f"{name}: {desc}" for name, desc in LABEL_DESCRIPTIONS.items()]
name_by_text = {t: n for n, t in zip(LABELS, label_texts)}

results = []
for row in subset:
    out = pipe(row["text"], label_texts, threshold=0.0)[0]
    pred_text = max(out, key=lambda x: x["score"])["label"]
    pred = name_by_text[pred_text]
    is_correct = pred == row["label_name"]
    correct += is_correct
    results.append({"text": row["text"][:80], "true": row["label_name"], "pred": pred, "correct": is_correct})

print(f"\nFull test set accuracy (n={len(subset)}): {correct}/{len(subset)} = {correct/len(subset):.4f}")

from collections import Counter
cm = Counter((r["true"], r["pred"]) for r in results)
print("\nConfusion matrix (true, pred) -> count:")
for true_label in LABELS:
    row = [cm.get((true_label, p), 0) for p in LABELS]
    print(f"  {true_label:10s} {row}  (cols: {LABELS})")

per_class_acc = {}
for label in LABELS:
    label_rows = [r for r in results if r["true"] == label]
    per_class_acc[label] = sum(r["correct"] for r in label_rows) / len(label_rows) if label_rows else None
print("\nPer-class recall:", per_class_acc)

with open(_HERE / "scx_zeroshot_results.json", "w") as f:
    json.dump({"accuracy": correct / len(subset), "confusion": {str(k): v for k, v in cm.items()},
               "per_class_recall": per_class_acc, "results": results}, f, indent=2)
