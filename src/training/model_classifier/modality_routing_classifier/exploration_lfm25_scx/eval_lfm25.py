"""Evaluate the fine-tuned LFM2.5-Encoder (LoRA + classifier head) on the modality
routing test split. Rebuilds the exact training-time wrapper: body loaded via
AutoModelForMaskedLM(...).lfm2 (NOT the broken documented AutoModel path), LoRA adapter
applied on top, masked-mean pooling, then the saved linear head."""
import json
import os
import sys
from collections import Counter

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForMaskedLM, AutoTokenizer
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_CLASSIFIER_DIR = _HERE.parent  # modality_routing_classifier/
_DATA_DIR = _CLASSIFIER_DIR / "exported_modality_routing_dataset"


MODEL_ID = "LiquidAI/LFM2.5-Encoder-350M"
RUN_DIR = os.environ.get("LFM25_OUTPUT_DIR", str(_HERE / "runs" / "lfm25_encoder_finetuned"))
DATA = str(_DATA_DIR / "test.jsonl")
LABELS = ["AR", "DIFFUSION", "BOTH"]

rows = [json.loads(l) for l in open(DATA) if l.strip()]
tokenizer = AutoTokenizer.from_pretrained(RUN_DIR, trust_remote_code=True)
body = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True).lfm2
body = PeftModel.from_pretrained(body, f"{RUN_DIR}/lora_adapter").to("cuda").eval()
head = nn.Linear(1024, len(LABELS))
head.load_state_dict(torch.load(f"{RUN_DIR}/classifier_head.pt", map_location="cpu"))
head = head.to("cuda").eval()

preds = []
BS = 32
with torch.no_grad():
    for s in range(0, len(rows), BS):
        texts = [r["text"] for r in rows[s : s + BS]]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt").to("cuda")
        hidden = body(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        preds.extend(LABELS[i] for i in head(pooled.float()).argmax(-1).tolist())

truth = [r["label_name"] for r in rows]
correct = sum(p == t for p, t in zip(preds, truth))
cm = Counter(zip(truth, preds))
print(f"MODEL: LFM2.5-Encoder-350M + LoRA (fine-tuned)")
print(f"ACCURACY: {correct / len(rows):.4f} ({correct}/{len(rows)})")
print("CONFUSION (rows=true, cols=pred, order AR/DIFFUSION/BOTH):")
for t in LABELS:
    print(f"  {t:10s}", [cm.get((t, p), 0) for p in LABELS])
for t in LABELS:
    tp = cm.get((t, t), 0)
    print(f"  {t:10s} precision={tp / max(sum(cm.get((x, t), 0) for x in LABELS), 1):.4f} "
          f"recall={tp / max(sum(cm.get((t, x), 0) for x in LABELS), 1):.4f}")
json.dump({"model": "lfm25", "accuracy": correct / len(rows), "preds": preds},
          open(_HERE / "lfm25_finetuned_preds.json", "w"))
