#!/usr/bin/env python3
"""
Step 4 of probe-2026-05-25-image-drift-isolation: corpus-wide Python reference.

Runs Python mmes encode_image on all 20 fixtures in
docs/probe-2026-05-20-calibration/fixtures/, deterministic fp32 on CPU,
saves the resulting 20-by-384 embedding matrix to
python_corpus_embeddings.npy plus python_corpus_index.json (image names
in row order).

Step 5 (the Go-side corpus + diff) consumes these to validate that today's
preprocessing fix matches Python across the entire corpus, not just the
passport image used in step 1.

Run:
  ~/vllm-semantic-router-multimodal-testing/.venv/bin/python3 step4_extract_python_corpus.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
    AutoModel,
    AutoTokenizer,
    SiglipModel,
    SiglipProcessor,
)

logging.basicConfig(
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("step4")

PROBE_DIR = Path(__file__).parent.resolve()
FIXTURE_DIR = PROBE_DIR.parent / "probe-2026-05-20-calibration" / "fixtures"
DEVICE = torch.device("cpu")


class MultiModalEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.image_processor = SiglipProcessor.from_pretrained(
            "google/siglip-base-patch16-512"
        )
        self.image_encoder = SiglipModel.from_pretrained(
            "google/siglip-base-patch16-512"
        ).vision_model
        self.image_proj = nn.Linear(768, 384)

    def encode_image(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.image_encoder(**inputs)
        embeddings = outputs.pooler_output
        embeddings = self.image_proj(embeddings)
        return F.normalize(embeddings, p=2, dim=-1)


def load_weights(model):
    ckpt_path = hf_hub_download(
        repo_id="llm-semantic-router/multi-modal-embed-small",
        filename="model.pt",
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    image_sd = {
        k.replace("image_encoder.vision_encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("image_encoder.vision_encoder.")
    }
    model.image_encoder.load_state_dict(image_sd, strict=False)
    proj_sd = {
        k.replace("image_encoder.projection.", ""): v
        for k, v in state_dict.items()
        if k.startswith("image_encoder.projection.")
    }
    model.image_proj.load_state_dict(proj_sd, strict=False)


def main():
    log.info(f"Fixture dir: {FIXTURE_DIR}")
    fixtures = sorted(
        p
        for p in FIXTURE_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    log.info(f"Found {len(fixtures)} fixtures")

    t0 = time.time()
    model = MultiModalEmbedder()
    load_weights(model)
    model = model.to(DEVICE).eval()
    log.info(f"Model ready in {time.time() - t0:.1f}s")

    names = []
    embeddings = []
    for path in fixtures:
        img = Image.open(path).convert("RGB")
        with torch.no_grad():
            emb = model.encode_image(img)[0].cpu().numpy().astype(np.float32)
        names.append(path.name)
        embeddings.append(emb)
        log.info(f"  {path.name:50s} norm={float(np.linalg.norm(emb)):.6f}")

    matrix = np.stack(embeddings, axis=0)  # [N, 384]
    out_npy = PROBE_DIR / "python_corpus_embeddings.npy"
    out_idx = PROBE_DIR / "python_corpus_index.json"
    np.save(out_npy, matrix)
    with open(out_idx, "w") as f:
        json.dump({"images": names, "shape": list(matrix.shape)}, f, indent=2)
    log.info(f"Wrote {out_npy} (shape={matrix.shape})")
    log.info(f"Wrote {out_idx}")


if __name__ == "__main__":
    main()
