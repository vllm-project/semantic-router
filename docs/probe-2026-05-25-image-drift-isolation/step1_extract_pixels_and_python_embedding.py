#!/usr/bin/env python3
"""
Step 1 of probe-2026-05-25-image-drift-isolation.

Produces three artifacts in this directory:
  - pixels_pil_chw_512_01.bin : raw float32, [3, 512, 512] CHW, range [0, 1].
                                What SiglipProcessor would feed the model AFTER
                                bicubic+antialias resize but BEFORE the
                                (x - 0.5) / 0.5 SigLIP normalization. This is
                                the exact shape candle-binding's Rust FFI
                                MultiModalEncodeImage expects (RGB, [0, 1]).
  - embedding_python.npy      : float32 [384], Python mmes encode_image result
                                on the raw image (PIL preprocessing internal to
                                SiglipProcessor, then mmes forward).
  - metadata.json             : metadata (paths, hashes, PIL/torch versions)

Reproduces the production-path mmes architecture per
docs/probe-2026-05-20-calibration/probe_mmes_20img.py.

Run:
  ~/vllm-semantic-router-multimodal-testing/.venv/bin/python3 step1_extract_pixels_and_python_embedding.py
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from torch.nn import functional
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
log = logging.getLogger("step1")

PROBE_DIR = Path(__file__).parent.resolve()
PASSPORT = (
    PROBE_DIR.parent
    / "probe-2026-05-20-calibration"
    / "fixtures"
    / "inrule_identifier_passport.jpg"
)

# Force CPU for deterministic fp32. MPS introduces its own drift class.
DEVICE = torch.device("cpu")
DTYPE = torch.float32


class MultiModalEmbedder(nn.Module):
    """Production-path mmes architecture (verbatim from probe_mmes_20img.py)."""

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
        return functional.normalize(embeddings, p=2, dim=-1)


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
    log.info(
        f"Loaded {len(image_sd)} image_encoder keys, {len(proj_sd)} projection keys"
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    log.info(f"Probe dir: {PROBE_DIR}")
    log.info(f"Passport: {PASSPORT}")
    log.info(f"Device: {DEVICE}, dtype: {DTYPE}")

    if not PASSPORT.exists():
        raise FileNotFoundError(f"Passport fixture not found: {PASSPORT}")

    img = Image.open(PASSPORT).convert("RGB")
    log.info(f"Source image: size={img.size}, mode={img.mode}")

    # ===== Artifact 1: PIL-preprocessed pixels in [0, 1] CHW =====
    # Use the SAME processor the Python reference uses, then undo its
    # (x - 0.5) / 0.5 normalization so the result matches what Rust FFI expects.
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-512")
    proc_out = processor(images=img, return_tensors="pt")
    px_norm = proc_out["pixel_values"][0]  # [3, 512, 512] in [-1, 1]
    log.info(
        f"SiglipProcessor output: shape={tuple(px_norm.shape)}, "
        f"range=[{px_norm.min().item():.4f}, {px_norm.max().item():.4f}]"
    )
    # Invert normalization: x_norm = (x_01 - 0.5) / 0.5 -> x_01 = x_norm * 0.5 + 0.5
    px_01 = (px_norm * 0.5 + 0.5).clamp(0.0, 1.0).contiguous().to(torch.float32)
    log.info(
        f"Recovered [0,1] pixels: shape={tuple(px_01.shape)}, "
        f"range=[{px_01.min().item():.6f}, {px_01.max().item():.6f}]"
    )

    pixels_path = PROBE_DIR / "pixels_pil_chw_512_01.bin"
    px_01.numpy().astype(np.float32).tofile(pixels_path)
    log.info(f"Wrote {pixels_path} ({pixels_path.stat().st_size} bytes)")

    # ===== Artifact 2: Python mmes embedding for the same image =====
    t0 = time.time()
    model = MultiModalEmbedder()
    load_weights(model)
    model = model.to(DEVICE).eval()
    log.info(f"Model ready in {time.time() - t0:.1f}s")

    with torch.no_grad():
        emb = model.encode_image(img)[0].cpu().numpy().astype(np.float32)
    log.info(
        f"Python embedding: shape={emb.shape}, "
        f"norm={float(np.linalg.norm(emb)):.6f}, "
        f"first5={emb[:5].tolist()}"
    )

    emb_path = PROBE_DIR / "embedding_python.npy"
    np.save(emb_path, emb)
    log.info(f"Wrote {emb_path}")

    # ===== Artifact 3: metadata =====
    meta = {
        "probe_date": "2026-05-25",
        "passport_path": str(PASSPORT),
        "passport_sha256": sha256_file(PASSPORT),
        "image_size_original": list(img.size),
        "image_size_resized": [512, 512],
        "torch_version": torch.__version__,
        "siglip_repo": "google/siglip-base-patch16-512",
        "mmes_repo": "llm-semantic-router/multi-modal-embed-small",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "pixels_dtype": "float32",
        "pixels_layout": "CHW",
        "pixels_range": "[0, 1]",
        "pixels_shape": [3, 512, 512],
        "embedding_norm": float(np.linalg.norm(emb)),
        "embedding_first5": emb[:5].tolist(),
    }
    meta_path = PROBE_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Wrote {meta_path}")

    print()
    print("Step 1 done. Now run step 2 (Go test) then step 3 (compare).")


if __name__ == "__main__":
    main()
