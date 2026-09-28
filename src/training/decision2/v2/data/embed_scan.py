"""Semantic near-duplicate scan of candidate arms against protected inputs.

States are split into overlapping character windows, embedded with a pinned
embedding model (last-token pooling, L2-normalized) and compared by cosine
similarity against every protected window. Candidate groups whose best match
reaches ``--quarantine`` are quarantined; a fixed-seed sample of pairs in
``[--review, --quarantine)`` is written to a private review packet. This bounds
paraphrase-level reuse; it does not certify semantic independence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from v2.data.textnorm import normalize

WINDOW_CHARS = 1500
STRIDE_CHARS = 1000
CJK_WINDOW_CHARS = 480
CJK_STRIDE_CHARS = 320
MIN_CHARS = 40


def state_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _cjk_share(text: str) -> float:
    cjk = sum(
        1
        for char in text
        if "\u3040" <= char <= "\u30ff"
        or "\u3400" <= char <= "\u9fff"
        or "\uac00" <= char <= "\ud7af"
    )
    return cjk / max(len(text), 1)


def windows(text: str) -> list[str]:
    text = normalize(text)
    if len(text) < MIN_CHARS:
        return []
    size, stride = (
        (CJK_WINDOW_CHARS, CJK_STRIDE_CHARS)
        if _cjk_share(text) > 0.3
        else (WINDOW_CHARS, STRIDE_CHARS)
    )
    if len(text) <= size:
        return [text]
    starts = list(range(0, len(text) - size + 1, stride))
    if starts[-1] + size < len(text):
        starts.append(len(text) - size)
    return [text[start : start + size] for start in starts]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_units(manifest: Path) -> tuple[list[tuple[str, str]], list[str]]:
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    units: list[tuple[str, str]] = []
    texts: list[str] = []
    for entry in entries:
        path = Path(entry["path"])
        if _sha(path) != entry["sha256"]:
            raise ValueError(f"{entry['role']}: protected file changed")
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            for text in windows(state_text(row.get("state", ""))):
                units.append((entry["role"], row["id"]))
                texts.append(text)
    return units, texts


def candidate_units(paths: list[Path]) -> tuple[list[tuple[str, str, str]], list[str]]:
    units: list[tuple[str, str, str]] = []
    texts: list[str] = []
    for path in paths:
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            for text in windows(state_text(row["state"])):
                units.append((path.name, row["group_id"], row["id"]))
                texts.append(text)
    return units, texts


def embed(texts: list[str], model_path: Path, device: str, batch: int) -> Any:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = (
        AutoModel.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()
    )
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    out = torch.empty(
        (len(texts), model.config.hidden_size), dtype=torch.float16, device=device
    )
    with torch.inference_mode():
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            encoded = tokenizer(
                [texts[index] for index in ids],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            hidden = model(**encoded).last_hidden_state[:, -1]
            hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
            out[torch.tensor(ids, device=device)] = hidden.half()
    return out


def scan(
    candidates: list[Path],
    manifest: Path,
    model_path: Path,
    *,
    device: str,
    batch: int,
    quarantine: float,
    review: float,
    sample: int,
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    p_units, p_texts = protected_units(manifest)
    c_units, c_texts = candidate_units(candidates)
    p_emb = embed(p_texts, model_path, device, batch)
    c_emb = embed(c_texts, model_path, device, batch)
    best_score = torch.full((len(c_units),), -1.0, device=device)
    best_index = torch.zeros((len(c_units),), dtype=torch.long, device=device)
    for start in range(0, len(c_units), 4096):
        block = c_emb[start : start + 4096] @ p_emb.T
        score, index = block.float().max(dim=1)
        best_score[start : start + 4096] = score
        best_index[start : start + 4096] = index
    scores, indexes = best_score.tolist(), best_index.tolist()
    group_best: dict[tuple[str, str], tuple[float, int, int]] = {}
    for unit, (score, p_index) in enumerate(zip(scores, indexes)):
        key = c_units[unit][:2]
        if key not in group_best or score > group_best[key][0]:
            group_best[key] = (score, unit, p_index)
    quarantined, band = [], []
    for key, (score, unit, p_index) in sorted(group_best.items()):
        record = {
            "file": key[0],
            "group_id": key[1],
            "row_id": c_units[unit][2],
            "cosine": round(score, 4),
            "protected_role": p_units[p_index][0],
            "protected_id": p_units[p_index][1],
        }
        if score >= quarantine:
            quarantined.append(record)
        elif score >= review:
            band.append((record, unit, p_index))
    ranked = sorted(
        band,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[0]['file']}:{item[0]['group_id']}".encode()
        ).hexdigest(),
    )[:sample]
    packet = [
        {**record, "candidate_text": c_texts[unit], "protected_text": p_texts[p_index]}
        for record, unit, p_index in ranked
    ]
    by_file: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    for key, (score, _, p_index) in group_best.items():
        stats = by_file[key[0]]
        stats["groups"] += 1
        stats["quarantined_groups"] += int(score >= quarantine)
        stats["review_band_groups"] += int(review <= score < quarantine)
        if score >= review:
            stats["roles_" + p_units[p_index][0]] += 1
    public = {
        "schema": "decision2-embed-scan/v1",
        "model_path_name": model_path.name,
        "protected_manifest_sha256": _sha(manifest),
        "protected_windows": len(p_units),
        "candidate_windows": len(c_units),
        "window_chars": WINDOW_CHARS,
        "stride_chars": STRIDE_CHARS,
        "thresholds": {"quarantine": quarantine, "review": review},
        "by_file": {
            name: dict(sorted(stats.items())) for name, stats in sorted(by_file.items())
        },
        "review_sample_size": len(packet),
        "note": "Approximate semantic neighbours; not a certificate of independence.",
    }
    private = {"quarantined": quarantined, "review_packet": packet, "public": public}
    return private, public


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--protected-inventory", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--quarantine", type=float, default=0.93)
    parser.add_argument("--review", type=float, default=0.85)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", default="decision2-embed-scan-v1")
    parser.add_argument("--private-receipt", type=Path, required=True)
    parser.add_argument("--public-receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.private_receipt, args.public_receipt):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    private, public = scan(
        args.candidates,
        args.protected_inventory,
        args.model_path,
        device=args.device,
        batch=args.batch,
        quarantine=args.quarantine,
        review=args.review,
        sample=args.sample,
        seed=args.seed,
    )
    _write_new(args.private_receipt, private)
    _write_new(args.public_receipt, public)
    print(json.dumps(public, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
