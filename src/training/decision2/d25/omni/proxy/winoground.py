"""Winoground proxy (GPU stage pending): generated image pairs for word-order-swapped caption pairs.

Pipeline (each stage resumable, outputs under ``$D25_OMNI_PROXY/_work/winoground/``):

1. ``pairs`` (CPU, now): deterministic caption pairs that use the same words in a different order
   and mean different things (Winoground's object, relation and attribute swaps).
2. ``generate`` (GPU): ``CANDIDATES`` images per caption with an Apache-2.0 text-to-image model
   (``Qwen/Qwen-Image`` by default; ``black-forest-labs/FLUX.1-schnell`` as the second generator).
3. ``verify`` (GPU): two permissive VLM judges from different families (Qwen3.5-397B-A17B,
   GLM-4.6V) read each candidate with both captions; a candidate is kept only when both prefer its
   own caption with probability >= 0.9, and a pair is kept only when both of its images survive.
4. ``build_item`` (CPU): two rows per kept pair (each image with the two captions, chance 0.5).

Judge-filtered items favour what the judges can see; using two judge families and a strict
threshold limits that, and the calibration absorbs a constant shift.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

BENCHMARK = "Winoground"
NAME = "winoground-proxy"
VERSION = "1"
CANDIDATES = 4
GENERATORS = {
    "qwen-image": "Qwen/Qwen-Image",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
}
JUDGES = {"qwen3.5-397b": "Qwen/Qwen3.5-397B-A17B", "glm-4.6v": "zai-org/GLM-4.6V"}
THRESHOLD = 0.9

AGENTS = [
    "a dog",
    "a cat",
    "a child",
    "an old man",
    "a woman",
    "a horse",
    "a bird",
    "a robot",
    "a chef",
    "a firefighter",
    "a monkey",
    "a goat",
    "a teenager",
    "a police officer",
    "a baby",
    "a puppy",
]
ACTIONS = [
    "chasing",
    "pushing",
    "hugging",
    "feeding",
    "painting",
    "pulling",
    "carrying",
    "watching",
    "following",
    "photographing",
    "splashing",
    "tickling",
    "lifting",
    "drawing",
    "kicking a ball to",
    "handing a gift to",
]
OBJECTS = [
    "mug",
    "book",
    "ball",
    "box",
    "vase",
    "lamp",
    "chair",
    "bag",
    "plate",
    "bottle",
    "hat",
    "pillow",
]
COLORS = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"]
RELATIONS = ["on top of", "inside", "in front of", "behind", "under", "to the left of"]
SCENES = [
    "in a park",
    "in a kitchen",
    "on a beach",
    "in a living room",
    "on a street",
    "in a garden",
    "in a classroom",
]


def caption_pairs(seed: int, n: int) -> list[dict[str, Any]]:
    """Distinct swapped caption pairs (same multiset of words, different meaning)."""
    r = random.Random(f"{NAME}:{seed}")
    out, seen = [], set()
    kinds, weights = ["agent", "attribute", "relation", "count"], [0.4, 0.25, 0.25, 0.1]
    for _ in range(100 * n):
        if len(out) == n:
            break
        kind = r.choices(kinds, weights)[0]
        scene = r.choice(SCENES)
        if kind == "agent":
            a, b = r.sample(AGENTS, 2)
            act = r.choice(ACTIONS)
            c0, c1 = f"{a} {act} {b} {scene}", f"{b} {act} {a} {scene}"
        elif kind == "attribute":
            x, y = r.sample(COLORS, 2)
            o1, o2 = r.sample(OBJECTS, 2)
            art = lambda w: "an" if w[0] in "aeiou" else "a"  # noqa: E731
            c0 = f"{art(x)} {x} {o1} and {art(y)} {y} {o2} on a table"
            c1 = f"{art(y)} {y} {o1} and {art(x)} {x} {o2} on a table"
        elif kind == "relation":
            o1, o2 = r.sample(OBJECTS, 2)
            rel = r.choice(RELATIONS)
            c0, c1 = f"a {o1} {rel} a {o2}", f"a {o2} {rel} a {o1}"
        else:
            o1, o2 = r.sample(
                [
                    "dogs",
                    "cats",
                    "apples",
                    "oranges",
                    "cups",
                    "books",
                    "birds",
                    "candles",
                    "chairs",
                    "balloons",
                ],
                2,
            )
            m, k = r.sample(["one", "two", "three", "four"], 2)
            c0, c1 = f"{m} {o1} and {k} {o2} {scene}", f"{k} {o1} and {m} {o2} {scene}"
        key = tuple(sorted((c0, c1)))
        if key in seen:
            continue
        seen.add(key)
        out.append({"pair": f"p{len(out):04d}", "kind": kind, "captions": [c0, c1]})
    return out


def generate(
    pairs: list[dict[str, Any]],
    out: Path,
    generator: str = "qwen-image",
    device: str = "cuda",
) -> None:
    """GPU: ``CANDIDATES`` images per caption, saved as PNG with a JSON index (resumable)."""
    import torch
    from diffusers import DiffusionPipeline

    from d25.omni.proxy.rows import item_seed

    pipe = DiffusionPipeline.from_pretrained(
        GENERATORS[generator], torch_dtype=torch.bfloat16
    ).to(device)
    out.mkdir(parents=True, exist_ok=True)
    for p in pairs:
        for c, caption in enumerate(p["captions"]):
            for k in range(CANDIDATES):
                target = out / f"{p['pair']}_{c}_{generator}_{k}.png"
                if target.exists():
                    continue
                g = torch.Generator(device).manual_seed(
                    item_seed(NAME, p["pair"], c, k, generator) % (2**31)
                )
                prompt = f"A realistic photograph of {caption}."
                image = pipe(
                    prompt=prompt,
                    generator=g,
                    num_inference_steps=4 if generator == "flux-schnell" else 30,
                    height=1024,
                    width=1024,
                ).images[0]
                image.save(target)


def verify(pairs: list[dict[str, Any]], images: Path, out: Path, judge: str) -> None:
    """GPU: judge probability that each candidate shows its own caption rather than the swapped one.

    Each candidate is scored with both option orders (A/B swapped) through vLLM single-token
    log-probabilities of the option letters, thinking off; results go to ``out/<judge>.jsonl``.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=JUDGES[judge],
        tensor_parallel_size=8,
        limit_mm_per_prompt={"image": 1},
        max_model_len=8192,
    )
    params = SamplingParams(max_tokens=1, logprobs=20, temperature=0.0)
    done = set()
    path = out / f"{judge}.jsonl"
    if path.exists():
        done = {json.loads(line)["file"] for line in path.read_text().splitlines()}
    out.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        for p in pairs:
            for file in sorted(images.glob(f"{p['pair']}_*.png")):
                if file.name in done:
                    continue
                own = int(file.name.split("_")[1])
                probs = []
                for order in ((0, 1), (1, 0)):
                    text = (
                        f"Which caption describes the image?\nA: {p['captions'][order[0]]}\nB: {p['captions'][order[1]]}\n"
                        "Answer with A or B."
                    )
                    msg = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": file.resolve().as_uri()},
                                },
                                {"type": "text", "text": text},
                            ],
                        }
                    ]
                    result = llm.chat(
                        msg, params, chat_template_kwargs={"enable_thinking": False}
                    )[0].outputs[0]
                    top = {
                        lp.decoded_token.strip(): lp.logprob
                        for lp in result.logprobs[0].values()
                    }
                    pa, pb = math.exp(top.get("A", -50)), math.exp(top.get("B", -50))
                    p_own = (pa if order[0] == own else pb) / max(pa + pb, 1e-12)
                    probs.append(p_own)
                handle.write(
                    json.dumps(
                        {
                            "file": file.name,
                            "pair": p["pair"],
                            "own": own,
                            "p_own": sum(probs) / 2,
                        }
                    )
                    + "\n"
                )


def prepare(work: Path, seed: int) -> dict[str, Any]:
    """Kept pairs from the verified generations; raises until the GPU stages have run."""
    root = work
    pairs = caption_pairs(seed, 300)
    (root / "pairs.json").write_text(json.dumps(pairs, indent=1))
    scores: dict[str, dict[str, float]] = {}
    for judge in JUDGES:
        path = root / "verify" / f"{judge}.jsonl"
        if not path.exists():
            raise SystemExit(
                f"winoground: GPU stages pending (no {path}); pairs written to {root / 'pairs.json'}"
            )
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            scores.setdefault(rec["file"], {})[judge] = rec["p_own"]
    kept = []
    for p in pairs:
        best = []
        for c in (0, 1):
            files = sorted(
                f
                for f, s in scores.items()
                if f.startswith(f"{p['pair']}_{c}_")
                and len(s) == len(JUDGES)
                and min(s.values()) >= THRESHOLD
            )
            best.append(files[0] if files else None)
        if all(best):
            kept.append(dict(p, files=best))
    return {"pairs": kept[:200], "images": str(root / "images")}


def size(context: Any) -> int:
    return 2 * len(context["pairs"])


def build_item(index: int, seed: int, context: Any = None) -> dict[str, Any]:
    from d25.omni.proxy.rows import LETTERS, rng

    pair = context["pairs"][index // 2]
    own = index % 2
    payload = (Path(context["images"]) / pair["files"][own]).read_bytes()
    order = [0, 1]
    rng(NAME, seed, index).shuffle(order)
    criteria = {LETTERS[i]: pair["captions"][c] for i, c in enumerate(order)}
    return {
        "item_id": f"{index:05d}",
        "subtask": pair["kind"],
        "payloads": [(payload, "png")],
        "instructions": "Which caption matches the image?",
        "criteria": criteria,
        "answer": LETTERS[order.index(own)],
        "provenance": [
            {
                "source": "generated",
                "generator": pair["files"][own].split("_")[2],
                "pair": pair["pair"],
                "licence": "generated with an Apache-2.0 model",
            }
        ],
    }
