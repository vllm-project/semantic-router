"""Reference: the same Perception Test multiple-choice video QA for the untrained Qwen base models.

    python -m d25.vega.release.video_eval_base run --suite PT --runtime PKG --base qwen3.5-0.8b=<dir> ... \
        --shard 0 --shards 8 --device cuda:0 --out OUT
    python -m d25.vega.release.video_eval merge --suite PT --out OUT

The video input is identical to the d3 evaluation (``video_eval``): the d3 runtime's ``load_video`` (2 frames per
second, at most 32 frames, the benchmark's cut) and the d3 video-processor settings (up to 0.2 MP per frame). A base
model answers the standard multiple-choice prompt (the question, then ``A. ...`` / ``B. ...`` / ``C. ...``, then
"Answer with the option's letter from the given choices directly."; thinking off) and its answer is the most probable
of the three letter tokens at the first generated position. Records use ``video_eval``'s format, so its ``merge``
scores them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from d25.vega.release.video_eval import CODES, load_suite, request

PROMPT = "{question}\n{options}\nAnswer with the option's letter from the given choices directly."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", required=True, type=Path)
    ap.add_argument(
        "--runtime",
        required=True,
        type=Path,
        help="a d3 package dir (its d3_runtime.py)",
    )
    ap.add_argument(
        "--base", action="append", required=True, help="name=checkpoint dir, repeatable"
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.runtime))
    import d3_runtime as rt
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    annotations, cuts = load_suite(args.suite)
    videos = sorted(annotations)[args.shard :: args.shards]
    if args.limit:
        videos = videos[: args.limit]
    out = args.out / f"shard-{args.shard:02d}"
    out.mkdir(parents=True, exist_ok=True)
    bases = [tuple(b.split("=", 1)) for b in args.base]
    done: dict[str, set[str]] = defaultdict(set)
    files, models = {}, {}
    for name, path in bases:
        target = out / f"{name}.jsonl"
        if target.exists():
            done[name] = {
                json.loads(line)["video"]
                for line in target.read_text().splitlines()
                if line.strip()
            }
        files[name] = target.open("a", encoding="utf-8")
        processor = AutoProcessor.from_pretrained(path)
        processor.tokenizer.padding_side = "left"
        rt.configure_video_processor(processor)
        model = AutoModelForImageTextToText.from_pretrained(
            path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": args.device},
        ).eval()
        rt.linearize_patch_embed(model)
        letters = [processor.tokenizer.convert_tokens_to_ids(c) for c in CODES]
        models[name] = (processor, model, letters)
        print(f"loaded {name} letters={letters}", flush=True)
    started = time.perf_counter()
    for number, video_id in enumerate(videos):
        todo = [name for name, _ in bases if video_id not in done[name]]
        if not todo:
            continue
        entry = annotations[video_id]
        questions, expected = request(entry)
        cut = int(cuts.get(video_id, -1))
        try:
            video = rt.load_video(
                str(args.suite / "videos" / f"{video_id}.mp4"),
                end_frame=cut if cut > 0 else None,
            )
        except ValueError as exc:
            for name in todo:
                files[name].write(
                    json.dumps(
                        {"video": video_id, "expected": expected, "error": str(exc)}
                    )
                    + "\n"
                )
            continue
        for name in todo:
            processor, model, letters = models[name]
            texts, keys = [], []
            for key, question in questions.items():
                options = "\n".join(
                    f"{code}. {text}" for code, text in question["criteria"].items()
                )
                content = [
                    {"type": "video"},
                    {
                        "type": "text",
                        "text": PROMPT.format(
                            question=question["instructions"], options=options
                        ),
                    },
                ]
                texts.append(
                    processor.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                keys.append(key)
            answers = {}
            begin = time.perf_counter()
            for start in range(0, len(texts), 8):
                chunk = texts[start : start + 8]
                inputs = processor(
                    text=chunk,
                    videos=[video.frames for _ in chunk],
                    video_metadata=[video.metadata() for _ in chunk],
                    do_sample_frames=False,
                    padding=True,
                    return_tensors="pt",
                ).to(args.device)
                with torch.inference_mode():
                    logits = (
                        model(**inputs, use_cache=False, logits_to_keep=1)
                        .logits[:, -1]
                        .float()
                    )
                probs = logits[:, letters].softmax(-1).cpu().tolist()
                for key, p in zip(keys[start : start + 8], probs):
                    answers[key] = (
                        CODES[max(range(3), key=p.__getitem__)],
                        [round(x, 6) for x in p],
                    )
            record = {
                "video": video_id,
                "cut": cut,
                "expected": expected,
                "ms": round((time.perf_counter() - begin) * 1000, 1),
                "frames": len(video.frames),
                "tokens": int(inputs["input_ids"].shape[1]),
                "answers": answers,
            }
            files[name].write(json.dumps(record) + "\n")
            files[name].flush()
        if number % 50 == 0:
            print(
                f"{number}/{len(videos)} {time.perf_counter() - started:.0f}s",
                flush=True,
            )
    for handle in files.values():
        handle.close()
    (out / "DONE-base").write_text("done\n")
    print("SHARD-DONE", args.shard, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
