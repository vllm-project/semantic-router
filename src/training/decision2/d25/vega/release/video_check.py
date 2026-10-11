"""Video checks of a d3 package on one GPU: decoding, prompt expansion, forward pass, server and AutoModel.

    python -m d25.vega.release.video_check --package PKG --device cuda:0 --out video-check.json [--clip clip.mp4]

- decoding: a path, a data URL and the bytes of the same file give identical frames and frame numbers; frame
  arrays and ``Video`` objects pass through;
- prompt expansion: the runtime's video prompt (placeholders expanded with the request's processed videos) is
  token for token what the checkpoint's own processor produces for the same text and frames;
- forward pass: the runtime's video path against the processor's own call and the stock ``Qwen3_5Model.forward``,
  one question at a time (identical); with several questions the runtime computes the vision features once per
  request while the stock forward runs every copy through the vision tower (reported, small differences); the
  fused fast path against the plain path (identical);
- requests: one video with several questions, a video with images, two videos, the per-request video token
  budget, strict (server) loading, the HTTP server and ``AutoModel.from_pretrained(..., trust_remote_code=True)``.

``--clip`` writes the synthetic example clip (a blue square moves right, stops and turns green) to that path; the
card's example video is this clip.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

QUESTIONS = {
    "direction": {
        "type": "choice",
        "instructions": "Which way does the square move?",
        "criteria": {
            "right": "From left to right",
            "left": "From right to left",
            "up": "Upward",
            "still": "It does not move",
        },
    },
    "color_change": {"type": "noul", "instructions": "Does the square change color?"},
    "count": {
        "type": "choice",
        "instructions": "How many squares are there?",
        "criteria": {"one": None, "two": None, "three": None},
    },
}


def make_clip(path: Path, seconds: float = 4.0, fps: int = 8, size=(640, 360)) -> Path:
    """A blue square moves from left to right for 3 s, then stops and turns green (MPEG-4, OpenCV)."""
    import cv2
    import numpy as np

    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write MPEG-4 video here")
    frames = int(seconds * fps)
    side = 80
    for i in range(frames):
        t = i / fps
        frame = np.full((height, width, 3), 235, dtype=np.uint8)
        progress = min(t / 3.0, 1.0)
        x = int(40 + progress * (width - side - 80))
        y = (height - side) // 2
        color = (
            (60, 170, 60) if t >= 3.0 else (210, 110, 40)
        )  # BGR: green after 3 s, blue before
        frame[y : y + side, x : x + side] = color
        writer.write(frame)
    writer.release()
    return path


def values(answer: dict[str, Any]) -> list[float]:
    """An answer's probabilities in option order (a Yes / No answer is [p(no), p(yes)])."""
    if answer.get("type") == "noul":
        return [1.0 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def max_dp(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(abs(x - y) for key in a for x, y in zip(values(a[key]), values(b[key])))


def stock_probabilities(model, rt, state, questions, video) -> dict[str, list[float]]:
    """The same request through the processor's own call and ``Qwen3_5Model.forward`` (no shortcuts)."""
    torch = model.torch
    keys, texts, counts = [], [], []
    for key, question in questions.items():
        normalized = rt.normalize_question(question)
        keys.append(key)
        texts.append(model.video_text(state, normalized.rendered, 0, 1))
        counts.append(len(normalized.keys))
    encoded = model.processor(
        text=texts,
        videos=[video.frames for _ in texts],
        video_metadata=[video.metadata() for _ in texts],
        do_sample_frames=False,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in encoded.items() if hasattr(v, "to")}
    with torch.inference_mode(), rt.sdpa_backends(model.device):
        hidden = model.backbone(**inputs, use_cache=False).last_hidden_state[:, -1]
        logits = hidden.float() @ model.readout.T
        limit = torch.as_tensor(counts, device=model.device)[:, None]
        invalid = torch.arange(rt.MAX_OPTIONS, device=model.device)[None] >= limit
        probs = (
            logits.masked_fill(invalid, float("-inf")) / model.temperature
        ).softmax(-1)
    return {k: p[:c] for k, p, c in zip(keys, probs.cpu().tolist(), counts)}, encoded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clip", type=Path, default=Path("/tmp/d3-example-video.mp4"))
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("--no-automodel", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(args.package))
    import d3_runtime as rt
    import numpy as np

    report: dict[str, Any] = {"package": str(args.package), "checks": {}}
    checks = report["checks"]

    def check(name: str, ok: bool, **detail: Any) -> None:
        checks[name] = {"ok": bool(ok), **detail}
        print(
            name,
            "OK" if ok else "FAIL",
            json.dumps(detail, default=str)[:400],
            flush=True,
        )

    clip = make_clip(args.clip)
    payload = clip.read_bytes()
    data_url = "data:video/mp4;base64," + base64.b64encode(payload).decode()
    report["clip"] = {"path": str(clip), "bytes": len(payload)}

    # Decoding
    from_path = rt.load_video(str(clip))
    from_url = rt.load_video(data_url)
    strict = rt.load_video(data_url, strict=True)
    same = (
        np.array_equal(from_path.frames, from_url.frames)
        and from_path.indices == from_url.indices
        and np.array_equal(from_path.frames, strict.frames)
    )
    check(
        "decode_path_equals_data_url",
        same,
        frames=list(from_path.frames.shape),
        indices=from_path.indices,
        fps=from_path.fps,
        total=from_path.total_frames,
    )
    frames_list = [frame for frame in from_path.frames]
    as_frames = rt.load_video(frames_list)
    check(
        "frame_arrays",
        as_frames.frames.shape == from_path.frames.shape
        and as_frames.fps == rt.VIDEO_FPS,
        frames=list(as_frames.frames.shape),
    )
    rejected = []
    for bad in (str(clip), "data:video/mp4;base64,!!", frames_list):
        try:
            rt.load_video(bad, strict=True)
        except ValueError as exc:
            rejected.append(str(exc)[:80])
    check("strict_rejects", len(rejected) == 3, messages=rejected)

    # Model
    started = time.perf_counter()
    model = rt.D3.from_pretrained(args.package, device=args.device)
    report["load_seconds"] = round(time.perf_counter() - started, 1)
    fast = model.fast_report()
    check(
        "fast_path_active",
        fast.get("active") is True and fast.get("fused_layers", 0) > 0,
        fast={k: fast.get(k) for k in ("active", "fused_layers", "reason")},
    )
    check(
        "video_supported",
        model.video_unavailable is None,
        contract=model.video_contract(),
    )

    state = {}
    video = model.load_videos([str(clip)])[0]
    prepared = model.prepare(state, QUESTIONS, videos=[video])
    texts = [prepared.texts[k] for k in prepared.runnable]
    image_texts: list[str] = []
    expanded = [
        model.processor.get_text_with_replacements(
            [t], image_texts, prepared.video_texts
        )[0][0]
        for t in texts
    ]
    ours = model.processor.tokenizer(expanded, padding=True, return_tensors="pt")
    stock, encoded = stock_probabilities(model, rt, state, QUESTIONS, video)
    tokens_equal = bool(
        ours["input_ids"].shape == encoded["input_ids"].shape
        and (ours["input_ids"] == encoded["input_ids"]).all()
        and (ours["attention_mask"] == encoded["attention_mask"]).all()
    )
    check(
        "prompt_tokens_equal_processor",
        tokens_equal
        and max(prepared.lengths.values()) == encoded["input_ids"].shape[1],
        width=int(encoded["input_ids"].shape[1]),
        planned=prepared.lengths,
        grid=encoded["video_grid_thw"].tolist(),
    )
    answer = model.system_one(state=state, questions=QUESTIONS, videos=[str(clip)])
    report["answer_fused"] = answer
    probs_fused = {k: values(v) for k, v in answer["answers"].items()}
    saved = model.fast
    model.fast = None
    try:
        plain = model.system_one(state=state, questions=QUESTIONS, videos=[str(clip)])
    finally:
        model.fast = saved
    report["answer_plain"] = plain
    probs_plain = {k: values(v) for k, v in plain["answers"].items()}

    def dp(a, b):
        return max(abs(x - y) for k in a for x, y in zip(a[k], b[k]))

    def same_top(a, b):
        return all(int(np.argmax(a[k])) == int(np.argmax(b[k])) for k in a)

    singles = {}
    saved = model.fast
    model.fast = None
    try:
        for key, question in QUESTIONS.items():
            one = model.system_one(
                state=state, questions={key: question}, videos=[video]
            )
            reference, _ = stock_probabilities(model, rt, state, {key: question}, video)
            singles[key] = max(
                abs(x - y) for x, y in zip(values(one["answers"][key]), reference[key])
            )
    finally:
        model.fast = saved
    check(
        "plain_path_equals_stock_forward",
        max(singles.values()) <= 1e-6,
        max_abs_dp_one_question=singles,
        batch_of_three_max_abs_dp=dp(probs_plain, stock),
        batch_of_three_same_argmax=same_top(probs_plain, stock),
    )
    check(
        "fused_path_equals_plain",
        dp(probs_fused, probs_plain) == 0.0,
        max_abs_dp=dp(probs_fused, probs_plain),
    )
    again = model.system_one(state=state, questions=QUESTIONS, videos=[data_url])
    check(
        "data_url_answers_equal_path",
        max_dp(answer["answers"], again["answers"]) == 0.0,
        max_abs_dp=max_dp(answer["answers"], again["answers"]),
    )
    from PIL import Image

    image = Image.new("RGB", (640, 480), (200, 40, 40))
    mixed = model.system_one(
        state="A photo and a clip.",
        questions={"q": {"type": "noul", "instructions": "Is the photo red?"}},
        images=[image],
        videos=[str(clip)],
    )
    two = model.system_one(
        state={},
        questions={
            "q": {"type": "noul", "instructions": "Are the two videos the same?"}
        },
        videos=[str(clip), data_url],
    )
    check(
        "mixed_and_two_videos",
        all(
            "error" not in a
            for a in (*mixed["answers"].values(), *two["answers"].values())
        ),
        mixed=mixed["answers"],
        two=two["answers"],
        usage=[mixed["usage"], two["usage"]],
    )
    big = rt.Video(
        np.zeros((32, 1080, 1920, 3), dtype=np.uint8), 2.0, list(range(32)), 32
    )
    try:
        model.prepare({}, {"q": {"type": "noul"}}, videos=[big] * 6)
        budget = None
    except ValueError as exc:
        budget = str(exc)
    check("token_budget", budget is not None and "tokens" in budget, message=budget)
    times = []
    for _ in range(5):
        model.synchronize()
        begin = time.perf_counter()
        model.system_one(
            state=state, questions={"q": QUESTIONS["direction"]}, videos=[data_url]
        )
        model.synchronize()
        times.append((time.perf_counter() - begin) * 1000)
    report["quick_latency_ms"] = [round(t, 1) for t in times]
    report["usage"] = answer["usage"]
    report["fast_after"] = model.fast_report()
    import gc

    import torch

    # One model on the GPU at a time (the 27B one does not fit three times with the loader's warm-up block).
    saved = model = prepared = None
    gc.collect()
    torch.cuda.empty_cache()

    if not args.no_server:
        from fastapi.testclient import TestClient

        import d3_server

        server_args = d3_server.argparse.Namespace(
            model=str(args.package),
            revision=None,
            device=args.device,
            batch_size=rt.DEFAULT_BATCH_SIZE,
            verify="fast",
            name=None,
            no_warmup=True,
        )
        with TestClient(d3_server.build_app(server_args)) as client:
            health = client.get("/health").json()
            models = client.get("/v1/models").json()
            ok = client.post(
                "/v1/systemone",
                json={"state": state, "questions": QUESTIONS, "videos": [data_url]},
            )
            bad = client.post(
                "/v1/systemone",
                json={"state": state, "questions": QUESTIONS, "videos": [str(clip)]},
            )
        served = ok.json() if ok.status_code == 200 else {}
        check(
            "server",
            ok.status_code == 200
            and bad.status_code == 422
            and "video" in health.get("modalities", [])
            and max_dp(served.get("answers", {}), answer["answers"]) == 0.0,
            status=[ok.status_code, bad.status_code],
            modalities=health.get("modalities"),
            video=models["models"][0].get("video"),
            bad=bad.json(),
        )
        gc.collect()
        torch.cuda.empty_cache()
    if not args.no_automodel:
        from transformers import AutoModel

        hf = AutoModel.from_pretrained(
            str(args.package), trust_remote_code=True, device=args.device
        )
        auto = hf.system_one(state=state, questions=QUESTIONS, videos=[str(clip)])
        check(
            "automodel",
            max_dp(auto["answers"], answer["answers"]) == 0.0
            and hf.runtime.fast_report().get("active") is True,
            max_abs_dp=max_dp(auto["answers"], answer["answers"]),
        )
    report["pass"] = all(c["ok"] for c in checks.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("PASS" if report["pass"] else "FAIL", flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
