"""Latency of one short video request through a d3 package (one request at a time, after warm-up).

    python -m d25.vega.release.video_latency --package PKG --suite PT --device cuda:0 --out latency-video.json

A request is one Perception Test validation clip of about 10 seconds (1920 x 1080, 30 frames per second, 270 to 330
frames, so 20 frames are read) sent as a base64 MP4 data URL, as the server receives it, with one Choice question
(the clip's first multiple-choice question). The time is the device-synchronized wall time of ``prepare`` + ``run``
+ ``respond`` (what ``system_one`` does: data URL decoding, video decoding and frame sampling, the video processor,
prompt rendering and tokenization, the forward pass and the readout), also reported per phase. Clips are picked
deterministically (sorted by the SHA-256 of their id) and cycled; ``--warmup`` untimed requests come first.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path


def pick_clips(suite: Path, count: int) -> list[tuple[str, dict]]:
    annotations = json.loads(
        (suite / "annotations" / "mc_question_valid.json").read_text()
    )
    cuts = json.loads((suite / "cut_frame_mapping_valid.json").read_text())
    chosen = []
    for video_id in sorted(
        annotations, key=lambda v: hashlib.sha256(v.encode()).hexdigest()
    ):
        meta = annotations[video_id]["metadata"]
        if (
            meta.get("resolution") == [1080, 1920]
            and 29.5 <= meta.get("frame_rate", 0) <= 30.5
            and 270 <= meta.get("num_frames", 0) <= 330
            and int(cuts.get(video_id, -1)) <= 0
        ):
            chosen.append((video_id, annotations[video_id]["mc_question"][0]))
        if len(chosen) == count:
            break
    return chosen


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q / 100
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "median_ms": round(statistics.median(ordered), 1),
        "mean_ms": round(statistics.fmean(ordered), 1),
        "p80_ms": round(percentile(80), 1),
        "min_ms": round(ordered[0], 1),
        "max_ms": round(ordered[-1], 1),
        "timed": len(ordered),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--suite", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clips", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(args.package))
    import d3_runtime as rt

    clips = pick_clips(args.suite, args.clips)
    if len(clips) < args.clips:
        raise SystemExit(f"only {len(clips)} clips match")
    requests = []
    for video_id, q in clips:
        payload = (args.suite / "videos" / f"{video_id}.mp4").read_bytes()
        url = "data:video/mp4;base64," + base64.b64encode(payload).decode()
        question = {
            "type": "choice",
            "instructions": q["question"],
            "criteria": {"ABC"[i]: option for i, option in enumerate(q["options"])},
        }
        requests.append((video_id, url, {"q": question}, len(payload)))
    started = time.perf_counter()
    model = rt.D3.from_pretrained(args.package, device=args.device)
    load_seconds = time.perf_counter() - started
    times, decode, prepare, run, tokens, frames = [], [], [], [], [], []
    for i in range(args.warmup + args.timed):
        video_id, url, questions, _ = requests[i % len(requests)]
        model.synchronize()
        begin = time.perf_counter()
        prepared = model.prepare({}, questions, videos=[url])
        middle = time.perf_counter()
        probabilities, used = model.run(prepared)
        response = model.respond(prepared, probabilities, used)
        model.synchronize()
        end = time.perf_counter()
        if any("error" in a for a in response["answers"].values()):
            raise SystemExit(f"{video_id}: {response['answers']}")
        if i >= args.warmup:
            times.append((end - begin) * 1000)
            prepare.append((middle - begin) * 1000)
            run.append((end - middle) * 1000)
            tokens.append(used)
            frames.append(len(prepared.videos[0].frames))
            again = time.perf_counter()
            rt.load_video(url)
            decode.append((time.perf_counter() - again) * 1000)
    report = {
        "what": "one request = one ~10 s 1080p30 Perception Test clip (base64 MP4 data URL) + one Choice question, "
        "one request at a time after warm-up",
        "package": str(args.package),
        "load_seconds": round(load_seconds, 1),
        "clips": len(clips),
        "clip_bytes_median": statistics.median(r[3] for r in requests),
        "warmup_requests": args.warmup,
        **stats(times),
        "prepare_median_ms": round(statistics.median(prepare), 1),
        "run_median_ms": round(statistics.median(run), 1),
        "decode_median_ms_untimed": round(statistics.median(decode), 1),
        "frames_median": statistics.median(frames),
        "input_tokens_median": statistics.median(tokens),
        "runtime": model.runtime_info(),
        "video_contract": model.video_contract(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "median_ms",
                    "mean_ms",
                    "p80_ms",
                    "prepare_median_ms",
                    "run_median_ms",
                    "decode_median_ms_untimed",
                    "frames_median",
                    "input_tokens_median",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
