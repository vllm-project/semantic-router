"""Zero-shot multiple-choice video QA of d3 packages: Perception Test (validation split, CC-BY 4.0).

    python -m d25.vega.release.video_eval run --suite PT --package d3=PKG --package d3-lite=PKG2 \
        --shard 0 --shards 8 --device cuda:0 --out OUT
    python -m d25.vega.release.video_eval merge --suite PT --out OUT

``PT`` holds ``annotations/mc_question_valid.json``, ``cut_frame_mapping_valid.json`` and ``videos/<id>.mp4``
(google-deepmind/perception_test). One request per video carries all of its questions, each a Choice question
over the three options in the benchmark's order (``{"A": option 1, "B": option 2, "C": option 3}``, state ``{}``,
the vision board's request shape); the answer is the most probable option, scored by top-1 accuracy. Videos are
decoded once per request by the package runtime's own ``load_video`` (the defaults of a file input: 2 frames per
second, at most 32 frames, up to 0.2 MP per frame), cut where the benchmark's cut-frame mapping says so (the
test split ships videos cut there), and the decoded video goes to every package of the process. Shards are
deterministic (videos sorted by id, every ``shards``-th); ``run`` resumes from its results file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

CODES = "ABC"


def load_suite(root: Path) -> tuple[dict[str, Any], dict[str, int]]:
    annotations = json.loads(
        (root / "annotations" / "mc_question_valid.json").read_text()
    )
    cuts = json.loads((root / "cut_frame_mapping_valid.json").read_text())
    return annotations, cuts


def request(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    questions, expected = {}, {}
    for q in entry["mc_question"]:
        key = f"q{q['id']}"
        questions[key] = {
            "type": "choice",
            "instructions": q["question"],
            "criteria": {CODES[i]: option for i, option in enumerate(q["options"])},
        }
        expected[key] = CODES[q["answer_id"]]
    return questions, expected


def cmd_run(args) -> int:
    packages = [tuple(p.split("=", 1)) for p in args.package]
    sys.path.insert(0, str(Path(packages[0][1])))
    import d3_runtime as rt

    annotations, cuts = load_suite(args.suite)
    videos = sorted(annotations)[args.shard :: args.shards]
    if args.limit:
        videos = videos[: args.limit]
    out = args.out / f"shard-{args.shard:02d}"
    out.mkdir(parents=True, exist_ok=True)
    done: dict[str, set[str]] = defaultdict(set)
    files = {}
    for name, _ in packages:
        path = out / f"{name}.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    done[name].add(json.loads(line)["video"])
        files[name] = path.open("a", encoding="utf-8")
    models = {}
    for name, package in packages:
        started = time.perf_counter()
        models[name] = rt.D3.from_pretrained(package, device=args.device)
        fast = models[name].fast_report()
        print(
            f"loaded {name} in {time.perf_counter() - started:.0f}s fast={fast.get('active')} "
            f"fused={fast.get('fused_layers')}",
            flush=True,
        )
    decode_s, counts = 0.0, defaultdict(int)
    started = time.perf_counter()
    for number, video_id in enumerate(videos):
        todo = [name for name, _ in packages if video_id not in done[name]]
        if not todo:
            continue
        questions, expected = request(annotations[video_id])
        cut = int(cuts.get(video_id, -1))
        path = args.suite / "videos" / f"{video_id}.mp4"
        begin = time.perf_counter()
        try:
            video = rt.load_video(str(path), end_frame=cut if cut > 0 else None)
            error = None
        except ValueError as exc:
            video, error = None, str(exc)
        decode_s += time.perf_counter() - begin
        for name in todo:
            record: dict[str, Any] = {
                "video": video_id,
                "cut": cut,
                "expected": expected,
            }
            if video is None:
                record["error"] = error
            else:
                model = models[name]
                model.synchronize()
                begin = time.perf_counter()
                response = model.system_one(
                    state={}, questions=questions, videos=[video]
                )
                model.synchronize()
                record["ms"] = round((time.perf_counter() - begin) * 1000, 1)
                record["frames"] = len(video.frames)
                record["tokens"] = response["usage"]["input_tokens"]
                record["answers"] = {
                    k: (
                        (
                            a.get("choice"),
                            [round(p, 6) for p in a["probabilities"].values()],
                        )
                        if "error" not in a
                        else (None, a["error"])
                    )
                    for k, a in response["answers"].items()
                }
            files[name].write(json.dumps(record) + "\n")
            files[name].flush()
            counts[name] += 1
        if number % 50 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"{number}/{len(videos)} {elapsed:.0f}s decode {decode_s:.0f}s done {dict(counts)}",
                flush=True,
            )
    for handle in files.values():
        handle.close()
    (out / "DONE").write_text(
        json.dumps({"videos": len(videos), "decode_s": round(decode_s, 1)}) + "\n"
    )
    print("SHARD-DONE", args.shard, flush=True)
    return 0


def cmd_merge(args) -> int:
    annotations, _ = load_suite(args.suite)
    meta = {
        (video_id, f"q{q['id']}"): q
        for video_id, entry in annotations.items()
        for q in entry["mc_question"]
    }
    total_questions = len(meta)
    names = sorted({p.stem for p in args.out.glob("shard-*/*.jsonl")})
    summary: dict[str, Any] = {
        "benchmark": "Perception Test, multiple-choice video QA, validation split",
        "licence": "CC-BY 4.0 (google-deepmind/perception_test)",
        "questions": total_questions,
        "videos": len(annotations),
        "metric": "top-1 accuracy (%)",
        "models": {},
    }
    for name in names:
        rows = {}
        for path in sorted(args.out.glob(f"shard-*/{name}.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    rows[record["video"]] = record
        correct = answered = errors = 0
        by_area: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        ms, tokens = [], []
        for video_id, record in rows.items():
            if "answers" not in record:
                errors += len(record["expected"])
                continue
            ms.append(record["ms"])
            tokens.append(record["tokens"])
            for key, want in record["expected"].items():
                choice = record["answers"].get(key, (None,))[0]
                hit = int(choice == want)
                correct += hit
                answered += 1
                area = meta[(video_id, key)]["area"]
                by_area[area][0] += hit
                by_area[area][1] += 1
        complete = len(rows) == len(annotations) and answered == total_questions
        ms.sort()
        summary["models"][name] = {
            "accuracy": round(100 * correct / max(1, total_questions), 2),
            "correct": correct,
            "answered": answered,
            "errors": errors,
            "videos": len(rows),
            "complete": complete,
            "areas": {
                a: round(100 * c / n, 2) for a, (c, n) in sorted(by_area.items())
            },
            "request_ms_median": ms[len(ms) // 2] if ms else None,
            "input_tokens_median": sorted(tokens)[len(tokens) // 2] if tokens else None,
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--suite", required=True, type=Path)
    run.add_argument(
        "--package", action="append", required=True, help="name=package dir, repeatable"
    )
    run.add_argument("--shard", type=int, default=0)
    run.add_argument("--shards", type=int, default=1)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--out", required=True, type=Path)
    merge = sub.add_parser("merge")
    merge.add_argument("--suite", required=True, type=Path)
    merge.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
