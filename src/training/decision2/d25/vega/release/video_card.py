"""The v3.1.0 card of a d3 package: video input, a video Quickstart example, measured video results and speed.

    python -m d25.vega.release.video_card --package <m>-v310 --out <m>-final --repo vllm-sr/<m> \
        --video-results summary.json --latency latency-video.json --example example-video.mp4 \
        [--text-rows rows.json] [--public-index 65.45]

Edits ``README.md`` of the package in place of its current wording (every edit must match exactly once): the
``video`` tag, the intro and the Inputs row name videos, a highlight with the Perception Test score, the speed line
adds the median of one request with a 10-second video, the Quickstart installs OpenCV and shows a video request
(``assets/example-video.mp4``), the input note describes how videos are read, and the evaluation gains a Perception
Test table (internal evaluation, under the card's existing label). ``--text-rows`` replaces whole text-table rows
(``{"old row": "new row"}``) and ``--public-index`` the public-suite line. Everything else is hard-linked from the
package; ``MODEL_MANIFEST.json`` gets the changed and added files in the same build, and the result is verified with
the package's own runtime.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

from d25.vega.release.examples import EXAMPLE_VIDEO, QUICKSTART_VIDEO

AREAS = ("memory", "abstraction", "physics", "semantics")
FORBIDDEN = re.compile(
    r"nvidia|\brtx\b|cuda|pending|hf jobs|teacher|distill|\bvega\b|\bomni\b|\bd25\b|decision 2\.5",
    re.I,
)


def py_literal(value, indent: int = 0) -> str:
    """A Python literal laid out like ``json.dumps(..., indent=4)`` (``None`` instead of ``null``), as the cards' code."""
    pad, inner = " " * indent, " " * (indent + 4)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f"{inner}{json.dumps(k)}: {py_literal(v, indent + 4)}"
            for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [f"{inner}{py_literal(v, indent + 4)}" for v in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    return json.dumps(value, ensure_ascii=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_block(repo: str) -> str:
    return "\n".join(
        [
            "# Text and a video (or several)",
            f'clip = hf_hub_download("{repo}", "{EXAMPLE_VIDEO}")',
            "result = model.system_one(",
            f"    state={json.dumps(QUICKSTART_VIDEO['state'], ensure_ascii=False)},",
            "    videos=[clip],  # local paths, http(s) URLs, base64 data URLs or frame arrays",
            f"    questions={py_literal(QUICKSTART_VIDEO['questions'], 4)},",
            ")",
            'print(json.dumps(result["answers"], indent=2))',
        ]
    )


def video_section(name: str, video: dict) -> str:
    rows = [
        "### Videos: Perception Test",
        "",
        f"| Perception Test (validation) | {name} |",
        "| --- | ---: |",
        f"| **All {video['questions']:,} questions** | **{video['accuracy']:.1f}** |",
    ]
    rows += [f"| {area.capitalize()} | {video['areas'][area]:.1f} |" for area in AREAS]
    rows += [
        "",
        "Multiple-choice video question answering (three options per question, chance 33.3): top-1 accuracy (%), "
        "videos read with the defaults above.",
    ]
    return "\n".join(rows)


def edit_card(
    text: str,
    *,
    name: str,
    repo: str,
    video: dict,
    latency_ms: float,
    text_rows: dict[str, str] | None = None,
    public_index: float | None = None,
) -> tuple[str, list[str]]:
    edits: list[str] = []

    def sub(old: str, new: str) -> None:
        nonlocal text
        count = text.count(old)
        if count != 1:
            raise ValueError(f"expected one {old[:70]!r}, found {count}")
        text = text.replace(old, new)
        edits.append(old.strip()[:60])

    sub("- multimodal\n- vision\n", "- multimodal\n- vision\n- video\n")
    sub(
        "Give it an input (text or JSON, optionally with images)",
        "Give it an input (text or JSON, optionally with images and videos)",
    )
    sub(
        "| **Inputs** | Text or JSON, plus images (several per request) |",
        "| **Inputs** | Text or JSON, plus images and videos (several per request) |",
    )
    images = (
        "- **Reads images:** multiple images per request (PNG, JPEG or WebP), given as paths, URLs, PIL images or "
        "base64 data URLs; every question of the request sees all of them.\n"
    )
    videos = (
        "- **Reads videos:** multiple videos per request (MP4, WebM, MOV or MKV), given as paths, URLs, base64 data "
        "URLs or frame arrays, read at 2 frames per second; "
        f"**Perception Test (multiple-choice video QA): {video['accuracy']:.1f}** (internal evaluation).\n"
    )
    sub(images, images + videos)
    family = re.search(
        r"- \*\*Speed:\*\* a median of ([\d.]+) ms for a text request and ([\d.]+) ms for a request with an image, "
        r"on one AMD Instinct MI325X GPU, one request at a time\.",
        text,
    )
    big = re.search(
        r"- \*\*Speed:\*\* text requests take a median of (\d+) ms, and requests with an image a median of (\d+) ms, "
        r"on one AMD Instinct MI325X, one request at a time\.",
        text,
    )
    if bool(family) == bool(big):
        raise ValueError("no unique speed line")
    if family:
        sub(
            family.group(0),
            f"- **Speed:** a median of {family.group(1)} ms for a text request, {family.group(2)} ms for a request "
            f"with an image and {latency_ms:.1f} ms for a request with a 10-second video, on one AMD Instinct "
            "MI325X GPU, one request at a time.",
        )
    else:
        sub(
            big.group(0),
            f"- **Speed:** text requests take a median of {big.group(1)} ms, requests with an image a median of "
            f"{big.group(2)} ms, and requests with a 10-second video a median of {int(latency_ms + 0.5)} ms, on one "
            "AMD Instinct MI325X, one request at a time.",
        )
    sub(
        'pip install "transformers==5.17.0" torch torchvision pillow safetensors accelerate\n',
        'pip install "transformers==5.17.0" torch torchvision pillow opencv-python-headless safetensors accelerate\n',
    )
    sub(
        "\n\n# Or as a pipeline:\n",
        "\n\n" + video_block(repo) + "\n\n# Or as a pipeline:\n",
    )
    sub(
        "(state=..., questions=..., images=...)\n```",
        "(state=..., questions=..., images=..., videos=...)\n```",
    )
    sub(
        "Images go before the text of the request, each read at up to 1.6 megapixels; every question of the request "
        "sees all of them.",
        "Images go before the text of the request, each read at up to 1.6 megapixels; videos follow the images, read "
        "at 2 frames per second (at most 32 frames spread over the whole video, each at up to 0.2 megapixels). Every "
        "question of the request sees all of them.",
    )
    label = f"<sub>{name}: internal evaluation."
    sub("\n\n" + label, "\n\n" + video_section(name, video) + "\n\n" + label)
    for old, new in (text_rows or {}).items():
        sub(old, new)
    if public_index is not None:
        line = re.search(
            r"- \*\*Jev Decision Index 0\.3, public suite: ([\d.]+)\*\*", text
        )
        if not line:
            raise ValueError("no public-suite line")
        if f"{public_index:.2f}" != line.group(1):
            sub(
                line.group(0),
                f"- **Jev Decision Index 0.3, public suite: {public_index:.2f}**",
            )
    return text, edits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--video-results", required=True, type=Path)
    ap.add_argument("--latency", required=True, type=Path)
    ap.add_argument("--example", required=True, type=Path)
    ap.add_argument("--example-sha256", required=True)
    ap.add_argument("--text-rows", type=Path)
    ap.add_argument("--public-index", type=float)
    ap.add_argument("--receipt", type=Path)
    args = ap.parse_args(argv)

    package, out = args.package, args.out
    manifest = json.loads((package / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    name = manifest["model_name"]
    results = json.loads(args.video_results.read_text())
    model = results["models"][name]
    if not model["complete"] or model["errors"]:
        raise SystemExit(f"{name}: the video evaluation is not complete")
    video = {
        "accuracy": model["accuracy"],
        "areas": model["areas"],
        "questions": results["questions"],
    }
    latency = json.loads(args.latency.read_text())
    if latency.get("timed", 0) < 100 or not latency.get("median_ms"):
        raise SystemExit("latency report incomplete")
    if sha256(args.example) != args.example_sha256:
        raise SystemExit("the example video is not the pinned file")
    old = (package / "README.md").read_text(encoding="utf-8")
    rows = json.loads(args.text_rows.read_text()) if args.text_rows else None
    new, edits = edit_card(
        old,
        name=name,
        repo=args.repo,
        video=video,
        latency_ms=latency["median_ms"],
        text_rows=rows,
        public_index=args.public_index,
    )
    bad = FORBIDDEN.search(new)
    if bad:
        raise SystemExit(f"forbidden wording in the card: {bad.group(0)!r}")
    if new.rstrip("\n").split("\n")[-1] != "Trained on AMD Instinct MI325X GPUs.":
        raise SystemExit("last line changed")
    if out.exists():
        shutil.rmtree(out)
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(package).as_posix()
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel not in ("README.md", "MODEL_MANIFEST.json"):
            os.link(path, target)
    (out / "README.md").write_bytes(new.encode("utf-8"))
    asset = out / EXAMPLE_VIDEO
    asset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.example, asset)
    for rel in ("README.md", EXAMPLE_VIDEO):
        manifest["files_sha256"][rel] = sha256(out / rel)
        manifest["files_bytes"][rel] = (out / rel).stat().st_size
    manifest["files_sha256"] = dict(sorted(manifest["files_sha256"].items()))
    manifest["files_bytes"] = dict(sorted(manifest["files_bytes"].items()))
    (out / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    sys.path.insert(0, str(out))
    import d3_runtime

    d3_runtime.verify_package(out, "full")
    names = lambda root: {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    }  # noqa: E731
    if names(out) != names(package) | {EXAMPLE_VIDEO}:
        raise SystemExit("unexpected file set")
    diff = list(
        difflib.unified_diff(
            old.split("\n"), new.split("\n"), "before", "after", lineterm="", n=0
        )
    )
    report = {
        "model": name,
        "repo": args.repo,
        "edits": edits,
        "video": video,
        "latency_median_ms": latency["median_ms"],
        "readme_sha256": sha256(out / "README.md"),
        "manifest_sha256": sha256(out / "MODEL_MANIFEST.json"),
        "example_sha256": sha256(asset),
        "files": len(names(out)),
        "diff": diff,
        "ok": True,
    }
    if args.receipt:
        args.receipt.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "diff"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
