"""After a release upload: ``main`` of the repository serves exactly the built package.

With NO revision anywhere: the Hub's ``main`` sha, the tag's target and the expected commit must agree; a fresh
``snapshot_download(repo)`` resolves to that sha and every file re-hashes equal to the built package (no extra or
missing files); the remote code files and ``config.json`` ``auto_map`` are named explicitly. With ``--device``,
``AutoModel.from_pretrained(repo, trust_remote_code=True)`` (no revision) loads from that snapshot and answers a
text request and requests with 1, 4 and 6 images.

    python -m d25.vega.release.main_check --repo vllm-sr/d3 --tag v3.0.1 --expect <commit> --package <built dir> \
        --cache /tmp/fresh --out main-check.json [--device cuda:0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CODE = (
    "d3_runtime.py",
    "d3_engine.py",
    "d3_server.py",
    "d3_format.py",
    "modeling_d3.py",
    "pipeline_d3.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def files(root: Path) -> dict[str, Path]:
    return {
        str(p.relative_to(root)): p
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and not {".cache", "__pycache__"} & set(p.parts)
        and p.name != ".gitattributes"
    }


def smoke(repo: str, device: str, images_dir: Path) -> dict:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(repo, trust_remote_code=True, device=device)
    state = "The customer says the blender arrived cracked and attached the receipt."
    questions = {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Refunds and damaged deliveries",
                "billing": "Payments and invoices",
                "technical": "Product setup and faults",
            },
        },
        "on_receipt": {
            "type": "noul",
            "instructions": "Does the receipt list the blender?",
        },
    }
    assets = [str(p) for p in sorted(images_dir.glob("*.png"))]
    receipt = str(images_dir / "example-receipt.png")
    report = {"class": type(model).__name__}
    for name, images in (
        ("text", None),
        ("1_image", [receipt]),
        ("4_images", (assets * 2)[:4]),
        ("6_images", (assets * 2)[:6]),
    ):
        response = model.system_one(state=state, questions=questions, images=images)
        report[name] = {
            "model": response["model"],
            "answered": all("error" not in a for a in response["answers"].values()),
            "input_tokens": response["usage"]["input_tokens"],
            "answers": {
                k: a.get("choice", a.get("noul"))
                for k, a in response["answers"].items()
            },
        }
    report["ok"] = all(
        report[k]["answered"] and report[k]["model"] == repo.rsplit("/", 1)[-1]
        for k in ("text", "1_image", "4_images", "6_images")
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--expect", required=True, help="the release commit")
    ap.add_argument(
        "--package",
        required=True,
        type=Path,
        help="the built package that was uploaded",
    )
    ap.add_argument(
        "--cache",
        required=True,
        type=Path,
        help="an empty Hugging Face cache directory",
    )
    ap.add_argument("--device")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    os.environ["HF_HUB_CACHE"] = str(args.cache)
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    main_sha = api.model_info(args.repo).sha
    refs = api.list_repo_refs(args.repo)
    # An annotated tag's ref names the tag object; the commit it points to is what the revision resolves to.
    tag_object = next((t.target_commit for t in refs.tags if t.name == args.tag), None)
    tag = api.model_info(args.repo, revision=args.tag).sha if tag_object else None
    branches = {b.name: b.target_commit for b in refs.branches}
    snapshot = Path(snapshot_download(args.repo))
    got, want = files(snapshot), files(args.package)
    mismatched = sorted(
        n for n in want if n in got and sha256(got[n]) != sha256(want[n])
    )
    config = json.loads((snapshot / "config.json").read_text())
    report = {
        "repo": args.repo,
        "expect": args.expect,
        "main_sha": main_sha,
        "branches": branches,
        "tag": {args.tag: tag, "tag_object": tag_object},
        "snapshot_sha": snapshot.name,
        "files": len(got),
        "missing": sorted(set(want) - set(got)),
        "extra": sorted(set(got) - set(want)),
        "mismatched": mismatched,
        "remote_code_sha256": {n: sha256(got[n]) for n in CODE if n in got},
        "auto_map": config.get("auto_map"),
    }
    report["hub_ok"] = main_sha == tag == snapshot.name == args.expect and list(
        branches
    ) == ["main"]
    report["files_ok"] = not (report["missing"] or report["extra"] or mismatched)
    report["code_ok"] = (
        all(n in got for n in CODE)
        and all(sha256(got[n]) == sha256(want[n]) for n in CODE)
        and config.get("auto_map")
        == json.loads((args.package / "config.json").read_text()).get("auto_map")
    )
    if args.device:
        sys.path.insert(0, str(snapshot))
        report["smoke"] = smoke(args.repo, args.device, snapshot / "assets")
    report["ok"] = (
        report["hub_ok"]
        and report["files_ok"]
        and report["code_ok"]
        and (not args.device or report["smoke"]["ok"])
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "main_sha",
                    "tag",
                    "snapshot_sha",
                    "files",
                    "hub_ok",
                    "files_ok",
                    "code_ok",
                    "ok",
                )
            }
        )
    )
    if args.device:
        print(json.dumps(report["smoke"]))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
