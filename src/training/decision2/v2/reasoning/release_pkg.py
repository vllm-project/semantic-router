"""A Reasoning release package from the base model's released package and a scored BF16 checkpoint.

The base package (a verified download of the released Decision 2.0 repository at a pinned revision) supplies the
runtime, remote code, tokenizer and layout; the model files listed in its manifest (``model_files``) are replaced
by the candidate's BF16 release copy (``v2.release.bf16_copy``), the card and figures by the Reasoning card. The
manifest is rewritten so the package verifies: model name and repository, identity (recomputed with the package's
own vendored ``checkpoint_fingerprint`` and required to equal ``--model-sha256``), parameter counts, origin, licence
components, card figure digests and every file digest. Then ``verify_bundle`` runs on the result.

usage: python3 -m v2.reasoning.release_pkg --template PKG --checkpoint CKPT --model-sha256 SHA --repo-id REPO
         --origin-revision REV --card README.md --assets DIR --scored JSON --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_count(path: Path) -> int:
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(size))
    total = 0
    for name, meta in header.items():
        if name != "__metadata__":
            count = 1
            for dim in meta["shape"]:
                count *= dim
            total += count
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--origin-revision", required=True)
    parser.add_argument(
        "--origin-summary",
        default="Continued training of the base Decision 2.0 model of the same "
        "size on reasoning problems with supervised intermediate steps, interpolated with the base "
        "weights. Same architecture, tokenizer, prompt format and runtime as the base model.",
    )
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument(
        "--scored",
        type=Path,
        required=True,
        help="JSON recorded as the manifest's scored entry",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    template = args.template.resolve(strict=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads(
        (template / "MODEL_MANIFEST.json").read_text(encoding="utf-8")
    )
    pointer = json.loads((template / "config.json").read_text(encoding="utf-8"))
    base_repo, base_name = manifest["repo_id"], manifest["model_name"]
    model_name = args.repo_id.split("/", 1)[1]
    if not model_name.startswith(base_name + "-"):
        raise ValueError(f"{model_name} is not a variant of {base_name}")
    shutil.copytree(
        template,
        args.out,
        ignore=shutil.ignore_patterns(
            ".cache", ".gitattributes", "MODEL_MANIFEST.json", "__pycache__"
        ),
    )
    for name in manifest["model_files"]:
        source = args.checkpoint / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, args.out / name)
    shutil.copyfile(args.card, args.out / "README.md")
    figures = {}
    for figure in manifest["card"]["figures_sha256"]:
        shutil.copyfile(args.assets / Path(figure).name, args.out / figure)
        figures[figure] = sha256(args.out / figure)
    pointer["model_name"] = model_name
    (args.out / "config.json").write_text(
        json.dumps(pointer, indent=2) + "\n", encoding="utf-8"
    )

    sys.path.insert(0, str(args.out))
    infer = importlib.import_module("decision2._vendor.dev2model.infer")
    identity = infer.checkpoint_fingerprint(args.out)
    if identity["model_sha256"] != args.model_sha256:
        raise ValueError(
            f"package identity {identity['model_sha256']} != scored {args.model_sha256}"
        )
    parameters = manifest["parameters"]
    old = sum(parameters["packaged"].values())
    parameters["packaged"] = {
        group: sum(tensor_count(args.out / f) for f in files)
        for group, files in parameters["packaged_files"].items()
    }
    parameters["loaded"] += sum(parameters["packaged"].values()) - old
    manifest.update(
        {
            "repo_id": args.repo_id,
            "model_name": model_name,
            "identity": {
                "model_sha256": identity["model_sha256"],
                "fingerprint_files": identity.get("files_sha256", {}),
            },
            "origin": {
                "repo_id": base_repo,
                "revision": args.origin_revision,
                "relation": "finetune",
                "summary": args.origin_summary,
            },
            "scored": json.loads(args.scored.read_text(encoding="utf-8")),
            "calibration": None,
        }
    )
    components = manifest["licence"]["components"]
    components[0] = {
        "component": f"{model_name} weights, decision head, package runtime, card and artwork",
        "licence": "apache-2.0",
        "source": args.repo_id,
    }
    components.insert(
        1,
        {
            "component": f"{base_name} (direct weight origin, continued training)",
            "licence": "apache-2.0",
            "source": f"{base_repo}@{args.origin_revision}",
        },
    )
    manifest["card"] = {"figures_sha256": figures}
    files = {}
    for path in sorted(args.out.rglob("*")):
        if (
            path.is_file()
            and path.name != "MODEL_MANIFEST.json"
            and "__pycache__" not in path.parts
        ):
            files[path.relative_to(args.out).as_posix()] = sha256(path)
    manifest["files_sha256"] = files
    for name, entry in manifest.get("runtime_files", {}).items():
        entry["sha256"] = files[name]
    (args.out / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for cached in args.out.rglob("__pycache__"):
        shutil.rmtree(cached)
    api = importlib.import_module("decision2.api")
    verified = api.verify_bundle(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "model_name": verified["model_name"],
                "model_sha256": verified["identity"]["model_sha256"],
                "loaded": verified["parameters"]["loaded"],
                "manifest_sha256": sha256(args.out / "MODEL_MANIFEST.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
