"""A weights-only revision of a d3 package: every file of --package except the weight files, which come from --export
(11 model shards + readout.safetensors). MODEL_MANIFEST.json gets the new weight hashes and sizes, the new identity
(build.model_identity: weights, tokenizer, inference fields) and built_utc; runtime, config, tokenizer, templates, card
and assets stay byte-identical. Checked with the runtime's own full verification.

    python -m d25.vega.release.weights_revision --package <released package> --export <export dir> --out <new dir> [--card <card dir>]

``--card``: README.md and assets/*.png of a rendered card replace the package's (their manifest entries refreshed too).
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from d25.vega.release.build import model_identity, sha256_file

MANIFEST = "MODEL_MANIFEST.json"


def place(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--card", type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    old = json.loads((args.package / MANIFEST).read_text(encoding="utf-8"))
    weights = sorted(n for n in old["files_sha256"] if n.endswith(".safetensors"))
    assert weights == sorted(
        [f"model-{i:05d}-of-00011.safetensors" for i in range(1, 12)]
        + ["readout.safetensors"]
    ), weights
    assert all(
        (args.export / n).is_file() for n in weights
    ), "export lacks a weight file"
    assert (args.export / "model.safetensors.index.json").read_bytes() == (
        args.package / "model.safetensors.index.json"
    ).read_bytes(), "the export's shard index differs from the package's"
    reproduced = model_identity(args.package)
    assert (
        reproduced["model_sha256"] == old["identity"]["model_sha256"]
    ), "cannot reproduce the package's identity"

    names = sorted(
        p.relative_to(args.package).as_posix()
        for p in args.package.rglob("*")
        if p.is_file()
        and p.name != MANIFEST
        and ".cache" not in p.parts
        and "__pycache__" not in p.parts
    )
    assert set(names) == set(old["files_sha256"]), sorted(
        set(names) ^ set(old["files_sha256"])
    )
    card = {}
    if args.card:
        card = {"README.md": args.card / "README.md"}
        card.update(
            {
                f"assets/{p.name}": p
                for p in sorted((args.card / "assets").glob("*.png"))
            }
        )
        assert set(card) <= set(names), sorted(set(card) - set(names))
    for n in names:
        if n in card:
            target = args.out / n
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(card[n], target)
        else:
            place((args.export if n in weights else args.package) / n, args.out / n)

    manifest = dict(old)
    fresh = set(weights) | set(card)
    manifest["files_sha256"] = {
        n: sha256_file(args.out / n) if n in fresh else old["files_sha256"][n]
        for n in names
    }
    manifest["files_bytes"] = {n: (args.out / n).stat().st_size for n in names}
    identity = model_identity(args.out)
    assert (
        identity["decision"] == old["identity"]["decision"]
    ), "inference fields changed"
    assert set(identity) == set(old["identity"]), (
        sorted(identity),
        sorted(old["identity"]),
    )
    manifest["identity"] = identity
    manifest["built_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (args.out / MANIFEST).write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )

    sys.path.insert(0, str(args.out))
    import d3_runtime

    d3_runtime.verify_package(args.out, "full")
    changed = sorted(
        n for n in names if manifest["files_sha256"][n] != old["files_sha256"][n]
    )
    manifest_keys = sorted(k for k in manifest if manifest[k] != old.get(k))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "files": len(names) + 1,
                "changed": changed,
                "manifest_keys_changed": manifest_keys,
                "identity": {
                    "old": old["identity"]["model_sha256"],
                    "new": identity["model_sha256"],
                },
                "verify": "full ok",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
