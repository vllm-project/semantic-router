"""Development copy of a downloaded package with the remote code added exactly as the builder adds it.

Every other file is a hard link to the download; ``config.json`` gains ``CONFIG_FIELDS`` and
``MODEL_MANIFEST.json`` the new hashes and its ``remote_code`` section, so ``verify_bundle`` and the
scored identity still hold. For parity work before a release build; releases use ``v2.release.build``.

    python3 -m v2.release.automap.overlay --package DOWNLOAD/DEV2.0-0.8B --output WORK/DEV2.0-0.8B
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from v2.release import automap, layout


def remote_code_section(records: dict) -> dict:
    return {"files": records, **automap.CONFIG_FIELDS}


def overlay(package: Path, output: Path) -> dict:
    package = package.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((package / layout.MANIFEST_NAME).read_text(encoding="utf-8"))
    pointer = json.loads((package / layout.POINTER_NAME).read_text(encoding="utf-8"))
    inventory = layout.inventory(package, allow_hub_added=True)
    stage = output.parent / f".{output.name}.tmp"
    stage.mkdir(parents=True)
    replaced = {layout.MANIFEST_NAME, layout.POINTER_NAME, *automap.FILES}
    for name in inventory:
        if name in replaced:
            continue
        (stage / name).parent.mkdir(parents=True, exist_ok=True)
        os.link(package / name, stage / name)
    records = automap.copy_into(stage)
    (stage / layout.POINTER_NAME).write_text(
        json.dumps(
            {**pointer, **automap.CONFIG_FIELDS},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    files = {
        n: d for n, d in manifest["files_sha256"].items() if n not in automap.FILES
    }
    files[layout.POINTER_NAME] = layout.sha_file(stage / layout.POINTER_NAME)
    files.update({name: record["sha256"] for name, record in records.items()})
    manifest = {
        **manifest,
        "files_sha256": files,
        "remote_code": remote_code_section(records),
    }
    (stage / layout.MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if layout.inventory(stage) != {
        **files,
        layout.MANIFEST_NAME: layout.sha_file(stage / layout.MANIFEST_NAME),
    }:
        raise ValueError("Overlay inventory differs from its manifest")
    stage.rename(output)
    return {
        "package": str(output),
        "manifest_sha256": layout.sha_file(output / layout.MANIFEST_NAME),
        "remote_code": {n: r["sha256"] for n, r in records.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(overlay(args.package, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
