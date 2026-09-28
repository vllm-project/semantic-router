"""Write the `v2.data.assemble_hf_upload` spec for A7 run directories.

Destinations are rooted at `a7/` so the assembler's registry lands at
`a7/registry.json`; the assembled `a7/` folder is uploaded to `v2/a7` of the
private dataset and never touches the research & data track's `v2/` files.
Only admitted rows, count-only manifests and public (aggregate) receipts are
listed; private overlap receipts with matched protected ids stay on the node.
The primary run keeps the top-level layout (`admission.json`, `views/`, ...);
each `--extra-run NAME=DIR` adds its sub-arms under `arms/` and its version
records under `versions/NAME/`, so one registry lists every A7 file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2.data.a7.build_a7 import ALL_SUB_ARMS


def _sub_arm_items(run_dir: Path) -> list[dict[str, str]]:
    items = []
    for name in ALL_SUB_ARMS:
        for part in ("train", "aho"):
            rows = run_dir / "final" / f"{name}.{part}.jsonl"
            manifest = run_dir / "manifests" / f"{name}.{part}.freeze.json"
            if rows.exists():
                if not manifest.exists():
                    raise FileNotFoundError(f"{manifest} is missing for {rows}")
                items.append({"src": str(rows), "dst": f"a7/arms/{name}/{part}.jsonl"})
                items.append(
                    {
                        "src": str(manifest),
                        "dst": f"a7/arms/{name}/manifest.{part}.json",
                    }
                )
        for kind, folder in (("overlap.public", "screens"), ("shortcut", "screens")):
            receipt = run_dir / folder / f"{name}.{kind}.json"
            if receipt.exists():
                items.append(
                    {"src": str(receipt), "dst": f"a7/audits/{name}/{kind}.json"}
                )
        for kind in ("shortcut", "selfscan.public"):
            receipt = run_dir / "post" / f"{name}.{kind}.json"
            if receipt.exists():
                items.append(
                    {"src": str(receipt), "dst": f"a7/audits/{name}/post-{kind}.json"}
                )
        receipt = run_dir / "rescreen" / f"{name}.overlap.public.json"
        if receipt.exists():
            items.append(
                {
                    "src": str(receipt),
                    "dst": f"a7/audits/{name}/rescreen-overlap.public.json",
                }
            )
    return items


def spec_for(
    run_dir: Path,
    readme: Path,
    license_registries: Path | list[Path],
    extra_runs: list[tuple[str, Path]] | None = None,
) -> list[dict[str, str]]:
    if isinstance(license_registries, Path):
        license_registries = [license_registries]
    items = [{"src": str(readme), "dst": "a7/README.md"}]
    items += [
        {"src": str(path), "dst": f"a7/{path.name}"} for path in license_registries
    ]
    items += [
        {
            "src": str(run_dir / "build" / "build-manifest.json"),
            "dst": "a7/build-manifest.json",
        },
        {"src": str(run_dir / "final" / "admission.json"), "dst": "a7/admission.json"},
        {"src": str(run_dir / "isolation.json"), "dst": "a7/audits/isolation.json"},
    ]
    items += _sub_arm_items(run_dir)
    embed = run_dir / "embed" / "embed.public.json"
    if embed.exists():
        items.append({"src": str(embed), "dst": "a7/audits/embed.public.json"})
    for view in sorted((run_dir / "final" / "views").glob("*.json")):
        items.append({"src": str(view), "dst": f"a7/views/{view.name}"})
    for name, extra in extra_runs or []:
        for src, dst in (
            (extra / "build" / "build-manifest.json", "build-manifest.json"),
            (extra / "final" / "admission.json", "admission.json"),
            (extra / "isolation.json", "isolation.json"),
            (extra / "embed" / "embed.public.json", "embed.public.json"),
        ):
            if src.exists():
                items.append({"src": str(src), "dst": f"a7/versions/{name}/{dst}"})
        items += _sub_arm_items(extra)
    destinations = [item["dst"] for item in items]
    if len(set(destinations)) != len(destinations):
        raise ValueError("two A7 runs publish the same destination")
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--license-registry", type=Path, action="append", required=True)
    parser.add_argument("--extra-run", action="append", default=[], help="NAME=DIR")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    extra = [
        (name, Path(path))
        for name, path in (value.split("=", 1) for value in args.extra_run)
    ]
    items = spec_for(args.run_dir, args.readme, args.license_registry, extra)
    args.out.write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(items)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
