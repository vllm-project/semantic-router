"""Parameter count of a package on node B for the node-A report (decoder M5 formal runs).

`same_panel report --count-safetensors DIR` reads only the 8-byte length and the JSON header of every
`*.safetensors` file under DIR. The weights stay on node B, so this writes header-only stub files (same relative
paths, exactly those leading bytes) that node A passes to `--count-safetensors`, and records the count of the real
package from the same function; both must agree (checked here and again after the node-A report).

usage: python3 m5-params.py --package <checkpoint dir> --stubs <out dir> --output PARAMS.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from v2.eval.same_panel import count_safetensors  # noqa: E402


def write_stubs(package: Path, stubs: Path) -> list[dict]:
    out = []
    for file in sorted(package.rglob("*.safetensors")):
        with file.open("rb") as stream:
            head = stream.read(8)
            (length,) = struct.unpack("<Q", head)
            blob = head + stream.read(length)
        target = stubs / file.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(blob)
        out.append(
            {
                "file": str(file.relative_to(package)),
                "stub_sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--stubs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    real = count_safetensors(args.package)
    stubs = write_stubs(args.package, args.stubs)
    stub = count_safetensors(args.stubs)
    if stub["parameters"] != real["parameters"] or stub["files"] != real["files"]:
        raise SystemExit(
            f"stub count {stub['parameters']} differs from package count {real['parameters']}"
        )
    manifest = "".join(f"{s['stub_sha256']}  {s['file']}\n" for s in stubs)
    result = {
        "schema": "dec-m5-params/1",
        "source": "safetensors header element count (v2.eval.same_panel.count_safetensors) on node B",
        "package": str(args.package),
        "parameters": real["parameters"],
        "files": real["files"],
        "stubs": stubs,
        "stubs_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"parameters": real["parameters"], "files": len(stubs)}))


if __name__ == "__main__":
    main()
