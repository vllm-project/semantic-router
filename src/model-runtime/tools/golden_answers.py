"""Print a model's golden answers on this host's device class, or record a family's into its golden file.

    python3 tools/golden_answers.py vllm-sr/Decision-2.0-Kai-0.6B --device cpu
    python3 tools/golden_answers.py vllm-sr/Decision-1.0-Lux-9B --device rocm:0 \\
        --autotune-cache /var/cache/vllm-srun/autotune
    python3 tools/golden_answers.py --family task_heads --device cpu --offline \\
        --record vllm_srun/registry/golden_answers_vela1.json

The model is served as ``vllm-srun serve`` serves it (verification,
pinned kernel choices, readiness) and answers its family's golden requests on
the exact profile, recorded as the readiness check compares them: decision
answers, or a surface response's numbers (``LoadedModel.golden_values``).
``--record`` merges this device class into the file's entries, keyed by
repository and pinned revision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.registry import builtin
from vllm_srun.runtime import Runtime


def answers(
    args: argparse.Namespace, model: str, revision: str | None
) -> dict[str, Any]:
    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model=model, revision=revision, device=args.device),),
            cache_dir=args.cache_dir,
            base_path=args.base_path,
            offline=args.offline,
            autotune_cache=args.autotune_cache,
        )
    )
    try:
        runtime.load()
        served = runtime.lookup(None)
        assert served.model is not None and served.placement is not None
        assert served.family is not None and served.package is not None
        values: dict[str, Any] = {}
        for golden in served.family.golden(served.package):
            values.update(served.golden_surface(golden["surface"], golden["body"]))
        info = served.model.info
        result = {
            "model": info.id,
            "repo": info.repo,
            "revision": info.revision,
            "model_sha256": info.model_sha256,
            "device": served.placement.device.label,
            "readiness": served.health.golden.describe(),
            "golden_answers": {served.placement.device.accelerator: values},
        }
        if (
            served.placement.device.accelerator == "mps"
            and getattr(args, "tolerance", None) is not None
        ):
            result["tolerances"] = {"mps": args.tolerance}
        return result
    finally:
        runtime.stop()


def record(path: Path, entries: list[dict[str, Any]]) -> None:
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for entry in entries:
        current = table.get(entry["repo"])
        if current is None or current.get("revision") != entry["revision"]:
            current = {"revision": entry["revision"], "answers": {}}
        current["answers"].update(entry["golden_answers"])
        if entry.get("tolerances"):
            current.setdefault("tolerances", {}).update(entry["tolerances"])
        table[entry["repo"]] = current
    path.write_text(
        json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?")
    parser.add_argument("--family", help="every built-in model of this family")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir")
    parser.add_argument("--base-path")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--autotune-cache")
    parser.add_argument(
        "--tolerance",
        type=float,
        help="measured absolute error bound for this model's MPS answers; required with --device mps --record",
    )
    parser.add_argument(
        "--record", type=Path, help="merge into this golden answers file"
    )
    args = parser.parse_args()
    if args.tolerance is not None and (
        not math.isfinite(args.tolerance) or args.tolerance < 0
    ):
        parser.error("--tolerance must be a finite non-negative number")
    if args.record and args.device == "mps" and args.tolerance is None:
        parser.error("MPS recording needs an explicitly measured --tolerance")
    if bool(args.model) == bool(args.family):
        parser.error("give a model or --family")
    targets = (
        [(m.repo_id, m.revision) for m in builtin.all_models(args.family)]
        if args.family
        else [(args.model, args.revision)]
    )
    entries = [answers(args, model, revision) for model, revision in targets]
    if args.record:
        record(args.record, entries)
    print(json.dumps(entries if args.family else entries[0], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
