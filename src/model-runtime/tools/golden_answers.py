"""Print a model's golden answers on this host's device class, or record a family's into its golden file.

    python3 tools/golden_answers.py vllm-sr/Decision-2.0-Kai-0.6B --device cpu
    python3 tools/golden_answers.py vllm-sr/Decision-1.0-Lux-9B --device rocm:0 \\
        --autotune-cache /var/cache/vllm-srun/autotune
    python3 tools/golden_answers.py --family task_heads --device cpu --offline \\
        --record vllm_srun/registry/golden_answers_vela1.json
    python3 tools/golden_answers.py --family decision3 --device cpu --dtype bfloat16 \\
        --record vllm_srun/registry/golden_answers_decision3.json
    VLLM_SRUN_ZENTORCH=1 python3 tools/golden_answers.py --family decision3 \\
        --device cpu --cpu-bf16 --record vllm_srun/registry/golden_answers_decision3.json

The model is served as ``vllm-srun serve`` serves it (verification,
pinned kernel choices, ``--dtype``) and answers its family's golden requests
alone, in their released forwards, recorded as the readiness check compares them:
decision answers, or a surface response's numbers (``LoadedModel.golden_values``).
Loading checks that the answers are deterministic and well formed but does not
compare them with the recorded references, which a recording replaces;
``previous`` counts the answers each recorded reference still matches.
``--record`` writes a GPU's hardware-qualified reference alone (``--generic``
adds the device-class one) and both on a CPU, keyed by repository and pinned
revision, with the PyTorch version
each was recorded with (``torch``); at a ``--dtype`` other than the model's
``auto`` both keys carry it (``cpu@bfloat16``). Within a revision and PyTorch
version, the first generic baseline is retained while hardware entries are
updated; a hardware entry that answers as its baseline is not kept. A new
revision discards the old references and starts a new baseline; a new PyTorch
version starts a new baseline for its device class. Record it on the intended
baseline device before adding other hardware classes.

``--cpu-bf16`` serves ``auto`` as a CPU class with BF16 references does
(``dtypes.on_host``), with the approximate CPU kernels the family consents to,
and records the BF16 reference that turns it on: under the hardware key that
names those kernels (``cpu:…:amd:zentorch@bfloat16``), or the plain key where
none run, with each kernel library's version beside the PyTorch version
(``kernels``: a new version can sum in another order, so it is recorded again).
Answers computed with approximate kernels never replace the generic baseline.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from vllm_srun import dtypes
from vllm_srun import runtime as serving
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.registry import builtin
from vllm_srun.runtime import Runtime
from vllm_srun.supervision.readiness import CPU_TOLERANCE, GPU_TOLERANCE, qualified

# Pinned by a family table but not served yet: the vela3 image package's
# serving path is not implemented (#5625), so Runtime.load refuses it and a
# family-wide run would stop before recording. An explicit --model still
# loads it and fails loudly.
UNSERVED = frozenset({"vllm-sr/vela3-fast-vision"})


@contextlib.contextmanager
def unreferenced() -> Iterator[None]:
    """The loading readiness check without the recorded references, which the recording replaces."""
    check = serving.golden_check

    def without_references(
        run: Any, compare: Any, goldens: list[dict[str, Any]], *args: Any
    ) -> Any:
        return check(run, compare, [{**g, "expected": {}} for g in goldens], *args)

    serving.golden_check = without_references
    try:
        yield
    finally:
        serving.golden_check = check


@contextlib.contextmanager
def bf16_referenced(enabled: bool) -> Iterator[None]:
    """``auto`` in BF16 on a native-BF16 CPU class as if its BF16 references were recorded."""
    referenced = dtypes.referenced
    if enabled:
        dtypes.referenced = lambda policy, key: True
    try:
        yield
    finally:
        dtypes.referenced = referenced


def answers(
    args: argparse.Namespace, model: str, revision: str | None
) -> dict[str, Any]:
    runtime = Runtime(
        ServeConfig(
            models=(
                ModelConfig(
                    model=model,
                    revision=revision,
                    device=args.device,
                    dtype=args.dtype,
                ),
            ),
            cache_dir=args.cache_dir,
            base_path=args.base_path,
            offline=args.offline,
            autotune_cache=args.autotune_cache,
        )
    )
    try:
        with unreferenced(), bf16_referenced(getattr(args, "cpu_bf16", False)):
            runtime.load()
        served = runtime.lookup(None)
        assert served.model is not None and served.placement is not None
        assert served.family is not None and served.package is not None
        accelerator = served.placement.device.accelerator
        tolerance = CPU_TOLERANCE if accelerator == "cpu" else GPU_TOLERANCE
        values: dict[str, Any] = {}
        previous: dict[str, dict[str, int]] = {}
        for golden in served.family.golden(served.package):
            answered = served.golden_surface(golden["surface"], golden["body"])
            values.update(answered)
            for reference_key, reference in (golden.get("expected") or {}).items():
                counts = served.model.golden_compare(
                    golden["surface"], answered, reference, tolerance
                )
                totals = previous.setdefault(
                    reference_key, {"checked": 0, "matched": 0}
                )
                checked, matched = counts or (0, 0)
                totals["checked"] += checked
                totals["matched"] += matched
        info = served.model.info
        # An ``auto`` model in BF16 on a CPU class with BF16 references records at bfloat16.
        dtype = served.golden_dtype or served.dtype
        key = served.placement.accelerator.golden_reference(served.placement.device)
        # Answers from approximate kernels are recorded only under the key naming them.
        goldens = (
            {} if served.golden_kernels else {qualified(accelerator, dtype): values}
        )
        if key is not None:
            key = dtypes.with_kernels(key, served.golden_kernels)
            goldens[qualified(key, dtype)] = values
        return {
            "model": info.id,
            "repo": info.repo,
            "revision": info.revision,
            "model_sha256": info.model_sha256,
            "device": served.placement.device.label,
            "dtype": dtype,
            "torch": torch.__version__,
            # A key naming kernels holds their libraries' summation order too.
            "kernels": kernel_versions(served.golden_kernels),
            "readiness": served.health.golden.describe(),
            "previous": previous,
            "golden_answers": goldens,
        }
    finally:
        runtime.stop()


def kernel_versions(sources: tuple[str, ...]) -> dict[str, str | None]:
    """Each approximate kernel source's library version (``zentorch``: its module's)."""
    return {
        source: getattr(sys.modules.get(source), "__version__", None)
        for source in sources
    }


def record(
    path: Path, entries: list[dict[str, Any]], generic_too: bool = False
) -> None:
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for entry in entries:
        current = table.get(entry["repo"])
        if current is None or current.get("revision") != entry["revision"]:
            current = {"revision": entry["revision"], "answers": {}}
        answers, recorded_with = current["answers"], current.get("torch", {})
        kernels_with = current.get("kernels", {})
        version = entry.get("torch")
        device_class = entry["device"].split(":", 1)[0]
        generic = qualified(device_class, entry.get("dtype", "auto"))
        # A class reference is checked against every other architecture of its
        # class, failing hosts that should serve `unverified`, and recording one
        # drops the hardware entry that answers as it. A CPU keeps its baseline.
        scoped = (
            device_class != "cpu"
            and not generic_too
            and any(key != generic for key in entry["golden_answers"])
        )
        for key, values in entry["golden_answers"].items():
            if key == generic:
                if scoped:
                    continue
                if key in answers and recorded_with.get(key) == version:
                    continue
            # A hardware key at the generic key's values adds nothing, except
            # at BF16 on a CPU: there it is what lets ``auto`` hold BF16 on
            # that class (``DtypePolicy.cpu_bf16_references``).
            elif (
                not scoped
                and values == answers.get(generic)
                and not (key.startswith("cpu:") and key.endswith("@bfloat16"))
            ):
                answers.pop(key, None)
                recorded_with.pop(key, None)
                kernels_with.pop(key, None)
                continue
            answers[key] = values
            if entry.get("kernels"):
                kernels_with[key] = entry["kernels"]
            else:
                kernels_with.pop(key, None)
            if version is None:
                recorded_with.pop(key, None)
            else:
                recorded_with[key] = version
        if recorded_with:
            current["torch"] = recorded_with
        else:
            current.pop("torch", None)
        if kernels_with:
            current["kernels"] = kernels_with
        else:
            current.pop("kernels", None)
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
    parser.add_argument(
        "--dtype", default="auto", help="serve and record at this --dtype"
    )
    parser.add_argument(
        "--cpu-bf16",
        action="store_true",
        help="serve auto in BF16 on this native-BF16 CPU class and record its BF16 reference",
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--base-path")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--autotune-cache")
    parser.add_argument(
        "--record", type=Path, help="merge into this golden answers file"
    )
    parser.add_argument(
        "--generic",
        action="store_true",
        help="on a GPU, also record the device-class reference",
    )
    args = parser.parse_args()
    if bool(args.model) == bool(args.family):
        parser.error("give a model or --family")
    targets = (
        [
            (m.repo_id, m.revision)
            for m in builtin.all_models(args.family)
            if m.repo_id not in UNSERVED
        ]
        if args.family
        else [(args.model, args.revision)]
    )
    entries = [answers(args, model, revision) for model, revision in targets]
    if args.record:
        record(args.record, entries, args.generic)
    print(json.dumps(entries if args.family else entries[0], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
