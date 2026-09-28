"""Same-panel collector module for ~27B LoRA checkpoints (eval runner adapter).

Runs the current ``training.model.infer`` unchanged: the inference sources that
the release builder vendors for the ``qwen-adapter`` profile, so a scored run
and its release package share them. The release runtime runs under
``python -I``, where the image's FLA overlay is not importable, so it uses the
reference gated-delta path; this module drops the overlay from ``sys.path``
before any import to score on that same path, and writes a runtime sidecar next
to the predictions. It also adds the runner's ``--max-items`` smoke option.
"""

from __future__ import annotations

import sys

FLA_OVERLAY = "decision-fla"
sys.path[:] = [p for p in sys.path if FLA_OVERLAY not in p]

import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from training.model import infer  # noqa: E402


def smoke_argv(argv: list[str]) -> list[str]:
    if "--max-items" not in argv:
        return argv
    argv = list(argv)
    index = argv.index("--max-items")
    count = int(argv[index + 1])
    del argv[index : index + 2]
    source = Path(argv[argv.index("--input") + 1])
    output = Path(argv[argv.index("--output") + 1])
    subset = output.with_name(f"{output.name}.first{count}.prompts.jsonl")
    lines = [
        line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    subset.parent.mkdir(parents=True, exist_ok=True)
    with subset.open("x", encoding="utf-8") as stream:
        stream.write("".join(line + "\n" for line in lines[:count]))
    argv[argv.index("--input") + 1] = str(subset)
    return argv


def main(argv: list[str] | None = None) -> None:
    args = smoke_argv(list(sys.argv[1:] if argv is None else argv))
    if importlib.util.find_spec("fla") is not None:
        raise SystemExit(
            "FLA is importable; the reference gated-delta path is required"
        )
    sys.argv = [infer.__file__, *args]
    infer.main()
    output = Path(args[args.index("--output") + 1])
    runtime = {
        "schema": "decision2-27b-collect-runtime/1",
        "gated_delta": "reference (FLA overlay removed from sys.path)",
        "fla_loaded": any(
            name == "fla" or name.startswith("fla.") for name in sys.modules
        ),
        "sys_path_has_overlay": any(FLA_OVERLAY in p for p in sys.path),
    }
    if runtime["fla_loaded"] or runtime["sys_path_has_overlay"]:
        raise SystemExit("FLA was loaded during collection")
    output.with_name(output.name + ".runtime.json").write_text(
        json.dumps(runtime, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(runtime, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
