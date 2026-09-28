"""Kernel-path same-panel collector for ~27B LoRA checkpoints (eval runner adapter).

Runs the current ``training.model.infer`` exactly as ``typed_collect`` does (base
via ``--source-path``, CAL calibration, ``--max-length``), but on the image's FLA
and causal-conv1d kernels with a persisted Triton autotune cache, as the
AutoJev-27B comparator ran. Before anything imports Transformers it puts the
image's FLA site on ``sys.path`` ahead of site-packages if the launcher's
``PYTHONPATH`` dropped it, and fails unless ``fla`` resolves inside that site (a
working directory or earlier path entry cannot shadow it). The process fails
before the model loads unless ``v2.dec.runtime_check.require_runtime`` passes and
``TRITON_CACHE_DIR`` is an existing directory. A runtime sidecar lands next to
the predictions.

Options on top of ``training.model.infer``: the runner's ``--max-items N`` smoke
(first N prompts), and ``--longest`` (the one prompt whose longest admitted
question at ``--max-length`` is the longest, for memory smokes; tokens only,
never gold).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from training.model import infer

FLA_SITE = "/opt/decision-fla"
SCHEMA = "decision2-27b-collect-runtime-kernel/1"


def _same(a: str, b: str) -> bool:
    return os.path.realpath(a or ".") == os.path.realpath(b)


def ensure_site(path: list[str], site: str = FLA_SITE) -> bool:
    """Insert ``site`` before site-packages unless present; True if inserted."""
    if any(entry and _same(entry, site) for entry in path):
        return False
    index = next(
        (
            i
            for i, entry in enumerate(path)
            if entry.rstrip("/").endswith(("site-packages", "dist-packages"))
        ),
        len(path),
    )
    path.insert(index, site)
    return True


def fla_origin(site: str = FLA_SITE) -> str:
    spec = importlib.util.find_spec("fla")
    origin = getattr(spec, "origin", None) if spec is not None else None
    if origin is None:
        raise SystemExit(f"FLA is not importable; the kernel path needs {site}")
    if os.path.commonpath([os.path.realpath(origin), os.path.realpath(site)]) != (
        os.path.realpath(site)
    ):
        raise SystemExit(f"fla resolves to {origin}, not the image overlay {site}")
    return origin


def kernel_runtime(site: str = FLA_SITE) -> dict[str, Any]:
    """Verified kernel runtime identity; call before anything loads a model."""
    inserted = ensure_site(sys.path, site)
    origin = fla_origin(site)
    runtime_check = importlib.import_module("v2.dec.runtime_check")
    try:
        identity = runtime_check.require_runtime()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    cache = identity["triton_cache_dir"]
    if not os.path.isdir(cache):
        raise SystemExit(f"TRITON_CACHE_DIR {cache} is not an existing directory")
    return {
        **identity,
        "fla_site": site,
        "fla_site_inserted": inserted,
        "fla_origin": origin,
    }


def versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "peft", "tokenizers", "safetensors"):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "__version__"):
            out[name] = module.__version__
            continue
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def memory() -> dict[str, Any]:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return {}
    return {
        "device_name": torch.cuda.get_device_name(0),
        "hip": torch.version.hip,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
    }


def smoke_argv(argv: list[str]) -> list[str]:
    # typed_collect strips the FLA overlay from sys.path at import, so its copy
    # of this helper is not imported here.
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


def select_longest(
    rows: list[dict[str, Any]], length: Callable[[dict[str, Any]], int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The prompt with the longest admitted question; ``length`` raises ValueError if over."""
    best, best_key, over = None, None, 0
    for item in rows:
        lengths = []
        for question_id, question in item["questions"].items():
            try:
                lengths.append(
                    length(infer.question_to_row(item, question_id, question))
                )
            except ValueError:
                over += 1
        if not lengths:
            continue
        key = (max(lengths), sum(lengths))
        if best_key is None or key > best_key:
            best, best_key = item, key
    if best is None:
        raise SystemExit("no prompt has an admitted question at this max_length")
    return best, {
        "id": best["id"],
        "longest_question_tokens": best_key[0],
        "admitted_tokens": best_key[1],
        "questions": len(best["questions"]),
        "prompts_scanned": len(rows),
        "questions_over_limit_in_input": over,
    }


def longest_argv(argv: list[str]) -> tuple[list[str], dict[str, Any] | None]:
    if "--longest" not in argv:
        return argv, None
    argv = [token for token in argv if token != "--longest"]
    from transformers import AutoTokenizer

    from training.model.decision_model import encode

    source = Path(argv[argv.index("--input") + 1])
    output = Path(argv[argv.index("--output") + 1])
    max_length = int(argv[argv.index("--max-length") + 1])
    tokenizer = AutoTokenizer.from_pretrained(
        argv[argv.index("--checkpoint") + 1], local_files_only=True
    )
    item, info = select_longest(
        infer.load_prompts(source),
        lambda row: len(encode(row, tokenizer, max_length)["ids"]),
    )
    subset = output.with_name(f"{output.name}.longest.prompts.jsonl")
    subset.parent.mkdir(parents=True, exist_ok=True)
    with subset.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    argv[argv.index("--input") + 1] = str(subset)
    return argv, {**info, "source_input": str(source)}


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    max_items = int(raw[raw.index("--max-items") + 1]) if "--max-items" in raw else None
    runtime = kernel_runtime()
    args = smoke_argv(raw)
    args, longest = longest_argv(args)
    started = time.perf_counter()
    sys.argv = [infer.__file__, *args]
    infer.main()
    wall = time.perf_counter() - started
    output = Path(args[args.index("--output") + 1])
    after = importlib.import_module("v2.dec.runtime_check").runtime_identity()
    record = {
        "schema": SCHEMA,
        "gated_delta": "kernel (image FLA / causal-conv1d, v2.dec.runtime_check)",
        "fla_loaded": any(
            name == "fla" or name.startswith("fla.") for name in sys.modules
        ),
        "fla_site": runtime["fla_site"],
        "fla_site_inserted": runtime["fla_site_inserted"],
        "fla_origin": runtime["fla_origin"],
        "kernel_bindings": runtime["kernel_bindings"],
        "kernel_bindings_after": after["kernel_bindings"],
        "triton_cache_dir": runtime["triton_cache_dir"],
        "triton_cache_autotuning": runtime["triton_cache_autotuning"],
        "hip_force_dev_kernarg": os.environ.get("HIP_FORCE_DEV_KERNARG"),
        "versions": {
            **versions(),
            "flash_linear_attention": runtime["flash_linear_attention"],
            "causal_conv1d": runtime["causal_conv1d"],
            "triton": runtime["triton"],
        },
        "max_length": int(args[args.index("--max-length") + 1]),
        "max_items": max_items,
        "longest": longest,
        "infer_wall_seconds": wall,
        "memory": memory(),
        "sys_path_head": sys.path[:6],
    }
    if (
        not record["fla_loaded"]
        or after["kernel_bindings"] != runtime["kernel_bindings"]
    ):
        raise SystemExit("the kernel path was not used for the whole collection")
    output.with_name(output.name + ".runtime.json").write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
