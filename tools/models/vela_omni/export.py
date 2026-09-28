#!/usr/bin/env python3
# Phase-local imports preserve memory limits and authenticated source loading.
# ruff: noqa: PLC0415
"""Export reviewed pinned Nano/Mini artifacts; publish a manifest only after parity.

Example: python tools/models/vela_omni/export.py --variant nano --source /models/nano
         --output /models/nano-ort
The source must contain the exact snapshot in sources.json. No network access is
needed with --source. --download explicitly provisions that immutable snapshot.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from contract import (
    MANIFEST,
    PENDING_MANIFEST,
    VARIANTS,
    artifact_manifest,
    inventory,
    write_json,
)
from source import load_reference, provision


def require_versions() -> None:
    for name, expected in (
        ("transformers", "4.57.6"),
        ("torch", "2.8.0"),
        ("torchaudio", "2.8.0"),
    ):
        actual = importlib.metadata.version(name).split("+")[0]
        if actual != expected:
            raise ValueError(f"{name} must be {expected}; found {actual}")


def graph_nodes(graph):
    """Visit control-flow bodies as well as top-level tensor operators."""
    import onnx

    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from graph_nodes(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from graph_nodes(child)


def validate_portable_graph(graph, name):
    for node in graph_nodes(graph):
        if node.domain not in ("", "ai.onnx"):
            raise ValueError(f"{name} graph contains nonportable custom operators")
        if (
            name == "clap"
            and node.op_type == "Resize"
            and any(
                attribute.name == "mode" and attribute.s == b"cubic"
                for attribute in node.attribute
            )
        ):
            raise ValueError("CLAP graph still contains unsupported cubic Resize")


def export_graph(source: Path, output: Path, name: str) -> dict:
    import onnx
    import torch
    from graphs import (
        AudioGraph,
        ClapGraph,
        ImageGraph,
        TextGraph,
    )
    from processors import (
        export_processors,
    )

    reference = load_reference(source)
    manifest = artifact_manifest(
        reference.variant,
        reference.tokenizer.padding_side,
        reference.tokenizer.pad_token_id,
    )
    if name == "text":
        export_processors(reference, source, output)

    constructors = {
        "text": TextGraph,
        "image": ImageGraph,
        "clap": ClapGraph,
        "audio": AudioGraph,
    }
    size = VARIANTS[reference.variant]["image_size"]
    sample = reference.tokenizer(
        "A small boat beside a wooden pier.", return_tensors="pt"
    )
    examples = {
        "text": (sample["input_ids"], sample["attention_mask"]),
        "image": (torch.zeros(1, 3, size, size),),
        "clap": (torch.zeros(1, 1, 1001, 64),),
        "audio": (torch.zeros(1, 80, 3000), torch.ones(1, 512) / (512**0.5)),
    }
    started = time.monotonic()
    print(f"Exporting {reference.variant} {name}", flush=True)
    module = constructors[name](reference).eval()
    # The wrapper owns only this modality. Free unrelated native components
    # before tracing/serialization, and return allocator state on process exit.
    del reference
    gc.collect()
    path = output / manifest["graphs"][name]["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = (
        {"input_ids": {1: "sequence"}, "attention_mask": {1: "sequence"}}
        if name == "text"
        else None
    )
    with torch.inference_mode():
        torch.onnx.export(
            module,
            examples[name],
            str(path),
            input_names=[item["name"] for item in manifest["graphs"][name]["inputs"]],
            output_names=["embedding"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )
    graph = onnx.load(str(path), load_external_data=False)
    expected_names = [item["name"] for item in manifest["graphs"][name]["inputs"]]
    if [item.name for item in graph.graph.input] != expected_names or [
        item.name for item in graph.graph.output
    ] != ["embedding"]:
        raise ValueError(f"{name} graph input/output contract changed")
    # The normalizer leaves symbolic output dimensions in legacy export,
    # although batch=1 and the native projection width are fixed. State the
    # real readout shape explicitly; runtime must not guess missing shapes.
    output_type = graph.graph.output[0].type.tensor_type
    declared = manifest["graphs"][name]["output"]
    if output_type.elem_type != onnx.TensorProto.FLOAT or len(
        output_type.shape.dim
    ) != len(declared["shape"]):
        raise ValueError(f"{name} graph output type/rank changed")
    for dimension, expected in zip(
        output_type.shape.dim, declared["shape"], strict=True
    ):
        if dimension.HasField("dim_value") and dimension.dim_value != expected:
            raise ValueError(f"{name} graph output dimension changed")
        dimension.ClearField("dim_param")
        dimension.dim_value = expected
    onnx.save_model(graph, str(path))
    # Use the path: Mini external data can exceed protobuf's 2 GB limit.
    onnx.checker.check_model(str(path), full_check=True)
    validate_portable_graph(graph.graph, name)
    del module, graph
    gc.collect()
    print(f"Exported {name} in {time.monotonic() - started:.1f}s", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "export", "verify"), default="all")
    parser.add_argument(
        "--graph", choices=("text", "image", "clap", "audio"), help=argparse.SUPPRESS
    )
    parser.add_argument("--reference-device", default="cpu")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--max-address-space-gib",
        type=int,
        help="optional per-process memory guard for large CPU qualification runs",
    )
    parser.add_argument(
        "--full-context",
        action="store_true",
        help="also execute the exact public text budget; may need substantial memory",
    )
    args = parser.parse_args()
    if args.max_address_space_gib is not None:
        if args.max_address_space_gib < 1:
            parser.error("--max-address-space-gib must be positive")
        limit = args.max_address_space_gib * 1024**3
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    require_versions()
    import torch
    from parity import (
        verify,
    )

    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    source = provision(args.variant, args.source, args.download)
    if args.output.resolve().is_relative_to(
        source.resolve()
    ) or source.resolve().is_relative_to(args.output.resolve()):
        parser.error("output must be separate from the native source")
    if args.graph and args.stage != "export":
        parser.error("internal --graph requires --stage export")
    if args.graph and (args.output / MANIFEST).exists():
        parser.error("refusing to overwrite a verified artifact")
    if (
        not args.graph
        and args.stage != "verify"
        and args.output.exists()
        and any(args.output.iterdir())
    ):
        parser.error(
            "export output must be an empty directory; use --stage verify for a pending export"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.graph:
        manifest = export_graph(source, args.output, args.graph)
        manifest["files"] = inventory(args.output)
        write_json(args.output / PENDING_MANIFEST, manifest)
        return
    if args.stage != "verify":
        # Isolation bounds native weights + ONNX serialization to one modality.
        # It also returns allocator arenas between large graph exports.
        for graph in ("text", "image", "clap", "audio"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--variant",
                    args.variant,
                    "--source",
                    str(source),
                    "--output",
                    str(args.output),
                    "--stage",
                    "export",
                    "--graph",
                    graph,
                    "--threads",
                    str(args.threads),
                ],
                check=True,
            )
    if args.stage in ("all", "verify"):
        verify(
            load_reference(source),
            args.output,
            args.reference_device,
            args.provider,
            args.threads,
            args.full_context,
        )
        print(f"Verified artifact: {args.output / MANIFEST}", flush=True)
    else:
        print(
            f"Export is pending reference parity: {args.output / PENDING_MANIFEST}",
            flush=True,
        )


if __name__ == "__main__":
    started = time.monotonic()
    try:
        main()
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        print(
            json.dumps(
                {
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "peak_rss_bytes": int(usage.ru_maxrss)
                    * (1 if sys.platform == "darwin" else 1024),
                    "largest_child_peak_rss_bytes": int(children.ru_maxrss)
                    * (1 if sys.platform == "darwin" else 1024),
                }
            ),
            flush=True,
        )
