#!/usr/bin/env python3
"""Rewrite ONNX patterns that block MIGraphX ownership.

This is intentionally an artifact-preparation tool, not a runtime mutation.
Run it after exporting an ONNX artifact and before uploading or benchmarking the
MIGraphX candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite known ONNX patterns into MIGraphX-safe equivalents."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input ONNX file.")
    parser.add_argument("--output", required=True, type=Path, help="Output ONNX file.")
    parser.add_argument(
        "--rewrite-skip-layer-normalization",
        action="store_true",
        help=(
            "Rewrite com.microsoft::SkipLayerNormalization into standard "
            "Add + LayerNormalization. This avoids MIGraphX 2.15 rejecting "
            "SkipLayerNormalization nodes with an empty beta input."
        ),
    )
    parser.add_argument(
        "--fail-if-unchanged",
        action="store_true",
        help="Exit non-zero if no node was rewritten.",
    )
    return parser.parse_args()


def rewrite_skip_layer_normalization(model: onnx.ModelProto) -> int:
    rewritten = 0
    nodes = []
    for node in model.graph.node:
        if node.domain == "com.microsoft" and node.op_type == "SkipLayerNormalization":
            if len(node.input) < 3:
                raise ValueError(
                    f"{node.name or node.op_type} has {len(node.input)} inputs; "
                    "expected at least input, skip, and gamma"
                )

            hidden, skip, gamma = node.input[:3]
            beta = node.input[3] if len(node.input) > 3 else ""
            bias = node.input[4] if len(node.input) > 4 else ""
            base_name = node.name or "SkipLayerNormalization"
            residual = f"{base_name}_migraphx_residual"
            add_inputs = [hidden, skip]
            if bias:
                biased = f"{base_name}_migraphx_bias"
                nodes.append(
                    helper.make_node(
                        "Add",
                        add_inputs,
                        [biased],
                        name=f"{base_name}_MIGraphX_AddSkip",
                    )
                )
                add_inputs = [biased, bias]
            nodes.append(
                helper.make_node(
                    "Add",
                    add_inputs,
                    [residual],
                    name=f"{base_name}_MIGraphX_Add",
                )
            )
            ln_inputs = [residual, gamma]
            if beta:
                ln_inputs.append(beta)
            attrs = {
                attr.name: helper.get_attribute_value(attr) for attr in node.attribute
            }
            nodes.append(
                helper.make_node(
                    "LayerNormalization",
                    ln_inputs,
                    list(node.output),
                    name=f"{base_name}_MIGraphX_LayerNormalization",
                    **attrs,
                )
            )
            rewritten += 1
        else:
            nodes.append(node)

    if rewritten:
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    return rewritten


def main() -> None:
    args = parse_args()
    if not args.rewrite_skip_layer_normalization:
        raise SystemExit("No rewrite selected.")

    model = onnx.load(args.input, load_external_data=True)
    rewrites = 0
    if args.rewrite_skip_layer_normalization:
        rewrites += rewrite_skip_layer_normalization(model)

    if args.fail_if_unchanged and rewrites == 0:
        raise SystemExit("No nodes were rewritten.")

    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(
        f"Wrote {args.output} with {rewrites} MIGraphX-safe rewrite"
        f"{'' if rewrites == 1 else 's'}."
    )


if __name__ == "__main__":
    main()
