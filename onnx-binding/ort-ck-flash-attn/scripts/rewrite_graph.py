#!/usr/bin/env python3
"""
Rewrite an mmBERT ONNX graph to replace the dense attention subgraph
with a single com.ck::CKFlashAttention custom-op node per layer.

Key optimisations over the naive rewrite:
  - Sliding-window (local) attention is handled via the CK kernel's built-in
    window_size_left / window_size_right parameters instead of materialising a
    dense [1, 1, seq, seq] mask tensor.
  - The 2-D attention mask subgraph (Expand, Where, etc.) is replaced by a
    lightweight 1-D padding bias [B, 1, 1, seq] derived directly from the
    attention_mask input.  This reduces mask memory from O(n^2) to O(n).
"""

import argparse
import math
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_maps(graph):
    out2node = {}
    name2node = {}
    for n in graph.node:
        name2node[n.name] = n
        for o in n.output:
            out2node[o] = n
    return out2node, name2node


def find_attention_blocks(graph, out2node):
    blocks = []
    softmaxes = [n for n in graph.node if n.op_type == "Softmax"]

    for sm in softmaxes:
        add_mask = out2node.get(sm.input[0])
        if not add_mask or add_mask.op_type != "Add":
            continue

        qk_mm = out2node.get(add_mask.input[0])
        if not qk_mm or qk_mm.op_type != "MatMul":
            continue

        q_mul = out2node.get(qk_mm.input[0])
        if not q_mul or q_mul.op_type != "Mul":
            continue

        k_mul = out2node.get(qk_mm.input[1])
        if not k_mul or k_mul.op_type != "Mul":
            continue

        # K path: either direct Transpose (classifier) or Reshape->Transpose->Reshape (embedding)
        k_transpose = None
        k_extra_nodes = []
        k_input0 = out2node.get(k_mul.input[0])
        if k_input0 and k_input0.op_type == "Transpose":
            k_transpose = k_input0
        elif k_input0 and k_input0.op_type == "Reshape":
            inner = out2node.get(k_input0.input[0])
            if inner and inner.op_type == "Transpose":
                k_transpose = inner
                k_extra_nodes.append(k_input0)
                pre_reshape = out2node.get(inner.input[0])
                if pre_reshape and pre_reshape.op_type == "Reshape":
                    k_extra_nodes.append(pre_reshape)

        if not k_transpose:
            continue

        av_consumers = [
            n for n in graph.node if sm.output[0] in n.input and n.op_type == "MatMul"
        ]
        if not av_consumers:
            continue
        av_mm = av_consumers[0]

        # For K tensor: use input of the outermost Reshape (before transpose chain)
        # or Transpose input directly if no wrapping Reshapes
        if k_extra_nodes:
            pre_transpose_reshape = out2node.get(k_transpose.input[0])
            k_tensor = (
                pre_transpose_reshape.input[0]
                if (
                    pre_transpose_reshape and pre_transpose_reshape.op_type == "Reshape"
                )
                else k_transpose.input[0]
            )
        else:
            k_tensor = k_transpose.input[0]

        blocks.append(
            {
                "softmax": sm,
                "add_mask": add_mask,
                "qk_matmul": qk_mm,
                "q_mul": q_mul,
                "k_mul": k_mul,
                "k_transpose": k_transpose,
                "k_extra_nodes": k_extra_nodes,
                "av_matmul": av_mm,
                "q_tensor": q_mul.input[0],
                "k_tensor": k_tensor,
                "v_tensor": av_mm.input[1],
                "mask_tensor": add_mask.input[1],
                "output_tensor": av_mm.output[0],
            }
        )

    return blocks


def compute_scale(graph, out2node, q_mul_node):
    try:
        scale_name = q_mul_node.input[1]
        sqrt_node = out2node[scale_name]
        cast_node = out2node[sqrt_node.input[0]]
        div_node = out2node[cast_node.input[0]]
        sqrt_hdim = out2node[div_node.input[1]]
        hdim_node = out2node[sqrt_hdim.input[0]]
        if hdim_node.op_type == "Constant":
            hdim_val = float(numpy_helper.to_array(hdim_node.attribute[0].t))
            return 1.0 / math.sqrt(hdim_val)
    except (KeyError, IndexError):
        pass
    return 1.0 / math.sqrt(64.0)


def classify_mask(mask_tensor_name, local_mask_name=None):
    """Determine if a mask tensor represents local or global attention.

    Two naming conventions are supported:
      - Classifier models:  Where_1 = local, Where_2 = global
      - Embedding models:   masked_fill = local, masked_fill_1 = global
    """
    # Classifier convention
    if "Where_1" in mask_tensor_name:
        return "local"
    if "Where_2" in mask_tensor_name:
        return "global"
    # Embedding convention (fewer layers use the base mask = local)
    if local_mask_name is not None:
        return "local" if mask_tensor_name == local_mask_name else "global"
    if mask_tensor_name == "masked_fill":
        return "local"
    if "masked_fill" in mask_tensor_name:
        return "global"
    return "unknown"


def find_mask_only_nodes(graph, out2node):
    """Find nodes that are exclusively part of the 2-D mask computation."""
    mask_roots = set()
    for n in graph.node:
        if n.op_type == "Where" and (
            n.name.startswith("/model/Where") or n.name.startswith("node_masked_fill")
        ):
            mask_roots.add(n.name)

    ancestors = set()
    visited = set()

    def trace_back(name):
        if name in visited:
            return
        visited.add(name)
        if name in out2node:
            n = out2node[name]
            ancestors.add(n.name)
            for inp in n.input:
                trace_back(inp)

    for n in graph.node:
        if n.name in mask_roots:
            for inp in n.input:
                trace_back(inp)
    ancestors.update(mask_roots)

    all_node_names = {n.name for n in graph.node}
    mask_only = set()
    for name in ancestors:
        node = next((n for n in graph.node if n.name == name), None)
        if not node:
            continue
        all_consumers_mask = True
        for out in node.output:
            consumers = [n2 for n2 in graph.node if out in n2.input]
            if not all(c.name in ancestors for c in consumers):
                all_consumers_mask = False
                break
        if all_consumers_mask:
            mask_only.add(name)

    return mask_only


def create_1d_padding_bias_nodes(graph):
    """
    Create ONNX nodes that derive a 1-D padding bias from attention_mask.

    attention_mask [B, seq] (int64, 1=valid 0=padding)
      -> Cast to float16
      -> Sub(1.0, ...)          = 1.0 for padding, 0.0 for valid
      -> Mul(-65504.0, ...)     = -65504 for padding, 0.0 for valid
      -> Reshape to [B, 1, 1, seq]

    Returns (nodes_list, output_tensor_name).
    """
    nodes = []
    prefix = "/model/_fa_pad_bias"

    # Cast attention_mask to float16
    cast_out = f"{prefix}/Cast_output_0"
    nodes.append(
        helper.make_node(
            "Cast",
            inputs=["attention_mask"],
            outputs=[cast_out],
            name=f"{prefix}/Cast",
            to=TensorProto.FLOAT16,
        )
    )

    # Sub(1.0, cast_out) -> inverted mask
    one_const = f"{prefix}/one"
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[one_const],
            name=f"{prefix}/Constant_one",
            value=numpy_helper.from_array(
                np.array(1.0, dtype=np.float16), name=one_const
            ),
        )
    )

    sub_out = f"{prefix}/Sub_output_0"
    nodes.append(
        helper.make_node(
            "Sub", inputs=[one_const, cast_out], outputs=[sub_out], name=f"{prefix}/Sub"
        )
    )

    # Mul(-65504.0, sub_out) -> padding positions get -65504
    neg_inf_const = f"{prefix}/neg_inf"
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[neg_inf_const],
            name=f"{prefix}/Constant_neg_inf",
            value=numpy_helper.from_array(
                np.array(-65504.0, dtype=np.float16), name=neg_inf_const
            ),
        )
    )

    mul_out = f"{prefix}/Mul_output_0"
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=[sub_out, neg_inf_const],
            outputs=[mul_out],
            name=f"{prefix}/Mul",
        )
    )

    # Reshape to [B, 1, 1, seq] -- use a constant shape [-1 is inferred]
    # Actually use Unsqueeze with axes [1, 2] which is cleaner
    axes_const = f"{prefix}/axes"
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[axes_const],
            name=f"{prefix}/Constant_axes",
            value=numpy_helper.from_array(
                np.array([1, 2], dtype=np.int64), name=axes_const
            ),
        )
    )

    unsqueeze_out = f"{prefix}/Unsqueeze_output_0"
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            inputs=[mul_out, axes_const],
            outputs=[unsqueeze_out],
            name=f"{prefix}/Unsqueeze",
        )
    )

    return nodes, unsqueeze_out


def rewrite(model_path, output_path, hdim=64, local_attention=128):
    model = onnx.load(model_path)
    graph = model.graph

    out2node, name2node = build_maps(graph)
    blocks = find_attention_blocks(graph, out2node)

    if not blocks:
        print("No attention blocks found -- nothing to rewrite.")
        sys.exit(1)

    scale = compute_scale(graph, out2node, blocks[0]["q_mul"])
    print(f"Found {len(blocks)} attention blocks, scale={scale:.6f}")

    # Auto-detect the local mask name: the mask tensor used by the fewest
    # layers is the local (sliding-window) mask.
    mask_names = [b["mask_tensor"] for b in blocks]
    mask_freq = {}
    for m in mask_names:
        mask_freq[m] = mask_freq.get(m, 0) + 1
    local_mask_name = min(mask_freq, key=mask_freq.get) if mask_freq else None

    n_local = sum(
        1 for b in blocks if classify_mask(b["mask_tensor"], local_mask_name) == "local"
    )
    n_global = sum(
        1
        for b in blocks
        if classify_mask(b["mask_tensor"], local_mask_name) == "global"
    )
    print(f"  Local attention layers: {n_local} (window={local_attention})")
    print(f"  Global attention layers: {n_global}")

    # Window sizes for local attention (symmetric window around diagonal)
    wl = local_attention // 2 - 1  # 63 for window=128
    wr = local_attention // 2  # 64 for window=128

    # Create 1-D padding bias nodes
    pad_bias_nodes, pad_bias_tensor = create_1d_padding_bias_nodes(graph)

    # Build value_info type map for precision detection
    vi_type_map = {}
    for vi in graph.value_info:
        if vi.type.HasField("tensor_type"):
            vi_type_map[vi.name] = vi.type.tensor_type.elem_type

    # Detect model precision from Q tensor of first block
    first_q = blocks[0]["q_tensor"]
    model_is_fp16 = vi_type_map.get(first_q) == TensorProto.FLOAT16
    if model_is_fp16:
        print("  Model precision: FP16 (skipping input/output Cast nodes)")
    else:
        print("  Model precision: FP32 (adding fp32↔fp16 Cast nodes)")

    # Collect nodes to remove (attention subgraph)
    nodes_to_remove = set()
    new_nodes = []

    for i, blk in enumerate(blocks):
        layer_name = blk["softmax"].name.rsplit("/", 1)[0]
        mask_type = classify_mask(blk["mask_tensor"], local_mask_name)

        for key in (
            "q_mul",
            "k_mul",
            "k_transpose",
            "qk_matmul",
            "add_mask",
            "softmax",
            "av_matmul",
        ):
            nodes_to_remove.add(blk[key].name)
        for extra in blk.get("k_extra_nodes", []):
            nodes_to_remove.add(extra.name)

        # Set window attributes based on local vs global
        if mask_type == "local":
            fa_wl, fa_wr = wl, wr
        else:
            fa_wl, fa_wr = -1, -1

        # Derive a clean layer prefix for the FA node name
        sm_name = blk["softmax"].name
        if "/" in sm_name:
            fa_name = sm_name.rsplit("/", 1)[0] + "/CKFlashAttention"
        else:
            fa_name = f"CKFlashAttention_{i}"

        if model_is_fp16:
            # Model is already fp16: wire Q/K/V directly, output directly
            fa_node = helper.make_node(
                "CKFlashAttention",
                inputs=[
                    blk["q_tensor"],
                    blk["k_tensor"],
                    blk["v_tensor"],
                    pad_bias_tensor,
                ],
                outputs=[blk["output_tensor"]],
                name=fa_name,
                domain="com.ck",
                scale=scale,
                window_size_left=fa_wl,
                window_size_right=fa_wr,
            )
            new_nodes.append(fa_node)
        else:
            # Model is fp32: add Cast(fp32→fp16) for inputs, Cast(fp16→fp32) for output
            q_fp16 = f"{fa_name}/q_cast_fp16"
            k_fp16 = f"{fa_name}/k_cast_fp16"
            v_fp16 = f"{fa_name}/v_cast_fp16"
            out_fp16 = f"{fa_name}/out_fp16"

            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[blk["q_tensor"]],
                    outputs=[q_fp16],
                    name=f"{fa_name}/Cast_Q_fp16",
                    to=TensorProto.FLOAT16,
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[blk["k_tensor"]],
                    outputs=[k_fp16],
                    name=f"{fa_name}/Cast_K_fp16",
                    to=TensorProto.FLOAT16,
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[blk["v_tensor"]],
                    outputs=[v_fp16],
                    name=f"{fa_name}/Cast_V_fp16",
                    to=TensorProto.FLOAT16,
                )
            )

            fa_node = helper.make_node(
                "CKFlashAttention",
                inputs=[q_fp16, k_fp16, v_fp16, pad_bias_tensor],
                outputs=[out_fp16],
                name=fa_name,
                domain="com.ck",
                scale=scale,
                window_size_left=fa_wl,
                window_size_right=fa_wr,
            )
            new_nodes.append(fa_node)

            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[out_fp16],
                    outputs=[blk["output_tensor"]],
                    name=f"{fa_name}/Cast_out_fp32",
                    to=TensorProto.FLOAT,
                )
            )

    # Remove dead scale-computation nodes
    remaining_node_names = {n.name for n in graph.node} - nodes_to_remove
    for blk in blocks:
        scale_tensor = blk["q_mul"].input[1]
        scale_node = out2node.get(scale_tensor)
        if scale_node:
            consumers = [
                n
                for n in graph.node
                if scale_tensor in n.input and n.name in remaining_node_names
            ]
            if not consumers:
                nodes_to_remove.add(scale_node.name)
                for inp in scale_node.input:
                    parent = out2node.get(inp)
                    if parent:
                        p_consumers = [
                            n
                            for n in graph.node
                            if inp in n.input
                            and n.name in remaining_node_names
                            and n.name != scale_node.name
                        ]
                        if not p_consumers:
                            nodes_to_remove.add(parent.name)
                            remaining_node_names.discard(parent.name)
                remaining_node_names.discard(scale_node.name)

    # DCE: remove 2-D mask subgraph nodes (Expand, Where, etc.)
    # After CKFlashAttention nodes no longer reference Where_1/Where_2,
    # the entire mask computation subgraph becomes dead code.
    # Iteratively remove nodes whose outputs have no consumers.
    all_new_node_names = {n.name for n in new_nodes}
    all_pad_bias_names = {n.name for n in pad_bias_nodes}
    kept_names = (
        remaining_node_names | all_new_node_names | all_pad_bias_names
    ) - nodes_to_remove

    # Build a set of all tensor names consumed by new + pad-bias nodes
    # so DCE doesn't remove their producers.
    new_node_inputs = set()
    for n in new_nodes + pad_bias_nodes:
        new_node_inputs.update(n.input)

    graph_outputs = {o.name for o in graph.output}
    changed = True
    while changed:
        changed = False
        for n in list(graph.node):
            if n.name not in kept_names or n.name in nodes_to_remove:
                continue
            if n.name in all_new_node_names or n.name in all_pad_bias_names:
                continue
            # Check if any output is consumed (by existing nodes OR new nodes)
            all_dead = True
            for out in n.output:
                if out in graph_outputs:
                    all_dead = False
                    break
                if out in new_node_inputs:
                    all_dead = False
                    break
                consumers = [
                    n2.name
                    for n2 in graph.node
                    if out in n2.input and n2.name in kept_names
                ]
                if consumers:
                    all_dead = False
                    break
            if all_dead:
                nodes_to_remove.add(n.name)
                kept_names.discard(n.name)
                changed = True

    # Rebuild node list: keep original order, then append new nodes
    kept = [n for n in graph.node if n.name not in nodes_to_remove]
    kept.extend(pad_bias_nodes)
    kept.extend(new_nodes)

    n_removed = len(graph.node) - len(kept) + len(pad_bias_nodes) + len(new_nodes)
    orig_count = len(graph.node)

    del graph.node[:]
    graph.node.extend(kept)

    # For fp16 models, cast graph outputs from fp16 to fp32 so that older
    # ONNX Runtime host code (which only tries extract_tensor::<f32>) works.
    if model_is_fp16:
        output_cast_nodes = []
        for graph_out in graph.output:
            if (
                graph_out.type.HasField("tensor_type")
                and graph_out.type.tensor_type.elem_type == TensorProto.FLOAT16
            ):
                old_name = graph_out.name
                intermediate = f"{old_name}__fp16_raw"
                # Rename the last node's output from old_name -> intermediate
                for n in reversed(graph.node):
                    for idx, out in enumerate(n.output):
                        if out == old_name:
                            n.output[idx] = intermediate
                            break
                    else:
                        continue
                    break
                # Also rename in value_info
                for vi in graph.value_info:
                    if vi.name == old_name:
                        vi.name = intermediate
                # Add Cast(fp16→fp32) as final output
                output_cast_nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[intermediate],
                        outputs=[old_name],
                        name=f"/model/_output_cast/{old_name}",
                        to=TensorProto.FLOAT,
                    )
                )
                # Update graph output type to fp32
                graph_out.type.tensor_type.elem_type = TensorProto.FLOAT
                print(f"  Added output Cast(fp16→fp32) for {old_name}")
        graph.node.extend(output_cast_nodes)

    # Add com.ck opset import
    has_ck = any(op.domain == "com.ck" for op in model.opset_import)
    if not has_ck:
        model.opset_import.append(helper.make_opsetid("com.ck", 1))

    onnx.save(model, output_path)
    print(f"Saved rewritten model to {output_path}")
    print(f"  Removed {n_removed} nodes (attention + 2-D mask subgraph)")
    print(
        f"  Added {len(new_nodes)} CKFlashAttention + {len(pad_bias_nodes)} padding-bias nodes"
    )
    print(f"  Total nodes: {len(graph.node)} (was {orig_count})")


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite mmBERT ONNX model to use CKFlashAttention custom op "
        "with sliding-window masking and O(n) padding bias"
    )
    parser.add_argument("input", help="Path to input ONNX model")
    parser.add_argument("output", help="Path to output ONNX model")
    parser.add_argument(
        "--hdim", type=int, default=64, help="Head dimension (default: 64)"
    )
    parser.add_argument(
        "--local-attention",
        type=int,
        default=128,
        help="Local attention window size (default: 128)",
    )
    args = parser.parse_args()
    rewrite(args.input, args.output, args.hdim, args.local_attention)


if __name__ == "__main__":
    main()
