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
from stable_pooling import stabilize_mean_pooling

WHERE_INPUT_COUNT = 3
MASKED_ATTENTION_MAX = -10000.0
ROPE_FREQUENCY_RANK = 3  # [1, rotary_dimension / 2, 1]


def build_maps(graph):
    out2node = {}
    name2node = {}
    for n in graph.node:
        name2node[n.name] = n
        for o in n.output:
            out2node[o] = n
    return out2node, name2node


def scalar_value(graph, out2node, name):
    """Read a scalar export constant without materialising any model weights."""
    tensor = next((t for t in graph.initializer if t.name == name), None)
    if tensor is not None and math.prod(tensor.dims) == 1:
        return float(numpy_helper.to_array(tensor).item())
    node = out2node.get(name)
    if node is None:
        return None
    if node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value" and math.prod(attr.t.dims) == 1:
                return float(numpy_helper.to_array(attr.t).item())
    if node.op_type == "Cast":
        return scalar_value(graph, out2node, node.input[0])
    return None


def attention_probability_path(graph, out2node, softmax):
    """Accept only direct probabilities or Where(IsNaN(p), scalar_zero, p)."""
    probability = softmax.output[0]
    extra = []
    for node in graph.node:
        if node.op_type != "Where" or len(node.input) != WHERE_INPUT_COUNT:
            continue
        guard = out2node.get(node.input[0])
        if (
            node.input[2] == probability
            and guard is not None
            and guard.op_type == "IsNaN"
            and list(guard.input) == [probability]
            and scalar_value(graph, out2node, node.input[1]) == 0
        ):
            probability = node.output[0]
            extra = [guard, node]
            break
    consumers = [
        n for n in graph.node if n.op_type == "MatMul" and n.input[0] == probability
    ]
    return consumers, extra


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

        av_consumers, probability_nodes = attention_probability_path(
            graph, out2node, sm
        )
        if len(av_consumers) != 1:
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
                "probability_nodes": probability_nodes,
                "q_tensor": q_mul.input[0],
                "k_tensor": k_tensor,
                "v_tensor": av_mm.input[1],
                "mask_tensor": add_mask.input[1],
                "output_tensor": av_mm.output[0],
            }
        )

    return blocks


def compute_scale(graph, out2node, q_mul_node, k_mul_node=None, hdim=64):
    q_scale = scalar_value(graph, out2node, q_mul_node.input[1])
    k_scale = scalar_value(graph, out2node, (k_mul_node or q_mul_node).input[1])
    if q_scale is not None and k_scale is not None:
        return q_scale * k_scale
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
    return 1.0 / math.sqrt(hdim)


def inverted_padding_fill(graph, out2node, mask):
    """Recognise HF's Where(bool(1-padding), negative, 1-padding) export.

    The broadcast axes are part of the contract: attention_mask [B,S] becomes
    [B,1,1,S], so only key padding is masked. Do not accept arbitrary inversions.
    """
    condition = out2node.get(mask.input[0])
    if (
        condition is None
        or condition.op_type != "Cast"
        or list(condition.input) != [mask.input[2]]
        or not any(
            a.name == "to" and a.i == TensorProto.BOOL for a in condition.attribute
        )
    ):
        return False
    sub = out2node.get(mask.input[2])
    if (
        sub is None
        or sub.op_type != "Sub"
        or scalar_value(graph, out2node, sub.input[0]) != 1
    ):
        return False
    padding = out2node.get(sub.input[1])
    if padding is not None and padding.op_type == "Cast":
        padding = out2node.get(padding.input[0])
    if padding is None or padding.op_type != "Expand":
        return False
    padding = out2node.get(padding.input[0])
    if (
        padding is None
        or padding.op_type != "Unsqueeze"
        or padding.input[0] != "attention_mask"
    ):
        return False
    axes = next((t for t in graph.initializer if t.name == padding.input[1]), None)
    if axes is None:
        node = out2node.get(padding.input[1])
        if node is not None and node.op_type == "Constant":
            axes = next((a.t for a in node.attribute if a.name == "value"), None)
    return axes is not None and list(numpy_helper.to_array(axes)) == [1, 2]


def attention_window(graph, out2node, mask_tensor_name):
    """Read symmetric positional windows from mask semantics, never names/counts.

    ModernBERT exports abs(query_position - key_position) <= local_attention/2.
    Both endpoints are inclusive: a radius of 64 means (64, 64), not (63, 64).
    Unknown comparisons are refused instead of silently becoming global attention.
    """
    mask = out2node.get(mask_tensor_name)
    if mask is None or mask.op_type != "Where" or len(mask.input) != WHERE_INPUT_COUNT:
        raise ValueError(f"unsupported additive attention mask {mask_tensor_name!r}")
    keep = scalar_value(graph, out2node, mask.input[1])
    masked = scalar_value(graph, out2node, mask.input[2])
    positive_condition = (
        keep == 0 and masked is not None and masked <= MASKED_ATTENTION_MAX
    )
    negative_condition = False
    if keep is not None and keep <= MASKED_ATTENTION_MAX:
        negative_condition = inverted_padding_fill(graph, out2node, mask)
        condition = out2node.get(mask.input[0])
        if condition is not None and condition.op_type == "Not":
            window = out2node.get(condition.input[0])
            while window is not None and window.op_type == "Unsqueeze":
                window = out2node.get(window.input[0])
            if window is not None and window.op_type in {"Less", "LessOrEqual"}:
                # HF applies the window after the padding mask. A recursive
                # global-mask check rejects reversed or otherwise unknown fills.
                negative_condition = attention_window(
                    graph, out2node, mask.input[2]
                ) == (-1, -1)
    if not positive_condition and not negative_condition:
        raise ValueError(
            f"mask {mask_tensor_name!r} must map allowed positions to zero"
        )
    ancestors = {}
    pending = [mask_tensor_name]
    leaves = set()
    while pending:
        name = pending.pop()
        node = out2node.get(name)
        if node is None:
            leaves.add(name)
        elif node.name not in ancestors:
            ancestors[node.name] = node
            pending.extend(node.input)
    if "attention_mask" not in leaves:
        raise ValueError(f"mask {mask_tensor_name!r} does not depend on attention_mask")

    def position_range(name):
        node = out2node.get(name)
        while node is not None:
            if node.op_type in {
                "Cast",
                "Identity",
                "Unsqueeze",
                "Squeeze",
                "Reshape",
            } or (
                node.op_type == "Transpose"
                and any(
                    a.name == "perm" and list(a.ints) == [1, 0] for a in node.attribute
                )
            ):
                node = out2node.get(node.input[0])
            else:
                break
        return node.output[0] if node is not None and node.op_type == "Range" else None

    radii = set()
    for node in ancestors.values():
        if node.op_type not in {
            "Less",
            "LessOrEqual",
            "Greater",
            "GreaterOrEqual",
            "Equal",
        }:
            continue
        lhs = out2node.get(node.input[0])
        bound = scalar_value(graph, out2node, node.input[1])
        if (
            node.op_type == "GreaterOrEqual"
            and bound == 0
            and position_range(node.input[0]) is not None
        ):
            continue  # exporter bounds-checks non-negative query positions
        if node.op_type in {"Less", "LessOrEqual"} and lhs is not None:
            sub = out2node.get(lhs.input[0]) if lhs.op_type == "Abs" else None
            if sub is not None and sub.op_type == "Sub" and bound is not None:
                q_pos, k_pos = map(position_range, sub.input)
                radius = bound - (1 if node.op_type == "Less" else 0)
                if (
                    q_pos is not None
                    and q_pos == k_pos
                    and radius >= 0
                    and radius.is_integer()
                ):
                    radii.add(int(radius))
                    continue
        raise ValueError(f"unsupported attention-mask comparison {node.name!r}")
    if len(radii) > 1:
        raise ValueError(f"mask {mask_tensor_name!r} has conflicting local windows")
    return (next(iter(radii)),) * 2 if radii else (-1, -1)


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


def rotary_fp32_initializers(graph, out2node):
    """Identify FP32 RoPE frequencies used only with positions, then cast to FP16.

    Torch exports this intentional FP32 numerical island as an initializer.
    It is not an encoder/head weight, and must not be narrowed to satisfy the
    weight-name guard. Require the complete observed position-to-sin/cos path.
    """
    consumers = {}
    for node in graph.node:
        for name in set(node.input):
            consumers.setdefault(name, []).append(node)
    graph_outputs = {v.name for v in graph.output}

    def only_consumer(name, op):
        uses = consumers.get(name, [])
        if name not in graph_outputs and len(uses) == 1 and uses[0].op_type == op:
            return uses[0]
        return None

    def attr(node, name):
        return next(
            (helper.get_attribute_value(a) for a in node.attribute if a.name == name),
            None,
        )

    exempt = set()
    for tensor in graph.initializer:
        if (
            tensor.data_type != TensorProto.FLOAT
            or len(tensor.dims) != ROPE_FREQUENCY_RANK
        ):
            continue
        if tensor.dims[0] != 1 or tensor.dims[2] != 1:
            continue
        multiply = only_consumer(tensor.name, "MatMul")
        if multiply is None or multiply.input[0] != tensor.name:
            continue
        position = out2node.get(multiply.input[1])
        if position is None or position.op_type != "Cast" or attr(position, "to") != 1:
            continue
        position = out2node.get(position.input[0])
        while position is not None and position.op_type == "Unsqueeze":
            position = out2node.get(position.input[0])
        if (
            position is None
            or position.op_type != "Range"
            or scalar_value(graph, out2node, position.input[0]) != 0
            or scalar_value(graph, out2node, position.input[2]) != 1
        ):
            continue
        transpose = only_consumer(multiply.output[0], "Transpose")
        if transpose is None or attr(transpose, "perm") != [0, 2, 1]:
            continue
        concat = only_consumer(transpose.output[0], "Concat")
        if (
            concat is None
            or list(concat.input) != [transpose.output[0]] * 2
            or attr(concat, "axis") not in (-1, 2)
            or concat.output[0] in graph_outputs
        ):
            continue
        trig = consumers.get(concat.output[0], [])
        expected_trig = {"Cos", "Sin"}
        if (
            len(trig) != len(expected_trig)
            or {n.op_type for n in trig} != expected_trig
        ):
            continue
        casts = [only_consumer(n.output[0], "Cast") for n in trig]
        if all(n is not None and attr(n, "to") == TensorProto.FLOAT16 for n in casts):
            exempt.add(tensor.name)
    return exempt


def weight_precision(initializers, *, position_constants=frozenset()):
    """Return the one floating-point elem_type every weight tensor shares.

    Integer initializers (shapes, gather indices) carry no precision and are
    ignored. A graph with no floating-point weights, with more than one
    floating-point precision, or with a precision the CK kernel does not take
    (only FLOAT and FLOAT16 are) is refused, since neither name would describe
    it.
    """
    counts = {}
    for tensor in initializers:
        if tensor.data_type in _FLOAT_TYPES and tensor.name not in position_constants:
            counts[tensor.data_type] = counts.get(tensor.data_type, 0) + 1
    if not counts:
        raise ValueError("the graph holds no floating-point weight tensors")
    if len(counts) > 1:
        found = ", ".join(
            f"{n} {TensorProto.DataType.Name(t)}" for t, n in sorted(counts.items())
        )
        raise ValueError(f"the graph mixes weight precisions: {found}")
    (elem_type,) = counts
    if elem_type not in (TensorProto.FLOAT, TensorProto.FLOAT16):
        raise ValueError(
            f"the graph holds {TensorProto.DataType.Name(elem_type)} weights; "
            "CKFlashAttention takes FLOAT or FLOAT16"
        )
    return elem_type


_FLOAT_TYPES = (
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    TensorProto.BFLOAT16,
    TensorProto.DOUBLE,
)


def output_precision_conflict(output_path, model_is_fp16):
    """Return a message when the output name claims a precision the graph lacks.

    find_onnx_models ranks candidates by filename and puts model_fa_fp16.onnx
    first whenever CK Flash Attention is available, so the name is a contract
    rather than a label. The rewriter keeps the weights as it finds them and,
    for an FP32 graph, only casts around each CKFlashAttention node. Running it
    on model.onnx under an fp16 name is what produced the FP32
    model_fa_fp16.onnx in five of the six published 32K heads (#3256): a valid
    graph with twice the memory of the file its name promises.
    """
    name = os.path.basename(output_path).lower()
    if "fp16" in name and not model_is_fp16:
        return (
            f"{output_path}: the input graph holds FP32 weights, so this file would "
            "hold FP32 weights under an fp16 name. Name an FP32 rewrite "
            "model_fa.onnx; an fp16 name needs an FP16 input graph."
        )
    return None


def enforce_output_precision(output_path, model_is_fp16):
    """Exit on an fp16 name over FP32 weights; warn on the reverse."""
    conflict = output_precision_conflict(output_path, model_is_fp16)
    if conflict:
        print(f"error: {conflict}", file=sys.stderr)
        sys.exit(1)
    if model_is_fp16 and "fp16" not in os.path.basename(output_path).lower():
        print(
            f"warning: {output_path} will hold FP16 weights but its name does not "
            "say so; find_onnx_models ranks FP16 candidates by name."
        )


def rewrite(model_path, output_path, hdim=64, local_attention=128):
    model = onnx.load(model_path)
    graph = model.graph

    out2node, _name2node = build_maps(graph)
    blocks = find_attention_blocks(graph, out2node)

    if not blocks:
        print("No attention blocks found -- nothing to rewrite.")
        sys.exit(1)

    softmax_count = sum(n.op_type == "Softmax" for n in graph.node)
    if len(blocks) != softmax_count:
        raise ValueError(
            f"matched {len(blocks)} of {softmax_count} Softmax nodes; "
            "refusing a partial rewrite that retains dense attention"
        )
    windows = {
        b["mask_tensor"]: attention_window(graph, out2node, b["mask_tensor"])
        for b in blocks
    }
    for window in windows.values():
        if window[0] >= 0 and window != (local_attention // 2,) * 2:
            raise ValueError(
                f"source mask window {window} disagrees with "
                f"--local-attention={local_attention}"
            )
    n_local = sum(windows[b["mask_tensor"]][0] >= 0 for b in blocks)
    n_global = len(blocks) - n_local
    print(f"Found {len(blocks)} attention blocks")
    print(
        f"  Local attention layers: {n_local} (source windows={set(windows.values())})"
    )
    print(f"  Global attention layers: {n_global}")

    # Create 1-D padding bias nodes
    pad_bias_nodes, pad_bias_tensor = create_1d_padding_bias_nodes(graph)

    # Detect model precision from the weights themselves. value_info is
    # optional and describes activations, so a valid FP16 export that carries
    # none would read as FP32, and one activation cannot vouch for every weight.
    try:
        position_constants = rotary_fp32_initializers(graph, out2node)
        model_is_fp16 = (
            weight_precision(graph.initializer, position_constants=position_constants)
            == TensorProto.FLOAT16
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    if model_is_fp16:
        print("  Model precision: FP16 (skipping input/output Cast nodes)")
    else:
        print("  Model precision: FP32 (adding fp32↔fp16 Cast nodes)")
    if position_constants:
        print(f"  Preserved {len(position_constants)} FP32 RoPE frequency constants")

    enforce_output_precision(output_path, model_is_fp16)

    # Collect nodes to remove (attention subgraph)
    nodes_to_remove = set()
    new_nodes = []

    for i, blk in enumerate(blocks):
        scale = compute_scale(graph, out2node, blk["q_mul"], blk["k_mul"], hdim)

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
        for extra in blk["probability_nodes"]:
            nodes_to_remove.add(extra.name)

        fa_wl, fa_wr = windows[blk["mask_tensor"]]

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
    if any(output in windows for node in kept for output in node.output):
        raise ValueError(
            "dense attention mask is still live after rewriting every layer"
        )
    kept.extend(pad_bias_nodes)
    kept.extend(new_nodes)

    n_removed = len(graph.node) - len(kept) + len(pad_bias_nodes) + len(new_nodes)
    orig_count = len(graph.node)

    del graph.node[:]
    graph.node.extend(kept)
    pooling_repairs = stabilize_mean_pooling(graph)
    print(f"  Stabilized {pooling_repairs} FP16 masked mean-pooling paths")

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

    # Replacement nodes were appended after their original consumers. Keep the
    # serialized graph valid without depending on an ORT-specific reordering.
    available = (
        {v.name for v in graph.input} | {t.name for t in graph.initializer} | {""}
    )
    pending = list(graph.node)
    ordered = []
    while pending:
        ready = [n for n in pending if all(name in available for name in n.input)]
        if not ready:
            raise ValueError("rewritten graph has unresolved inputs or a cycle")
        for node in ready:
            ordered.append(node)
            available.update(node.output)
        ready_names = {n.name for n in ready}
        pending = [n for n in pending if n.name not in ready_names]
    del graph.node[:]
    graph.node.extend(ordered)

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
