"""Express floating-point guards with widely supported standard ONNX operators."""

from onnx import AttributeProto, defs, helper


def _scopes(container):
    yield container
    for node in container.node:
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                yield from _scopes(attribute.g)
            elif attribute.type == AttributeProto.GRAPHS:
                for graph in attribute.graphs:
                    yield from _scopes(graph)


def _supports_self_equality(imports):
    version = next((item.version for item in imports if not item.domain), None)
    if version is None:
        return False
    try:
        predicate = defs.get_schema("IsNaN", version)
        equality = defs.get_schema("Equal", version)
    except defs.SchemaError:
        return False
    # Prove every legal input type, including generic local-function inputs.
    # Older Equal versions and newer IsNaN float8 types do not satisfy this.
    return set(predicate.type_constraints[0].allowed_type_strs).issubset(
        equality.type_constraints[0].allowed_type_strs
    )


def lower_nan_predicates(model) -> dict:
    """Replace IsNaN(x) with Not(Equal(x, x)) without removing NaN protection.

    IEEE self-equality is false precisely for NaN, including signed NaNs, and
    true for infinities and signed zeros. Keep the existing opsets and rewrite
    only scopes whose operator schemas prove that Equal supports every IsNaN
    input type. Custom-domain nodes and unsupported opsets stay unchanged.

    Nested graphs and local functions retain their scope, output names, tensor
    storage, and external-data references. Reserve names across all scopes so
    inserted temporaries cannot shadow a captured value. Mutates only nodes;
    callers must still validate the graph and qualify each execution provider.
    """
    roots = [(model.graph, model.opset_import)]
    roots.extend((function, function.opset_import) for function in model.functions)
    scopes = [
        (scope, _supports_self_equality(imports))
        for root, imports in roots
        for scope in _scopes(root)
    ]
    names = set()
    for scope, _ in scopes:
        for value in (*scope.input, *scope.output, *scope.value_info):
            names.add(value if isinstance(value, str) else value.name)
        for tensor in getattr(scope, "initializer", ()):
            names.add(tensor.name)
        for tensor in getattr(scope, "sparse_initializer", ()):
            names.update((tensor.values.name, tensor.indices.name))
        for node in scope.node:
            names.update((*node.input, *node.output, node.name))

    rewritten, unsupported = 0, 0
    # Rewrite children before their owners, whose protobuf insertion copies them.
    for scope, supported in reversed(scopes):
        nodes = []
        for node in scope.node:
            if node.domain or node.op_type != "IsNaN":
                nodes.append(node)
                continue
            if not supported:
                unsupported += 1
                nodes.append(node)
                continue
            if len(node.input) != 1 or len(node.output) != 1 or node.attribute:
                raise ValueError("Invalid standard IsNaN node")
            temporary = node.output[0] + "_self_equal"
            while temporary in names:
                temporary += "_"
            names.add(temporary)
            nodes.append(helper.make_node("Equal", [node.input[0]] * 2, [temporary]))
            replacement = type(node)()
            replacement.CopyFrom(node)
            replacement.op_type = "Not"
            replacement.input[:] = [temporary]
            nodes.append(replacement)
            rewritten += 1
        scope.ClearField("node")
        scope.node.extend(nodes)
    return {
        "rewritten_nan_predicates": rewritten,
        "unsupported_nan_predicates": unsupported,
    }
