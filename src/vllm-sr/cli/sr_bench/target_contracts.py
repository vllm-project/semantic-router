"""Resolve frozen evaluation targets independently of their subject role."""


def target_inventory(manifest):
    inventory = {}
    for target in [
        *manifest["targets"],
        *manifest.get("auxiliary_targets", {}).values(),
    ]:
        identity = target["id"]
        if identity in inventory:
            raise ValueError("target IDs must be unique nonempty strings")
        inventory[identity] = target
    return inventory


def resolve_auxiliary_target(config, role, manifest):
    ref = config.get(role)
    if not isinstance(ref, str):
        raise ValueError(f"{role} must reference a frozen target ID")
    target = target_inventory(manifest).get(ref)
    if target is None or target.get("kind") != "single":
        raise ValueError(f"{role} requires a fixed single-model target")
    return target


def auxiliary_bindings(manifest):
    return {
        (benchmark, role): resolve_auxiliary_target(options, role, manifest)
        for benchmark, options in manifest.get("benchmark_options", {}).items()
        for role in ("judge", "simulator")
        if role in options
    }


def effective_auxiliary_targets(manifest):
    return {
        key: {
            **target,
            "request_params": {
                **manifest["sampling"],
                **target.get("request_params", {}),
            },
        }
        for key, target in auxiliary_bindings(manifest).items()
    }
