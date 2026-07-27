#!/usr/bin/env python3
"""Negative conformance checks for the MoM v0alpha1 reference fixture."""

from __future__ import annotations

import copy

import verify


def must_reject(name: str, operation) -> None:
    try:
        operation()
    except verify.VerificationError:
        return
    raise AssertionError(f"negative fixture was accepted: {name}")


def main() -> int:
    schemas = verify.load_schemas()
    manifest = verify.load_json(verify.EXAMPLE / "manifest.json")
    ir, semantic_digest, bundle_digest = verify.check_manifest(manifest, schemas)
    binding = verify.load_yaml(verify.EXAMPLE / "deployment.binding.yaml")
    binding_digest, profile = verify.check_binding(
        binding,
        manifest,
        ir,
        semantic_digest,
        bundle_digest,
        schemas,
    )
    lock = verify.load_json(verify.EXAMPLE / "resolution.lock.json")

    cycle = copy.deepcopy(ir)
    next(
        operator
        for operator in cycle["operators"]
        if operator["id"] == "validate-answer"
    )["onFailure"] = "validate-answer"
    must_reject("cyclic nonterminal path", lambda: verify.check_graph(cycle))

    missing_entry = copy.deepcopy(ir)
    missing_entry["entrypoints"][0]["graphEntry"] = "missing-node"
    must_reject(
        "missing entrypoint graph target",
        lambda: verify.check_graph(missing_entry),
    )

    duplicate_model = copy.deepcopy(ir)
    duplicate_model["logicalModels"].append(
        copy.deepcopy(duplicate_model["logicalModels"][0])
    )
    must_reject(
        "duplicate logical-model identifier",
        lambda: verify.check_graph(duplicate_model),
    )

    bad_projection = copy.deepcopy(ir)
    bad_projection["projections"][0]["inputs"] = ["missing-signal"]
    must_reject(
        "unknown projection input",
        lambda: verify.check_graph(bad_projection),
    )

    partial_lock = copy.deepcopy(lock)
    del partial_lock["targets"]["local-generalist"]
    must_reject(
        "incomplete lock target set",
        lambda: verify.check_lock(
            partial_lock,
            binding,
            binding_digest,
            manifest,
            profile,
            schemas,
        ),
    )

    missing_driver = copy.deepcopy(lock)
    del missing_driver["targets"]["local-generalist"]["candidates"][0][
        "deploymentDriver"
    ]
    must_reject(
        "lock candidate without deployment driver",
        lambda: verify.check_lock(
            missing_driver,
            binding,
            binding_digest,
            manifest,
            profile,
            schemas,
        ),
    )

    bad_runtime = copy.deepcopy(lock)
    bad_runtime["targets"]["local-generalist"]["candidates"][0]["runtime"][
        "version"
    ] = "99.0.0"
    must_reject(
        "runtime outside binding range",
        lambda: verify.check_lock(
            bad_runtime,
            binding,
            binding_digest,
            manifest,
            profile,
            schemas,
        ),
    )

    bad_image = copy.deepcopy(lock)
    bad_image["targets"]["local-generalist"]["candidates"][0]["runtime"][
        "imageDigest"
    ] = "sha256:" + "e" * 64
    must_reject(
        "runtime image mismatch",
        lambda: verify.check_lock(
            bad_image,
            binding,
            binding_digest,
            manifest,
            profile,
            schemas,
        ),
    )

    bad_opset = copy.deepcopy(lock)
    bad_opset["controlEngine"]["supportedOpsets"] = ["mom.core@999.0.0"]
    must_reject(
        "opset outside manifest range",
        lambda: verify.check_lock(
            bad_opset,
            binding,
            binding_digest,
            manifest,
            profile,
            schemas,
        ),
    )

    variant = ir["behaviorVariants"][0]
    if verify.preference_is_valid(
        {"qualityBias": 0.1, "costBias": 0.1, "latencyBias": 0.1},
        variant,
    ):
        raise AssertionError("out-of-domain request preference was accepted")

    print("MoM negative conformance checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
