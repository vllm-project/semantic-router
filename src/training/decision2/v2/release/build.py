"""Build a Decision 2.0 Hugging Face package from a scored checkpoint and a release spec.

The scored model bytes are copied unchanged to their checkpoint-relative
paths (under the profile prefix). The builder verifies the checkpoint identity
against the spec before copying, counts parameters from safetensors headers,
checks the model name against the loaded size tier, vendors the exact runtime
sources, renders the product card from same-panel reports, writes the root
``config.json`` pointer and ``MODEL_MANIFEST.json``, screens every text file,
and only then renames the staged directory into place. Stdlib only; run from
an exact mirror. Nothing is uploaded here.

    python3 -m v2.release.build --spec SPEC.json --output WORK/package/<repo-name>
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from v2.release import automap
from v2.release import card as card_module
from v2.release import layout
from v2.release import licence as licence_policy
from v2.release.gate import no_own_1_0

SPEC_SCHEMA = "dev2-release-spec/1"
SOURCE_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = Path(__file__).resolve().parent
RUNTIME_TEMPLATE = RELEASE_DIR / "runtime"
KAI_RUNTIME = {
    "decision_runtime/__init__.py": "b0f9db94cfbcc73cab2f09e0269b8bd2eb87c0cc533b34ea5c9a8bfbb1ee48ca",
    "decision_runtime/_compat.py": "a0dd42a4e3b20eabbaf0d0d62ffc290ef4d9d20bc2fb2ba39d980771a8d0744c",
    "decision_runtime/native.py": "21c674a0d8a1406156504390974505ba5f981e7ecc9f0c3b29985876ef78511c",
    "decision_runtime/training.py": "166ea54629d5c90e0e0972b82c0b96e008c799f384400886ffd64312ab972177",
    "decision_inference/__init__.py": "8199829090f1a3ca5fe59aec1a6dce2beb57026c301d2c87d8c5b1c960981e66",
    "decision_inference/_auto.py": "3602e3a06386daab5494305413a08adb581f23007ffb5ca6e5df62fa23d11bd6",
    "decision_inference/_grouped.py": "80551e1bebd10634abcbbfce1aecf6cafab93b7e178399c428dd97022c6f9d95",
    "decision_inference/_request.py": "85b8349cdf08a3550027606b3108778955f84d66c52fa11dd908ea50311e69e5",
    "decision_inference/_system_one.py": "6efabe0b64cd30f4d6d24bc9d9a9da9a65251357d497134302ff0b9ee3834e02",
    "decision_inference/profile.py": "7e732fb3a9be93922a2e7e94920d9c75be7a3d07fbc3d88b1c2056bd374008f7",
}
KAI_ARCHITECTURE = "vela_decision_score_path_capacity_v1"
# The Qwen runtime's exact GPU fast path; runtime trees from before it lack these files.
FAST_RUNTIME = ("fast.py", "fast_kernels.py")
QWEN_MODULES = (
    "calibration.py",
    "data.py",
    "decision_model.py",
    "infer.py",
    "lora.py",
    "source.py",
)
# Imported by newer infer.py; mirrors of earlier scored runs lack it.
OPTIONAL_QWEN_MODULES = ("score_bias.py",)
HEAD_MODULES = {
    "type-separated": "type_separated_head.py",
    "candidate-interaction": "candidate_interaction_head.py",
    "score-cardinality": "score_cardinality_head.py",
}
IMPORT_REWRITE = (
    re.compile(r"^from training\.model\.(\w+) import", re.M),
    r"from .\1 import",
)
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".txt",
    ".jinja",
    ".svg",
    ".yaml",
    ".yml",
    ".html",
    "",
}
VOCAB_FILES = {
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
}
SECRET = re.compile(
    r"hf_[A-Za-z0-9]{30,}|jv_live_[A-Za-z0-9]|gho_[A-Za-z0-9]{20,}|apikey_[A-Za-z0-9]{8,}"
)
PRIVATE = re.compile(r"/data/dev2|/data/decision20|/root/|/home/[a-z]|\bnode [AB]\b")
IPV4 = re.compile(
    r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def name_base_model(spec: dict[str, Any]) -> str:
    """The base model whose size names a ``name_basis: base`` release; it must be in the lineage."""
    pinned = (spec.get("base") or {}).get("repo_id")
    named = spec.get("name_base_model") or pinned
    if not named:
        raise ValueError("name_basis base needs name_base_model (or a pinned base)")
    if pinned and named != pinned:
        raise ValueError(
            f"name_base_model {named} differs from the pinned base {pinned}"
        )
    lineage = {
        spec["origin"]["repo_id"],
        *(c["source"].split("@", 1)[0] for c in spec["licence"]["components"]),
        *([pinned] if pinned else []),
    }
    if named not in lineage:
        raise ValueError(
            f"name_base_model {named} is not in the declared weight lineage"
        )
    return named


def load_spec(path: Path) -> dict[str, Any]:
    spec = _object(path)
    required = {
        "schema",
        "kind",
        "repo_id",
        "model_name",
        "profile",
        "checkpoint",
        "expected_identity",
        "max_input_tokens",
        "origin",
        "licence",
        "card",
    }
    missing = required - set(spec)
    if spec.get("schema") != SPEC_SCHEMA or missing:
        raise ValueError(
            f"Release spec needs {sorted(missing)} and schema {SPEC_SCHEMA}"
        )
    if spec["kind"] not in ("staging", "release"):
        raise ValueError("kind is staging or release")
    if spec["profile"] not in layout.PROFILES or spec["profile"] == "encoder-marker":
        raise ValueError(f"Unsupported build profile: {spec['profile']}")
    layout.check_repo(
        spec["repo_id"], spec["model_name"], staging=spec["kind"] == "staging"
    )
    if spec.get("name_basis", "tier") == "base":
        name_base_model(spec)
    if spec.get("gate_profile") is not None:
        from v2.release.gate import gate_profile

        gate_profile(spec)
    if spec["kind"] == "release":
        from v2.release.gate import check

        if not spec.get("gate_receipt"):
            raise ValueError("A release build needs the coordinator's release decision")
        check(spec, Path(spec["gate_receipt"]))
    origin = spec["origin"]
    if not layout.REVISION.fullmatch(
        origin.get("revision", "")
    ) or "/" not in origin.get("repo_id", ""):
        raise ValueError("origin needs repo_id and a 40-hex revision")
    if type(spec["max_input_tokens"]) is not int or spec["max_input_tokens"] < 1:
        raise ValueError("max_input_tokens must be a positive integer")
    return spec


def _pinned_runtime_hashes(compat: Path) -> dict[str, str]:
    """Parse ``RUNTIME_SHA256`` from the vendored Kai runtime without executing it."""
    tree = ast.parse(compat.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "RUNTIME_SHA256" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Kai runtime has no pinned native runtime hashes")


def verify_kai_native(
    spec: dict[str, Any], checkpoint: Path, bundle: Path
) -> dict[str, Any]:
    manifest_path = checkpoint / "MANIFEST.json"
    expected = spec["expected_identity"].get("native_manifest_sha256")
    if layout.sha_file(manifest_path) != expected:
        raise ValueError("Native manifest differs from the scored export")
    manifest = _object(manifest_path)
    files = manifest.get("files")
    if manifest.get("schema") != "decision.files.v1" or not isinstance(files, dict):
        raise ValueError("Unexpected native manifest schema")
    for name, ref in files.items():
        path = checkpoint / name
        if (
            not layout.safe_relative(name)
            or set(ref) != {"bytes", "sha256"}
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != ref["bytes"]
            or layout.sha_file(path) != ref["sha256"]
        ):
            raise ValueError(f"Native file differs from its manifest: {name}")
    present = set(layout.inventory(checkpoint))
    if present != set(files) | {"MANIFEST.json"}:
        raise ValueError("Native export has files outside its manifest")
    for name, digest in KAI_RUNTIME.items():
        if layout.sha_file(bundle / name) != digest:
            raise ValueError(f"Kai runtime differs from the pinned revision: {name}")
    pinned = _pinned_runtime_hashes(bundle / "decision_runtime/_compat.py")
    runtime = {n: r["sha256"] for n, r in files.items() if n.endswith(".py")}
    if runtime != pinned:
        raise ValueError(
            "Native runtime sources differ from the pinned runtime revision"
        )
    config = _object(checkpoint / "decision_config.json")
    packing = config.get("packing") or {}
    if (
        config.get("architecture") != KAI_ARCHITECTURE
        or (config.get("arm"), config.get("training_arm")) != ("all22", "S22")
        or packing.get("max_length") != spec["max_input_tokens"]
        or packing.get("state_truncation") != "error"
    ):
        raise ValueError(
            "Native export is not a complete three-path Kai-family model at this cap"
        )
    return {
        "native_manifest_sha256": expected,
        "declared_parameters": config.get("parameters"),
        "transformers_version": config.get("transformers_version"),
        "calibration": config.get("calibration"),
    }


def _dec_fingerprint(identity: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Stdlib twin of ``v2.dec.dec_model.dec_fingerprint`` (adds the residual file)."""
    from training.model.data import canonical, file_sha256

    import hashlib

    residual = checkpoint / "dec_residual.safetensors"
    if not residual.is_file():
        return identity
    files = dict(identity["files_sha256"])
    prefix = "checkpoint/" if any(k.startswith("checkpoint/") for k in files) else ""
    files[f"{prefix}dec_residual.safetensors"] = file_sha256(residual)
    return {
        "model_sha256": hashlib.sha256(canonical(files).encode("utf-8")).hexdigest(),
        "files_sha256": files,
    }


def verify_score_bias(spec: dict[str, Any], model_sha256: str) -> dict[str, Any] | None:
    """Spec Score offsets, bound to their pinned bytes and the model hash."""
    entry = spec.get("score_bias")
    if not entry:
        return None
    from training.model.score_bias import load_score_bias

    path = Path(entry["path"])
    if layout.sha_file(path) != entry["sha256"]:
        raise ValueError("Score offsets file differs from its pinned sha256")
    offsets, _ = load_score_bias(path, model_sha256)
    return {
        "sha256": entry["sha256"],
        "offsets": {str(key): value for key, value in sorted(offsets.items())},
    }


BF16_COPY_SCHEMA = "dev2-release-bf16-copy/1"


def verify_bf16_copy(
    spec: dict[str, Any], checkpoint: Path, files: dict[str, str] | None = None
) -> dict[str, Any] | None:
    """A ``bf16_copy`` spec entry: the checkpoint is exactly that copy of the scored weights.

    Returns the scored (source) identity the copy was made from, so Score offsets
    can still be bound to the scored run by value.
    """
    entry = spec.get("bf16_copy")
    if not entry:
        return None
    path = Path(entry["receipt"])
    if layout.sha_file(path) != entry["sha256"]:
        raise ValueError("BF16 copy receipt differs from its pinned sha256")
    receipt = _object(path)
    if receipt.get("schema") != BF16_COPY_SCHEMA:
        raise ValueError("Not a v2.release.bf16_copy receipt")
    if receipt.get("model_sha256") != spec["expected_identity"].get("model_sha256"):
        raise ValueError("BF16 copy receipt names another package identity")
    if files is None:
        files = {
            p.relative_to(checkpoint).as_posix(): layout.sha_file(p)
            for p in checkpoint.rglob("*")
            if p.is_file()
        }
    recorded = receipt.get("files") or {}
    if set(recorded) != set(files) or any(
        files[name] != value["sha256"] for name, value in recorded.items()
    ):
        raise ValueError("Checkpoint is not the BF16 copy its receipt describes")
    if (
        spec.get("score_bias")
        and (receipt.get("score_bias") or {}).get("sha256")
        != spec["score_bias"]["sha256"]
    ):
        raise ValueError("BF16 copy receipt names other Score offsets")
    return {
        "receipt_sha256": entry["sha256"],
        "source_model_sha256": receipt["source_model_sha256"],
    }


BF16Z_SCHEMA = "dev2-release-bf16z/1"
FINGERPRINT_SUFFIXES = {".json", ".safetensors", ".bin", ".model", ".txt"}


def bf16z_files(
    spec: dict[str, Any], checkpoint: Path
) -> tuple[dict[str, str], dict[str, Any]] | None:
    """A ``bf16z`` spec entry: restored-name -> SHA-256 of every checkpoint file, and the storage record.

    Each .bf16z file must equal its compression receipt entry and name, in its own
    index, the restored bytes the receipt records; a pinned ``verify`` receipt
    (``bf16z verify``, which decompressed every file) must agree with both.
    """
    entry = spec.get("bf16z")
    if not entry:
        return None
    from v2.release.runtime import bf16z as codec

    path, verify = Path(entry["receipt"]), Path(entry["verify"])
    if layout.sha_file(path) != entry["sha256"]:
        raise ValueError("bf16z receipt differs from its pinned sha256")
    if layout.sha_file(verify) != entry["verify_sha256"]:
        raise ValueError("bf16z verify receipt differs from its pinned sha256")
    receipt, report = _object(path), _object(verify)
    if receipt.get("schema") != BF16Z_SCHEMA or receipt.get("format") != codec.FORMAT:
        raise ValueError("Not a bf16z compression receipt")
    recorded = receipt["files"]
    if not report.get("passed") or set(report.get("files", {})) != set(recorded):
        raise ValueError("bf16z verify receipt does not cover every compressed file")
    for name, value in recorded.items():
        checked = report["files"][name]
        if (
            checked["sha256"] != value["sha256"]
            or checked["restored_sha256"] != value["restored_sha256"]
        ):
            raise ValueError(f"bf16z verify receipt disagrees on {name}")
    restored: dict[str, str] = {}
    compressed = set()
    for file in sorted(checkpoint.rglob("*")):
        if not file.is_file():
            continue
        name = file.relative_to(checkpoint).as_posix()
        if not name.endswith(codec.SUFFIX):
            restored[name] = layout.sha_file(file)
            continue
        value = recorded.get(name)
        if value is None or layout.sha_file(file) != value["sha256"]:
            raise ValueError(f"{name} differs from the bf16z receipt")
        index, _, _ = codec.read_index(file)
        if (
            index["source"]["sha256"] != value["restored_sha256"]
            or index["source"]["bytes"] != value["restored_bytes"]
        ):
            raise ValueError(f"{name} restores other bytes than its receipt records")
        restored[codec.restored_name(name)] = value["restored_sha256"]
        compressed.add(name)
    if compressed != set(recorded):
        raise ValueError("Checkpoint and bf16z receipt list different compressed files")
    storage = {
        "codec": codec.FORMAT,
        "receipt_sha256": entry["sha256"],
        "verify_sha256": entry["verify_sha256"],
        "level": receipt["level"],
        "zstandard": receipt["zstandard"],
        "files": {
            name: {
                key: value[key]
                for key in (
                    "sha256",
                    "bytes",
                    "restored",
                    "restored_sha256",
                    "restored_bytes",
                )
            }
            for name, value in sorted(recorded.items())
        },
    }
    return restored, storage


def identity_from_hashes(files: dict[str, str]) -> dict[str, Any]:
    """checkpoint_fingerprint of a full checkpoint, from its files' SHA-256 (no weights read)."""
    import hashlib

    from training.model.data import canonical
    from training.model.infer import MODEL_ROOT_FILES

    hashes = {
        name: digest
        for name, digest in sorted(files.items())
        if ("/" not in name and name in MODEL_ROOT_FILES)
        or (
            name.split("/", 1)[0] == "backbone"
            and Path(name).suffix in FINGERPRINT_SUFFIXES
        )
    }
    if (
        "decision_config.json" not in hashes
        or "decision_head.safetensors" not in hashes
    ):
        raise ValueError(
            "Checkpoint needs decision_config.json and decision_head.safetensors"
        )
    return {
        "model_sha256": hashlib.sha256(canonical(hashes).encode("utf-8")).hexdigest(),
        "files_sha256": hashes,
    }


def verify_qwen(spec: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    from training.model.calibration import load_calibration
    from training.model.infer import checkpoint_fingerprint
    from training.model.lora import LORA_FORMAT

    metadata = _object(checkpoint / "decision_config.json")
    adapter = spec["profile"] == "qwen-adapter"
    if (metadata.get("checkpoint_format") == LORA_FORMAT) != adapter:
        raise ValueError("Checkpoint format and package profile disagree")
    base_path = Path(spec["base"]["path"]) if adapter else None
    stored = bf16z_files(spec, checkpoint)
    if stored is not None and (adapter or metadata.get("dec_residual") is not None):
        raise ValueError("bf16z storage is defined for full non-residual checkpoints")
    label = metadata.get("readout") == "label_token"
    if label and stored is not None:
        raise ValueError("bf16z storage is not defined for label-token checkpoints")
    if label:
        from v2.dec.label_token import label_fingerprint

        identity = label_fingerprint(checkpoint, base_path)
    elif stored is not None:
        identity = identity_from_hashes(stored[0])
    else:
        identity = checkpoint_fingerprint(checkpoint, base_path)
    if metadata.get("dec_residual") is not None:
        identity = _dec_fingerprint(identity, checkpoint)
    if identity["model_sha256"] != spec["expected_identity"].get("model_sha256"):
        raise ValueError(
            "Checkpoint fingerprint differs from the scored model identity"
        )
    calibration = spec.get("calibration")
    temperatures = None
    if calibration:
        path = Path(calibration["path"])
        if layout.sha_file(path) != calibration["sha256"]:
            raise ValueError("Calibration file differs from the scored calibration")
        temperatures, report = load_calibration(path, identity["model_sha256"])
        if report.get("inference", {}).get("max_length") not in (
            None,
            spec["max_input_tokens"],
        ):
            raise ValueError("Calibration context differs from the package limit")
    score_bias = verify_score_bias(spec, identity["model_sha256"])
    result = {
        "model_sha256": identity["model_sha256"],
        "fingerprint_files": identity["files_sha256"],
        "architecture": metadata.get("architecture"),
        "head_variant": metadata.get("head_variant", "shared"),
        "dec_residual": metadata.get("dec_residual") is not None,
        "readout": metadata.get("readout", "head"),
        "text_parameter_count": metadata.get("text_parameter_count"),
        "temperature_by_type": temperatures,
    }
    if score_bias is not None:
        result["score_bias"] = score_bias
    bf16 = verify_bf16_copy(spec, checkpoint, stored[0] if stored else None)
    if bf16 is not None:
        result["bf16_copy"] = bf16
    if stored is not None:
        result["storage"] = stored[1]
    if adapter:
        contract = metadata["lora"]
        base = spec["base"]
        if not layout.REVISION.fullmatch(base.get("revision", "")):
            raise ValueError("The external base needs a 40-hex revision")
        result["base"] = {
            "repo_id": base["repo_id"],
            "revision": base["revision"],
            "source_kind": contract.get("source_kind"),
            "files_sha256": contract["source_fingerprint"]["files_sha256"],
            "licence": base.get("licence"),
            "redistribution": base.get("redistribution"),
        }
    return result


def base_text_parameters(base: Path, files: dict[str, str], source_kind: str) -> int:
    """Text-backbone tensors the decision model instantiates from the base."""
    import struct

    total = 0
    for name in files:
        if not name.endswith(".safetensors"):
            continue
        with (base / name).open("rb") as stream:
            (size,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(size))
        for tensor, meta in header.items():
            if tensor == "__metadata__":
                continue
            if source_kind in ("base", "posttrained"):
                keep = tensor.startswith(
                    (
                        "model.language_model.",
                        "model.layers.",
                        "model.embed_tokens.",
                        "model.norm.",
                    )
                )
            else:
                keep = name.startswith("backbone/")
            if keep:
                count = 1
                for dim in meta["shape"]:
                    count *= dim
                total += count
    return total


def _mirror_tree(value: str, key: str, inside: str) -> tuple[Path, dict[str, Any]]:
    root = Path(value).resolve(strict=True)
    mirror = next(
        (
            p / ".dev2-mirror.json"
            for p in (root, *root.parents)
            if (p / ".dev2-mirror.json").is_file()
        ),
        None,
    )
    if mirror is None or not (root / inside).is_dir():
        raise ValueError(
            f"{key} must be a decision2 tree with {inside} inside an exact node mirror"
        )
    record = json.loads(mirror.read_text(encoding="utf-8"))
    return root, {
        "commit": record.get("commit"),
        "tree": record.get("tree"),
        "content_manifest_sha256": record.get("content_manifest_sha256"),
    }


def vendor_root(spec: dict[str, Any]) -> tuple[Path, dict[str, Any] | None]:
    """Tree whose training/model (and v2/dec) sources are vendored.

    Default: this builder's tree. ``vendor_source`` names the decision2 tree of an
    exact node mirror (e.g. the scored run's own source mirror) so a later change
    to a shared inference module does not block a release of an earlier scored run.
    """
    value = spec.get("vendor_source")
    if not value:
        return SOURCE_ROOT, None
    return _mirror_tree(value, "vendor_source", "training/model")


def runtime_root(spec: dict[str, Any]) -> tuple[Path, dict[str, Any] | None]:
    """Tree whose v2/release/runtime supplies the package runtime (decision2/*.py).

    Default: this builder's tree. ``runtime_source`` names the decision2 tree of an
    exact node mirror (e.g. the one that built the revision a card-only revision
    replaces) so the package runtime stays byte-identical to that revision's.
    """
    value = spec.get("runtime_source")
    if not value:
        return SOURCE_ROOT, None
    return _mirror_tree(value, "runtime_source", "v2/release/runtime")


def automap_root(spec: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    """Tree whose v2/release/automap supplies the root 🤗 Transformers remote code.

    Default: this builder's tree. ``automap_source`` names the decision2 tree of an exact node
    mirror, so a later card-only revision can keep the remote code of the revision it replaces.
    """
    value = spec.get("automap_source")
    if not value:
        return None, None
    return _mirror_tree(value, "automap_source", "v2/release/automap")


def remote_code(spec: dict[str, Any], stage: Path) -> dict[str, Any]:
    """Copy the trust_remote_code modules to the package root; the manifest section that records them."""
    tree, mirror = automap_root(spec)
    records = automap.copy_into(stage, tree)
    return {
        "files": records,
        **automap.CONFIG_FIELDS,
        **({"automap_source": mirror} if mirror else {}),
        **(
            {"tested": spec["remote_code"]["tested"]}
            if (spec.get("remote_code") or {}).get("tested")
            else {}
        ),
    }


def vendor_runtime(
    spec: dict[str, Any], stage: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    """Copy the runtime template and the exact scored inference sources."""
    root, _ = vendor_root(spec)
    runtime_tree, _ = runtime_root(spec)
    records: dict[str, Any] = {}
    target = stage / layout.RUNTIME_DIR
    target.mkdir()
    profile_module = "kai_native.py" if spec["profile"] == "kai-native" else "qwen.py"
    runtime_files = ["__init__.py", "api.py", profile_module]
    if profile_module == "qwen.py":
        runtime_files += [
            name
            for name in FAST_RUNTIME
            if (runtime_tree / "v2/release/runtime" / name).is_file()
        ]
    if spec.get("bf16z"):
        runtime_files.append("bf16z.py")
    for name in runtime_files:
        shutil.copyfile(runtime_tree / "v2/release/runtime" / name, target / name)
        records[f"decision2/{name}"] = {
            "source": f"v2/release/runtime/{name}",
            "rewritten": False,
        }
    vendor = stage / layout.VENDOR_DIR
    vendor.mkdir()
    (vendor / "__init__.py").write_text(
        '"""Vendored exact inference sources."""\n', encoding="utf-8"
    )
    records["decision2/_vendor/__init__.py"] = {"source": None, "rewritten": False}
    if spec["profile"] == "kai-native":
        bundle = Path(spec["runtime_bundle"])
        for name in KAI_RUNTIME:
            (vendor / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle / name, vendor / name)
            records[f"decision2/_vendor/{name}"] = {
                "source": f"{spec['runtime_revision']['repo_id']}@{spec['runtime_revision']['revision']}:{name}",
                "rewritten": False,
            }
        return records
    package = vendor / "dev2model"
    package.mkdir()
    (package / "__init__.py").write_text(
        '"""Exact Decision 2.0 model sources."""\n', encoding="utf-8"
    )
    records["decision2/_vendor/dev2model/__init__.py"] = {
        "source": None,
        "rewritten": False,
    }
    modules = list(QWEN_MODULES)
    modules += [
        name
        for name in OPTIONAL_QWEN_MODULES
        if (root / "training/model" / name).is_file()
    ]
    if spec.get("score_bias") and "score_bias.py" not in modules:
        raise ValueError("Score offsets need score_bias.py in the vendored sources")
    if identity["head_variant"] in HEAD_MODULES:
        modules.append(HEAD_MODULES[identity["head_variant"]])
    for name in modules:
        shutil.copyfile(root / "training/model" / name, package / name)
        records[f"decision2/_vendor/dev2model/{name}"] = {
            "source": f"training/model/{name}",
            "rewritten": False,
        }
    if identity["dec_residual"]:
        original = (root / "v2/dec/dec_model.py").read_text(encoding="utf-8")
        pattern, replacement = IMPORT_REWRITE
        (package / "dec_model.py").write_text(
            pattern.sub(replacement, original), encoding="utf-8"
        )
        records["decision2/_vendor/dev2model/dec_model.py"] = {
            "source": "v2/dec/dec_model.py",
            "source_sha256": layout.sha_file(root / "v2/dec/dec_model.py"),
            "rewritten": "absolute training.model imports -> package-relative",
        }
    if identity.get("readout") == "label_token":
        original = (root / "v2/dec/label_token.py").read_text(encoding="utf-8")
        pattern, replacement = IMPORT_REWRITE
        rewritten = pattern.sub(replacement, original)
        if "training.model" in rewritten:
            raise ValueError("label_token.py keeps an absolute training.model import")
        (package / "label_token.py").write_text(rewritten, encoding="utf-8")
        records["decision2/_vendor/dev2model/label_token.py"] = {
            "source": "v2/dec/label_token.py",
            "source_sha256": layout.sha_file(root / "v2/dec/label_token.py"),
            "rewritten": "absolute training.model imports -> package-relative",
        }
    for name, record in records.items():
        source = record.get("source")
        if source and "@" not in source and "source_sha256" not in record:
            tree = runtime_tree if source.startswith("v2/release/") else root
            record["source_sha256"] = layout.sha_file(tree / source)
    return records


def check_scored_score_bias(
    spec: dict[str, Any], native: dict[str, Any], score_bias: dict[str, Any] | None
) -> str | None:
    """Packaged offsets must equal, by value, the offsets the scored run applied.

    The packaged file may differ in bytes (e.g. public-only ``fit`` provenance);
    answers depend only on the model hash and the offsets. Returns the scored
    file's SHA-256.
    """
    scored_entry = native.get("score_bias")
    scored_sha = native.get("score_bias_sha256")
    if not spec.get("score_bias"):
        if scored_entry is not None or scored_sha is not None:
            raise ValueError("The scored run applied Score offsets; the spec has none")
        return None
    if score_bias is None:
        raise ValueError("Score offsets are not verified against the checkpoint")
    scored_identity = spec["expected_identity"].get("model_sha256")
    if spec.get("bf16_copy"):
        scored_identity = _object(Path(spec["bf16_copy"]["receipt"]))[
            "source_model_sha256"
        ]
    if (
        not isinstance(scored_sha, str)
        or not isinstance(scored_entry, dict)
        or scored_entry.get("file_sha256") != scored_sha
        or scored_entry.get("offsets") != score_bias["offsets"]
        or native.get("model_sha256") != scored_identity
    ):
        raise ValueError("Score offsets differ from the ones the scored run applied")
    return scored_sha


def check_scored_runtime(
    spec: dict[str, Any],
    records: dict[str, Any],
    score_bias: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Vendored sources must equal the scored adapter sources recorded at scoring time.

    ``score_bias`` is the verified spec entry (``verify_score_bias``).
    """
    scored = spec.get("scored") or {}
    path = scored.get("native_manifest")
    if not path:
        if spec.get("score_bias"):
            raise ValueError("Score offsets need the scored native manifest")
        return None
    native = _object(Path(path))
    scored_score_bias = check_scored_score_bias(spec, native, score_bias)
    recorded = native.get("adapter_files_sha256") or {}
    if "score_bias.py" in recorded and not any(
        record.get("source") == "training/model/score_bias.py"
        for record in records.values()
    ):
        raise ValueError("The scored run used score_bias.py; it is not vendored")
    checked = {}
    for record in records.values():
        source = record.get("source") or ""
        for key in (source, source.removeprefix("training/model/")):
            if key in recorded:
                if recorded[key] != record["source_sha256"]:
                    raise ValueError(
                        f"Vendored {source} differs from the scored adapter source"
                    )
                checked[source] = recorded[key]
    result = {"native_manifest_sha256": layout.sha_file(Path(path)), "checked": checked}
    if scored_score_bias is not None:
        result["score_bias_sha256"] = scored_score_bias
    return result


def _pinned(item: Any, what: str, key: str = "path") -> Path:
    if not isinstance(item, dict) or not item.get(key):
        raise ValueError(f"card.{what} needs {key} and its SHA-256")
    path = Path(item[key])
    return path if path.is_absolute() else SOURCE_ROOT / path


def card_inputs(spec: dict[str, Any]) -> tuple[Path, Path]:
    """The private Index input and the rendered card assets, both pinned by SHA-256."""
    card = spec["card"]
    index = _pinned(card.get("index"), "index")
    if layout.sha_file(index) != card["index"].get("sha256"):
        raise ValueError("card.index differs from its pinned SHA-256")
    assets = _pinned(card.get("assets"), "assets", "dir")
    receipt = assets / card_module.ASSETS_RECEIPT
    if not receipt.is_file() or layout.sha_file(receipt) != card["assets"].get(
        "receipt_sha256"
    ):
        raise ValueError("card.assets receipt differs from its pinned SHA-256")
    return index, assets


def card_speed(spec: dict[str, Any]) -> dict[str, Any]:
    """Median single-question latency from a pinned bench receipt of this runtime."""
    item = spec["card"]["speed"]
    path = _pinned(item, "speed", "evidence")
    if layout.sha_file(path) != item.get("sha256"):
        raise ValueError("card.speed evidence differs from its pinned SHA-256")
    bench = json.loads(path.read_text(encoding="utf-8"))
    if bench.get("passed") is not True or not bench.get("count"):
        raise ValueError("card.speed evidence is not a passed bench receipt")
    return {"median_ms": bench["latency_ms"]["p50"], "requests": bench["count"]}


def write_licences(spec: dict[str, Any], stage: Path, decision: dict[str, Any]) -> None:
    lic = spec["licence"]
    for item in lic.get("files", []):
        source, target = Path(item["source"]), stage / item["path"]
        if layout.sha_file(source) != item["sha256"] or not layout.safe_relative(
            item["path"]
        ):
            raise ValueError(
                f"Licence file differs from its pinned bytes: {item['path']}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if not (stage / "LICENSE").is_file():
        raise ValueError("Every package carries a root LICENSE")
    upstream = lic.get("notice")
    if upstream:
        path = Path(upstream["source"])
        if layout.sha_file(path) != upstream["sha256"]:
            raise ValueError("Upstream NOTICE differs from its pinned bytes")
        shutil.copyfile(path, stage / "NOTICE")
    if decision["spdx"] == "other":
        rows = [
            "# Component licences",
            "",
            "The model-card licence is `other` because not every upstream component of the weight "
            "lineage is under an Apache-2.0-compatible licence. Each component keeps its own terms:",
            "",
            "| Component | Licence | Source |",
            "| --- | --- | --- |",
        ]
        rows += [
            f"| {c['component']} | {c['licence']} | {c.get('source', '—')} |"
            for c in decision["components"]
        ]
        rows += ["", *lic.get("licensing_notes", [])]
        (stage / "LICENSING.md").write_text(
            "\n".join(rows).rstrip("\n") + "\n", encoding="utf-8"
        )


def screen(stage: Path) -> dict[str, Any]:
    """Refuse credentials, private paths, machine identities and metadata in public files."""
    checked = 0
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(stage).as_posix()
        if path.suffix == ".safetensors":
            text = json.dumps(layout.safetensors_metadata(path))
            vocab = False
        elif path.name.endswith(layout.BF16Z_SUFFIX):
            from v2.release.runtime.bf16z import original_header, read_index

            text = json.dumps([read_index(path)[0], original_header(path)])
            vocab = False
        elif path.suffix == ".png":
            data = path.read_bytes()
            if not data.startswith(b"\x89PNG\r\n\x1a\n") or any(
                t in data for t in (b"tEXt", b"iTXt", b"zTXt", b"eXIf")
            ):
                raise ValueError(f"PNG must carry no text or EXIF chunks: {relative}")
            continue
        elif path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            vocab = path.name in VOCAB_FILES
        else:
            raise ValueError(f"Unexpected file type in a public package: {relative}")
        checked += 1
        if SECRET.search(text):
            raise ValueError(f"Credential-like text in {relative}")
        if not vocab and (PRIVATE.search(text) or IPV4.search(text)):
            raise ValueError(f"Private path, machine identity or address in {relative}")
    return {"text_files_screened": checked}


def _components_text(packaged: dict[str, int], external: int | None) -> str:
    names = {
        "native": "native encoder paths and heads",
        "backbone": "backbone",
        "head": "decision head",
        "residual": "residual readout",
        "adapter": "LoRA adapter",
    }
    parts = [f"{names.get(k, k)} {v:,}" for k, v in packaged.items() if v]
    if external:
        parts.insert(
            0, f"pinned base text backbone {external:,}, not redistributed here"
        )
    return "; ".join(parts)


def build(spec_path: Path, output: Path) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    spec = load_spec(spec_path)
    decision_record = None
    if spec["kind"] == "release":
        decision_path = Path(spec["gate_receipt"])
        decision_record = {
            "sha256": layout.sha_file(decision_path),
            "status": _object(decision_path).get("status", "final"),
        }
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if output.name != spec["repo_id"].rsplit("/", 1)[1]:
        raise ValueError(
            "Name the package directory after the repository (the card example loads it by that name)"
        )
    profile = spec["profile"]
    checkpoint = Path(spec["checkpoint"]).resolve(strict=True)
    if profile == "kai-native":
        if spec.get("score_bias"):
            raise ValueError("Score offsets are a Qwen-runtime feature")
        identity = verify_kai_native(
            spec, checkpoint, Path(spec["runtime_bundle"]).resolve(strict=True)
        )
    else:
        identity = verify_qwen(spec, checkpoint)
    names = layout.select_model_files(profile, checkpoint)
    for name in names:
        if name.split("/")[-1] in layout.TRAINING_ONLY:
            raise ValueError(f"Training state cannot enter a package: {name}")
    files = [layout.package_path(profile, name) for name in names]
    groups = layout.parameter_files(profile, files)
    packaged = {
        group: sum(
            layout.safetensors_count(
                checkpoint / name[len(layout.PROFILES[profile].prefix) :]
            )
            for name in paths
        )
        for group, paths in groups.items()
    }
    external = None
    if profile == "qwen-adapter":
        base = identity["base"]
        external = base_text_parameters(
            Path(spec["base"]["path"]), base["files_sha256"], base["source_kind"]
        )
    loaded = sum(packaged.values()) + (external or 0)
    if profile == "kai-native" and identity["declared_parameters"] not in (
        None,
        loaded,
    ):
        raise ValueError(
            "Native header count differs from the export's declared parameters"
        )
    if profile == "qwen-full" and identity["text_parameter_count"] not in (
        None,
        packaged["backbone"],
    ):
        raise ValueError(
            "Backbone header count differs from the checkpoint's text parameter count"
        )
    name_basis = spec.get("name_basis", "tier")
    base_model = name_base_model(spec) if name_basis == "base" else None
    named = layout.release_name(loaded, name_basis, base_model)
    if named != spec["model_name"]:
        raise ValueError(
            f"{loaded:,} loaded parameters name the model "
            f"{named} ({name_basis}), not {spec['model_name']}"
        )
    decision = licence_policy.package_licence(spec["licence"]["components"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".dev2-release-", dir=output.parent
    ) as temporary:
        work = Path(temporary)
        stage = work / output.name
        stage.mkdir(mode=0o755)
        for name, path in zip(names, files):
            target = stage / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint / name, target)
        calibration = None
        if spec.get("calibration"):
            shutil.copyfile(spec["calibration"]["path"], stage / "calibration.json")
            calibration = {
                "file": "calibration.json",
                "sha256": layout.sha_file(stage / "calibration.json"),
                "temperature_by_type": identity.get("temperature_by_type"),
            }
        score_bias = None
        if identity.get("score_bias"):
            shutil.copyfile(spec["score_bias"]["path"], stage / "score_bias.json")
            staged = layout.sha_file(stage / "score_bias.json")
            if staged != identity["score_bias"]["sha256"]:
                raise ValueError("Score offsets file changed while staging")
            score_bias = {"file": "score_bias.json", **identity["score_bias"]}
        runtime = vendor_runtime(spec, stage, identity)
        for name, record in runtime.items():
            record["sha256"] = layout.sha_file(stage / name)
        remote = remote_code(spec, stage)
        scored_runtime = check_scored_runtime(spec, runtime, identity.get("score_bias"))
        if score_bias:
            score_bias["scored_sha256"] = scored_runtime["score_bias_sha256"]
        write_licences(spec, stage, decision)
        facts = {
            "model_name": spec["model_name"],
            "repo_id": spec["repo_id"],
            "profile": profile,
            "parameters": {
                "loaded": loaded,
                "components_text": _components_text(packaged, external),
            },
            "max_input_tokens": spec["max_input_tokens"],
            "origin": spec["origin"],
            "licence": decision,
            "model_sha256": identity["model_sha256"],
            **({"speed": card_speed(spec)} if spec["card"].get("speed") else {}),
            "runtime_requirements": spec.get("runtime_requirements", {}),
            **(
                {"name_basis": name_basis, "name_base_model": base_model}
                if base_model
                else {}
            ),
            "calibration_text": spec["card"].get("calibration_text")
            or (
                "per-type temperatures fitted on the frozen calibration partition (`calibration.json`)."
                if calibration
                else "raw native probabilities (no post-hoc temperature)."
            ),
            "requirements_text": spec["card"].get("requirements_text"),
            "remote_code": {
                "tested": (spec.get("remote_code") or {}).get("tested"),
                "base": identity.get("base"),
            },
            **(
                {
                    "comparison": (
                        "no-1.0"
                        if no_own_1_0(spec["gate_profile"].get("tier") or {})
                        else spec["gate_profile"]["name"]
                    )
                }
                if spec.get("gate_profile")
                else {}
            ),
        }
        roster = Path(spec["card"]["roster"])
        index, assets = card_inputs(spec)
        card = card_module.build_card(
            entries=spec["card"]["reports"],
            roster=roster if roster.is_absolute() else SOURCE_ROOT / roster,
            paired=Path(spec["card"]["paired"]) if spec["card"].get("paired") else None,
            facts=facts,
            text=spec["card"]["text"],
            work=work / "card-work",
            output=stage,
            index=index,
            assets=assets,
            paired_peers={
                key: Path(path)
                for key, path in (spec["card"].get("paired_peers") or {}).items()
            },
        )
        pointer = layout.pointer(
            profile,
            spec["model_name"],
            files,
            calibration=calibration and calibration["file"],
            base=identity.get("base"),
            max_input_tokens=spec["max_input_tokens"],
            remote_code=True,
        )
        (stage / layout.POINTER_NAME).write_text(
            json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        screened = screen(stage)
        inventory = layout.inventory(stage)
        mirror = next(
            (
                json.loads((p / ".dev2-mirror.json").read_text())
                for p in (SOURCE_ROOT, *SOURCE_ROOT.parents)
                if (p / ".dev2-mirror.json").is_file()
            ),
            None,
        )
        manifest = {
            "schema": layout.MANIFEST_SCHEMA,
            "package_schema": layout.PACKAGE_SCHEMA,
            "kind": spec["kind"],
            "repo_id": spec["repo_id"],
            "model_name": spec["model_name"],
            "tier": layout.tier_for(loaded),
            **({"name_basis": name_basis} if name_basis != "tier" else {}),
            **({"name_base_model": base_model} if base_model else {}),
            "profile": profile,
            "files_sha256": inventory,
            "model_files": files,
            "runtime_files": runtime,
            "remote_code": remote,
            "parameters": {
                "loaded": loaded,
                "packaged": packaged,
                "packaged_files": groups,
                "external_base_text": external,
                "source": "safetensors header element counts; the runtime asserts the loaded count",
            },
            "identity": {
                k: v
                for k, v in identity.items()
                if k in ("native_manifest_sha256", "model_sha256", "fingerprint_files")
            },
            "origin": spec["origin"],
            "base": identity.get("base"),
            "calibration": calibration,
            **({"score_bias": score_bias} if score_bias else {}),
            **({"storage": identity["storage"]} if identity.get("storage") else {}),
            "max_input_tokens": spec["max_input_tokens"],
            "runtime": {
                "requirements": spec.get("runtime_requirements", {}),
                "scored_runtime_check": scored_runtime,
                "equivalence": spec.get("runtime_equivalence"),
                **(
                    {"vendor_source": vendor_root(spec)[1]}
                    if spec.get("vendor_source")
                    else {}
                ),
                **(
                    {"runtime_source": runtime_root(spec)[1]}
                    if spec.get("runtime_source")
                    else {}
                ),
            },
            "scored": {
                key: value
                for key, value in (spec.get("scored") or {}).items()
                if key
                in (
                    "report_sha256",
                    "seal_sha256",
                    "predictions_sha256",
                    "paired_sha256",
                    "label",
                )
            }
            or None,
            "licence": {"spdx": decision["spdx"], "components": decision["components"]},
            "card": {
                "readme_sha256": card["readme_sha256"],
                "figures_sha256": card["figures_sha256"],
                "assets_receipt_sha256": card["assets_receipt_sha256"],
                "index_sha256": card["index_sha256"],
            },
            "builder": {
                "source_commit": (mirror or {}).get("commit"),
                "module_sha256": layout.sha_file(Path(__file__)),
            },
        }
        (stage / layout.MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # The manifest carries spec text (origin, scored label, licence sources) too.
        screened = screen(stage)
        if layout.inventory(stage) != {
            **inventory,
            layout.MANIFEST_NAME: layout.sha_file(stage / layout.MANIFEST_NAME),
        }:
            raise ValueError("Staged package changed while writing its manifest")
        stage.rename(output)
        receipt_dir = output.parent / f"{output.name}.build"
        receipt_dir.mkdir()
        for name in (card_module.ASSETS_RECEIPT,):
            source = work / "card-work" / name
            if source.is_file():
                shutil.copyfile(source, receipt_dir / name)
    receipt = {
        "schema": "dev2-release-build/1",
        "started_utc": started.isoformat(),
        "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "spec_sha256": layout.sha_file(spec_path),
        "package": str(output),
        "manifest_sha256": layout.sha_file(output / layout.MANIFEST_NAME),
        "files": len(inventory) + 1,
        "parameters": {
            "loaded": loaded,
            "packaged": packaged,
            "external_base_text": external,
        },
        "identity": {k: v for k, v in identity.items() if k != "fingerprint_files"},
        **({"score_bias": score_bias} if score_bias else {}),
        "card": card,
        "screen": screened,
        "licence": decision["spdx"],
        "release_decision": decision_record,
        "builder_commit": (mirror or {}).get("commit"),
    }
    layout.write_json(receipt_dir / "BUILD.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.spec, args.output)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ("manifest_sha256", "files", "parameters", "licence")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
