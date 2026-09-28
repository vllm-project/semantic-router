"""Metadata-only verification of the ~27B candidate starts.

Reads the Hugging Face API, small repository files and every safetensors
header (HTTP range requests), never full weight shards. Parameter counts are
exact tensor-element sums grouped by component. The token, if present, is read
from the default HF token file and is never printed or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HUB = "https://huggingface.co"
TOKEN_FILE = Path("/root/.cache/huggingface/token")
CANDIDATES = {
    "qwen38-27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "qwen35-27b": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"),
    "gemma4-26b-a4b": (
        "google/gemma-4-26B-A4B",
        "24548b62aa021d562695c04aaf7758a1ea47990b",
    ),
    "gemma4-26b-a4b-it": (
        "google/gemma-4-26B-A4B-it",
        "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    ),
}
ABSENT_CHECKS = ("Qwen/Qwen3.5-27B-Base", "Qwen/Qwen3.8-27B-Base")
SMALL_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "LICENSE",
    "README.md",
    "model.safetensors.index.json",
)
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "U8": 1,
    "I8": 1,
    "BOOL": 1,
}


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"User-Agent": "decision2-27b-verify/1"}
    if TOKEN_FILE.is_file():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            headers["Authorization"] = "Bearer " + token
    headers.update(extra or {})
    return headers


def _get(url: str, *, extra: dict[str, str] | None = None, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=_headers(extra))
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                raise
            if attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def model_info(repo: str, revision: str) -> dict[str, Any]:
    data = json.loads(_get(f"{HUB}/api/models/{repo}/revision/{revision}?blobs=true"))
    if data.get("sha") != revision:
        raise ValueError(
            f"{repo}: API revision {data.get('sha')} differs from pinned {revision}"
        )
    return data


def safetensors_header(repo: str, revision: str, filename: str) -> dict[str, Any]:
    url = f"{HUB}/{repo}/resolve/{revision}/{filename}"
    prefix = _get(url, extra={"Range": "bytes=0-7"})
    if len(prefix) != 8:
        raise ValueError(f"{filename}: short safetensors prefix")
    length = int.from_bytes(prefix, "little")
    if not 2 <= length <= 256 << 20:
        raise ValueError(f"{filename}: invalid header length")
    header = _get(url, extra={"Range": f"bytes=8-{7 + length}"})
    if len(header) != length:
        raise ValueError(f"{filename}: truncated header")
    parsed = json.loads(header)
    parsed.pop("__metadata__", None)
    return parsed


def component(name: str) -> str:
    """Map an official tensor name to an accounting component."""
    text_prefixes = (
        "model.language_model.",
        "language_model.model.",
        "model.text_model.",
    )
    if name.startswith(
        (
            "model.visual.",
            "model.vision_tower.",
            "model.embed_vision.",
            "visual.",
            "vision_tower.",
        )
    ):
        return "vision"
    if name.startswith(("model.audio_tower.", "model.embed_audio.", "audio_tower.")):
        return "audio"
    if name.startswith(("mtp.", "model.mtp.")) or ".mtp." in name:
        return "mtp"
    if name.startswith("lm_head."):
        return "lm_head"
    if name.startswith(text_prefixes):
        if "embed_tokens" in name:
            return "text_embeddings"
        if ".experts." in name or name.endswith(".experts") or ".experts_" in name:
            return "text_moe_experts"
        if ".router" in name:
            return "text_moe_router"
        return "text_other"
    return "other"


def count_parameters(headers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, int] = {}
    dtypes: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    stored_bytes = 0
    tensor_count = 0
    for shard, header in headers.items():
        for name, meta in header.items():
            shape = meta.get("shape")
            if not isinstance(shape, list):
                raise ValueError(f"{shard}:{name}: malformed tensor shape")
            numel = 1
            for dimension in shape:
                numel *= int(dimension)
            key = component(name)
            groups[key] = groups.get(key, 0) + numel
            dtypes[meta["dtype"]] = dtypes.get(meta["dtype"], 0) + numel
            stored_bytes += numel * DTYPE_BYTES.get(meta["dtype"], 0)
            tensor_count += 1
            sample = examples.setdefault(key, [])
            if len(sample) < 4 and not any(
                name.split(".")[:3] == seen.split(".")[:3] for seen in sample
            ):
                sample.append(name)
    text_decoder = sum(
        groups.get(k, 0)
        for k in (
            "text_embeddings",
            "text_other",
            "text_moe_experts",
            "text_moe_router",
        )
    )
    return {
        "tensor_count": tensor_count,
        "stored_parameter_count": sum(groups.values()),
        "by_component": dict(sorted(groups.items())),
        "by_dtype": dict(sorted(dtypes.items())),
        "component_name_examples": dict(sorted(examples.items())),
        "stored_tensor_bytes": stored_bytes,
        "text_decoder_parameters": text_decoder,
    }


def moe_active(config: dict[str, Any], counts: dict[str, Any]) -> dict[str, Any] | None:
    text = config.get("text_config", config)
    experts = text.get("num_experts") or text.get("num_local_experts")
    top_k = text.get("top_k_experts") or text.get("num_experts_per_tok")
    if not experts or not top_k:
        return None
    groups = counts["by_component"]
    expert_params = groups.get("text_moe_experts", 0)
    dense = counts["text_decoder_parameters"] - expert_params
    active = dense + expert_params * top_k // experts
    return {
        "num_experts": experts,
        "top_k_experts": top_k,
        "expert_parameters": expert_params,
        "non_expert_text_parameters": dense,
        "active_text_parameters_per_token": active,
        "active_text_parameters_per_token_excluding_embeddings": active
        - groups.get("text_embeddings", 0),
        "rule": "non-expert text decoder parameters + expert parameters * top_k / num_experts",
    }


def license_facts(
    readme: str | None, license_text: str | None, tags: list[str]
) -> dict[str, Any]:
    front = {}
    if readme and readme.startswith("---"):
        end = readme.find("\n---", 3)
        for line in readme[3:end].splitlines():
            match = re.match(
                r"^(license|license_name|license_link|base_model|pipeline_tag|library_name):\s*(.*)$",
                line.strip(),
            )
            if match:
                front[match.group(1)] = match.group(2).strip().strip("'\"")
    lowered = (readme or "").lower()
    return {
        "api_license_tags": [tag for tag in tags if tag.startswith("license:")],
        "card_front_matter": front,
        "license_file_present": license_text is not None,
        "license_file_first_line": (
            license_text.strip().splitlines()[0][:120] if license_text else None
        ),
        "license_file_is_apache_2": bool(
            license_text
            and "Apache License" in license_text
            and "Version 2.0" in license_text
        ),
        "card_mentions_gemma_terms_of_use": "gemma terms of use" in lowered,
        "card_mentions_prohibited_use_policy": "prohibited use policy" in lowered,
        "card_mentions_apache_2": "apache 2.0" in lowered
        or "apache-2.0" in lowered
        or "apache license" in lowered,
    }


def native_output_facts(
    files: dict[str, bytes], config: dict[str, Any]
) -> dict[str, Any]:
    tokenizer_config = (
        json.loads(files["tokenizer_config.json"])
        if "tokenizer_config.json" in files
        else {}
    )
    template = files.get("chat_template.jinja", b"").decode("utf-8", "replace") or str(
        tokenizer_config.get("chat_template") or ""
    )
    generation = (
        json.loads(files["generation_config.json"])
        if "generation_config.json" in files
        else {}
    )
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "text_model_type": config.get("text_config", {}).get("model_type"),
        "has_vision_config": "vision_config" in config,
        "has_audio_config": bool(config.get("audio_config")),
        "chat_template_present": bool(template),
        "chat_template_mentions_think": "think" in template,
        "chat_template_sha256": (
            hashlib.sha256(template.encode("utf-8")).hexdigest() if template else None
        ),
        "generation_config": {
            k: generation.get(k)
            for k in ("do_sample", "temperature", "top_p", "top_k", "max_new_tokens")
            if k in generation
        },
        "native_interface": "free-text generation through the language-model head; no native Choice/Noul/Score probabilities",
    }


def architecture_facts(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config", config)
    keys = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "moe_intermediate_size",
        "num_experts",
        "top_k_experts",
        "vocab_size",
        "max_position_embeddings",
        "sliding_window",
        "full_attention_interval",
        "mtp_num_hidden_layers",
        "tie_word_embeddings",
        "final_logit_softcapping",
    )
    facts = {key: text.get(key) for key in keys if key in text}
    if isinstance(text.get("layer_types"), list):
        kinds: dict[str, int] = {}
        for kind in text["layer_types"]:
            kinds[kind] = kinds.get(kind, 0) + 1
        facts["layer_type_counts"] = kinds
    return facts


def verify_candidate(key: str, repo: str, revision: str) -> dict[str, Any]:
    info = model_info(repo, revision)
    siblings = info.get("siblings") or []
    names = {item["rfilename"]: item for item in siblings}
    files: dict[str, bytes] = {}
    for filename in SMALL_FILES:
        if filename in names:
            files[filename] = _get(f"{HUB}/{repo}/resolve/{revision}/{filename}")
    config = json.loads(files["config.json"])
    shards = sorted(name for name in names if name.endswith(".safetensors"))
    headers = {shard: safetensors_header(repo, revision, shard) for shard in shards}
    counts = count_parameters(headers)
    index_total = None
    if "model.safetensors.index.json" in files:
        index = json.loads(files["model.safetensors.index.json"])
        index_total = index.get("metadata", {}).get("total_size")
        mapped = set(index.get("weight_map", {}))
        stored = {name for header in headers.values() for name in header}
        if mapped != stored:
            raise ValueError(f"{repo}: index weight_map differs from shard headers")
    readme = files.get("README.md", b"").decode("utf-8", "replace") or None
    license_text = files.get("LICENSE", b"").decode("utf-8", "replace") or None
    return {
        "key": key,
        "repo_id": repo,
        "revision": revision,
        "api_sha_matches": info.get("sha") == revision,
        "gated": info.get("gated"),
        "private": info.get("private"),
        "disabled": info.get("disabled"),
        "api_safetensors_summary": info.get("safetensors"),
        "files": {
            name: {
                "size": item.get("size"),
                "lfs_sha256": (item.get("lfs") or {}).get("sha256"),
            }
            for name, item in sorted(names.items())
        },
        "small_file_sha256": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(files.items())
        },
        "index_total_size_bytes": index_total,
        "parameters": counts,
        "moe": moe_active(config, counts),
        "architecture": architecture_facts(config),
        "license": license_facts(readme, license_text, info.get("tags") or []),
        "native_output": native_output_facts(files, config),
    }


def absent(repo: str) -> dict[str, Any]:
    try:
        _get(f"{HUB}/api/models/{repo}")
        return {"repo_id": repo, "exists": True}
    except urllib.error.HTTPError as exc:
        return {"repo_id": repo, "exists": False, "http_status": exc.code}


def check_local(
    receipt: dict[str, Any], key: str, snapshot: str | Path
) -> dict[str, Any]:
    """Hash a downloaded snapshot against the remote LFS and small-file digests."""
    snapshot = Path(snapshot)
    candidate = receipt["candidates"][key]
    mismatched, missing, checked = [], [], 0
    for name, meta in candidate["files"].items():
        path = snapshot / name
        if not path.is_file():
            if (
                name.endswith((".safetensors", ".json", ".jinja", ".txt"))
                or name == "LICENSE"
            ):
                missing.append(name)
            continue
        expected = meta.get("lfs_sha256") or candidate["small_file_sha256"].get(name)
        if expected is None:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 24), b""):
                digest.update(block)
        checked += 1
        if digest.hexdigest() != expected:
            mismatched.append(name)
    return {
        "key": key,
        "revision": candidate["revision"],
        "files_checked": checked,
        "missing": missing,
        "mismatched": mismatched,
        "passed": not missing and not mismatched and checked > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--receipt", type=Path, help="Existing receipt for --check-local"
    )
    parser.add_argument(
        "--check-local", action="append", default=[], help="KEY=SNAPSHOT_DIR"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a verification receipt")
    if args.check_local:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        results = [
            check_local(receipt, *spec.split("=", 1)) for spec in args.check_local
        ]
        payload = {
            "schema_version": "decision2-27b-local-snapshot-check/1",
            "remote_receipt_sha256": hashlib.sha256(
                args.receipt.read_bytes()
            ).hexdigest(),
            "results": [
                dict(item, snapshot=Path(spec.split("=", 1)[1]).name)
                for item, spec in zip(results, args.check_local)
            ],
        }
        args.output.write_text(
            json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload["results"]))
        raise SystemExit(0 if all(item["passed"] for item in results) else 1)
    started = time.time()
    receipt = {
        "schema_version": "decision2-27b-candidate-verification/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "HF API revision info, small files and safetensors headers by HTTP range; no weight shard downloaded",
        "candidates": {
            key: verify_candidate(key, repo, rev)
            for key, (repo, rev) in CANDIDATES.items()
        },
        "absent_repositories": [absent(repo) for repo in ABSENT_CHECKS],
    }
    receipt["elapsed_seconds"] = round(time.time() - started, 3)
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=1, sort_keys=True)
        stream.write("\n")
    summary = {
        key: {
            "stored": item["parameters"]["stored_parameter_count"],
            "by_component": item["parameters"]["by_component"],
            "moe_active": (item["moe"] or {}).get("active_text_parameters_per_token"),
            "license_tags": item["license"]["api_license_tags"],
        }
        for key, item in receipt["candidates"].items()
    }
    print(
        json.dumps(
            {"summary": summary, "absent": receipt["absent_repositories"]}, indent=1
        )
    )


if __name__ == "__main__":
    main()
