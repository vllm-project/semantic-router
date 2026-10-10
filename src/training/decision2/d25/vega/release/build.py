"""Build a Decision 2.5 Hugging Face package from a code-readout v1 export (stdlib only).

    python -m d25.vega.release.build --export <export dir> --out <package dir> \
        --model-name Decision-2.5-Vega-27B --repo-id vllm-sr/Decision-2.5-Vega-27B \
        --card <dir with README.md and assets/> [--teachers teachers.json] [--copy]

The package root is the export itself (``config.json`` + ``model*.safetensors`` of a ``Qwen3_5Model``,
``readout.safetensors``, ``decision_config.json``, tokenizer files), so every code-readout loader reads it
unchanged, plus:

- remote code (``release/package/*.py``) and ``auto_map``/``custom_pipelines`` in ``config.json`` for
  ``AutoModel.from_pretrained(repo, trust_remote_code=True).system_one(...)``;
- ``decision25_format.py``: ``d25/vega/common/decision_format.py`` byte for byte (the training prompt);
- ``LICENSE`` (Apache-2.0), the card (``README.md``, ``assets/``) and ``MODEL_MANIFEST.json`` (SHA-256 and
  size of every file, parameter counts, the weights identity, sanitized provenance, licence, teachers).

Weights are hard-linked when the export is on the same file system (``--copy`` copies). The only edited
files are ``config.json`` (two keys added) and ``decision_config.json`` (absolute paths in ``provenance``
reduced to their last component); ``identity.model_sha256`` covers the weights, the tokenizer and the
inference fields of ``decision_config.json``, so an export and its package have the same identity. The
build fails on private paths, addresses, host names or credential-like text in any text file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PACKAGE_SOURCE = HERE / "package"
FORMAT_SOURCE = HERE.parent / "common" / "decision_format.py"
FORMAT_FILE = "decision25_format.py"
SCHEMA = "d25-package-manifest/1"
MANIFEST = "MODEL_MANIFEST.json"
CODE_FILES = (
    "decision25_runtime.py",
    "modeling_decision25.py",
    "pipeline_decision25.py",
    "decision25_server.py",
    "decision25_engine.py",
    "requirements.txt",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
# Tokenizer vocabularies are upstream bytes; their tokens are not prose and are not linted.
NOT_LINTED = ("tokenizer.json", "vocab.json", "merges.txt")
INFERENCE_FIELDS = (
    "format_version",
    "format_id",
    "prompt",
    "base_model",
    "revision",
    "codes",
    "token_ids",
    "temperature",
    "attention_mode",
    "pooling",
    "max_length",
    "readout_dtype",
)
IDENTITY_FILES = (
    "readout.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
AUTO_MAP = {"AutoModel": "modeling_decision25.Decision25Model"}
CUSTOM_PIPELINES = {
    "decision": {
        "impl": "pipeline_decision25.Decision25Pipeline",
        "pt": ["AutoModel"],
        "type": "text",
    }
}
MODEL_NAME = re.compile(
    r"Decision-2\.5-(?P<codename>[A-Z][a-z]+)-(?P<size>[0-9.]+B)\Z|d25-[a-z0-9.-]+\Z"
    r"|d3(?:-(?:flash|mini|nano|lite|edge))?\Z"
)
# Public d3-family packages: their own runtime file names (release/package_d3/, generated from release/package/),
# public format and prompt ids, no provenance block, a manifest without sources, and a trace lint over every file.
D3_NAME = re.compile(r"d3(?:-(?:flash|mini|nano|lite|edge))?\Z")
D3_SOURCE = HERE / "package_d3"
D3_CODE_FILES = (
    "d3_runtime.py",
    "d3_fast.py",
    "d3_kernels.py",
    "modeling_d3.py",
    "pipeline_d3.py",
    "d3_server.py",
    "d3_engine.py",
    "requirements.txt",
)
D3_FORMAT_FILE = "d3_format.py"
D3_SCHEMA = "d3-package-manifest/1"
D3_DECISION = {"format_id": "d3-code-readout-v1", "prompt": "d3"}
D3_AUTO_MAP = {"AutoModel": "modeling_d3.D3Model"}
D3_PIPELINES = {
    "decision": {**CUSTOM_PIPELINES["decision"], "impl": "pipeline_d3.D3Pipeline"}
}
D3_NOTICE = "{name} (Decision 3.0)\nCopyright 2026 vLLM Semantic Router Team\n\nBuilt on {base} (Apache-2.0).\n"
PUBLIC_TRACES = (
    re.compile(
        r"decision25|Decision25|DECISION25|\bd25\b|d25[-_]|(?<![\d.])2\.5(?![\d])|\bVega\b|\bvega\b|\b[Oo]mni\b|"
        r"teacher|distill|soft[ -]label|\baudit|decontam|campaign|\bws-[a-z]|w1-t2|step-00|\btrain(?:ed|ing|er)\b|\btrain from\b|"
        r"rows_seen|data_sha256",
        re.I,
    ),
    "internal name or training trace",
)
# Public board entrants whose names contain a traced word; family cards list them as same-size peers.
PUBLIC_NAMES = ("Jev-Omni", "d1-omni-600M", "GLiNER2.5-Decide")


def traced(line: str) -> bool:
    for name in PUBLIC_NAMES:
        line = line.replace(name, "")
    return bool(PUBLIC_TRACES[0].search(line))


def same_contract(public: Path, internal: Path) -> list[str]:
    """Top-level functions and constants of the public format module equal the internal contract (AST)."""
    import ast

    def defs(path: Path) -> dict[str, str]:
        out = {}
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef):
                out[node.name] = ast.dump(node)
            elif isinstance(node, ast.Assign):
                out[ast.unparse(node.targets[0])] = ast.dump(node)
        return out

    mine, theirs = defs(public), defs(internal)
    return [k for k in mine if k != "FORMAT_ID" and theirs.get(k) != mine[k]]


LEAKS = (
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "IP address"),
    (re.compile(r"\b(?:[a-z0-9]+-)*node[-_ ]?\d{1,3}\b", re.I), "machine identity"),
    (re.compile(r"/data/|/root/|/home/|/traindata/"), "private path"),
    (
        re.compile(r"hf_[A-Za-z0-9]{20,}|gh[opsu]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"),
        "credential-like text",
    ),
)
# Site-specific patterns (real host names, fleet details) stay out of git: one regex per line in this file.
PRIVATE_PATTERNS_ENV = "D25_LEAK_PATTERNS"


def leak_patterns() -> list[tuple[re.Pattern, str]]:
    patterns = list(LEAKS)
    path = os.environ.get(PRIVATE_PATTERNS_ENV)
    if path:
        lines = Path(path).read_text().splitlines()
        patterns += [
            (re.compile(line.strip(), re.I), "site-specific pattern")
            for line in lines
            if line.strip() and not line.startswith("#")
        ]
    return patterns


_HASHES: dict[tuple[int, int, int, int], str] = {}


def sha256_file(path: Path) -> str:
    """SHA-256 of a file; hard links of one unchanged inode (export and package weights) are hashed once."""
    stat = os.stat(path)
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(16 << 20), b""):
                digest.update(block)
        _HASHES[key] = digest.hexdigest()
    return _HASHES[key]


def weight_files(directory: Path) -> list[str]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return ["model.safetensors.index.json", *names]
    if (directory / "model.safetensors").is_file():
        return ["model.safetensors"]
    raise FileNotFoundError(f"{directory}: no model*.safetensors")


def tensor_counts(path: Path) -> dict[str, int]:
    """Parameter counts per top-level key prefix from a safetensors header (no weights read)."""
    with open(path, "rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        if not 2 <= size <= 256 << 20:
            raise ValueError(f"{path.name}: invalid safetensors header")
        header = json.loads(stream.read(size))
    counts: dict[str, int] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        group = "vision" if name.startswith(("visual.", "model.visual.")) else "text"
        counts[group] = counts.get(group, 0) + math.prod(meta["shape"])
    return counts


def model_identity(
    directory: Path, decision: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Identity of the inference-relevant bytes; equal for an export and the package built from it."""
    directory = Path(directory)
    decision = (
        decision
        if decision is not None
        else json.loads((directory / "decision_config.json").read_text())
    )
    names = [n for n in weight_files(directory) if n.endswith(".safetensors")]
    names += [n for n in IDENTITY_FILES if (directory / n).is_file()]
    files = {name: sha256_file(directory / name) for name in sorted(names)}
    fields = {k: decision[k] for k in INFERENCE_FIELDS if k in decision}
    blob = json.dumps(
        {"files": files, "decision": fields}, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "model_sha256": hashlib.sha256(blob).hexdigest(),
        "files_sha256": files,
        "decision": fields,
    }


def sanitize(value: Any) -> Any:
    """Provenance without absolute paths or machine names (the last path component is kept)."""
    if isinstance(value, dict):
        return {sanitize(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        if value.startswith("/"):
            return "local:" + (Path(value).name or "root")
        for pattern, _ in leak_patterns():
            value = pattern.sub("<redacted>", value)
    return value


def lint_text(name: str, text: str) -> list[str]:
    problems = []
    for pattern, reason in leak_patterns():
        found = pattern.findall(text)
        if reason == "IP address":
            found = [ip for ip in found if ip not in ("127.0.0.1", "0.0.0.0")]
        if found:
            problems.append(f"{name}: {reason}")
    return problems


def validate_export(export: Path) -> dict[str, Any]:
    for name in (
        "config.json",
        "decision_config.json",
        "readout.safetensors",
        "tokenizer.json",
    ):
        if not (export / name).is_file():
            raise FileNotFoundError(f"{export}: missing {name}")
    config = json.loads((export / "config.json").read_text())
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != [
        "Qwen3_5Model"
    ]:
        raise ValueError("config.json must describe a Qwen3_5Model backbone")
    decision = json.loads((export / "decision_config.json").read_text())
    problems = []
    if decision.get("format_version") != 1:
        problems.append("format_version must be 1")
    if decision.get("prompt", "pplx") not in ("d25-vega", "pplx"):
        problems.append("unknown prompt family")
    if decision.get("attention_mode", "causal") not in (
        "causal",
        "noncausal_full_attention",
    ):
        problems.append("unknown attention_mode")
    if decision.get("pooling", "last") != "last":
        problems.append("pooling must be last")
    if (
        len(decision.get("codes") or []) != 255
        or len(decision.get("token_ids") or []) != 255
    ):
        problems.append("codes and token_ids must list 255 entries")
    if problems:
        raise ValueError(f"{export}/decision_config.json: " + "; ".join(problems))
    readout = tensor_counts(export / "readout.safetensors")
    hidden = config["text_config"]["hidden_size"]
    if readout.get("text") != 255 * hidden:
        raise ValueError(
            f"readout.safetensors holds {readout} values, expected 255 x {hidden}"
        )
    return decision


def place(source: Path, target: Path, copy: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        try:
            os.link(source, target)
            return "link"
        except OSError:
            pass
    shutil.copyfile(source, target)
    return "copy"


def build(
    export: Path,
    out: Path,
    *,
    model_name: str,
    repo_id: str,
    card: Path | None,
    teachers: list[dict] | None,
    copy: bool = False,
    licence_components: list[dict] | None = None,
    no_context_limit: bool = False,
) -> dict[str, Any]:
    export, out = Path(export).resolve(), Path(out)
    if not MODEL_NAME.match(model_name):
        raise ValueError(f"not a Decision model name: {model_name}")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    decision = validate_export(export)
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    placed: dict[str, str] = {}
    for name in [
        *weight_files(export),
        "readout.safetensors",
        *[n for n in TOKENIZER_FILES if (export / n).is_file()],
    ]:
        placed[name] = place(export / name, partial / name, copy)
    skipped = sorted(
        p.name
        for p in export.iterdir()
        if p.name not in placed
        and p.name not in ("config.json", "decision_config.json")
    )

    public = bool(D3_NAME.match(model_name))
    code_files, format_file = (
        (D3_CODE_FILES, D3_FORMAT_FILE) if public else (CODE_FILES, FORMAT_FILE)
    )
    config = json.loads((export / "config.json").read_text())
    config["auto_map"] = D3_AUTO_MAP if public else AUTO_MAP
    config["custom_pipelines"] = D3_PIPELINES if public else CUSTOM_PIPELINES
    (partial / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    released = dict(decision)
    if no_context_limit:
        released["max_length"] = None
    if public:
        released.pop("provenance", None)
        released.update(D3_DECISION)
    else:
        released["provenance"] = sanitize(decision.get("provenance") or {})
    (partial / "decision_config.json").write_text(json.dumps(released, indent=2) + "\n")

    if public:
        for name in code_files:
            shutil.copyfile(D3_SOURCE / name, partial / name)
        shutil.copyfile(D3_SOURCE / format_file, partial / format_file)
        differs = same_contract(partial / format_file, FORMAT_SOURCE)
        if differs:
            raise RuntimeError(
                f"{format_file} differs from the internal prompt contract: {differs}"
            )
        shutil.copyfile(D3_SOURCE / "LICENSE", partial / "LICENSE")
        base = decision.get("base_model") or "the base model"
        (partial / "NOTICE").write_text(
            D3_NOTICE.format(name=model_name, base=base), encoding="utf-8"
        )
    else:
        for name in CODE_FILES:
            shutil.copyfile(PACKAGE_SOURCE / name, partial / name)
        shutil.copyfile(FORMAT_SOURCE, partial / FORMAT_FILE)
        if sha256_file(partial / FORMAT_FILE) != sha256_file(FORMAT_SOURCE):
            raise RuntimeError(
                "decision25_format.py differs from d25/vega/common/decision_format.py"
            )
        shutil.copyfile(PACKAGE_SOURCE / "LICENSE", partial / "LICENSE")
    if card is not None:
        shutil.copyfile(card / "README.md", partial / "README.md")
        for asset in (
            sorted((card / "assets").glob("*.png"))
            if (card / "assets").is_dir()
            else []
        ):
            place(asset, partial / "assets" / asset.name, copy=True)

    problems = []
    for path in sorted(partial.rglob("*")):
        if (
            path.is_file()
            and path.suffix in (".json", ".py", ".md", ".txt", ".jinja")
            and path.name not in NOT_LINTED
        ):
            name = path.relative_to(partial).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            problems += lint_text(name, text)
            if public:
                problems += [
                    f"{name}:{n}: {PUBLIC_TRACES[1]} ({line.strip()[:80]})"
                    for n, line in enumerate(text.splitlines(), 1)
                    if traced(line)
                ]
    if public:
        problems += [
            f"file name {p.name}: {PUBLIC_TRACES[1]}"
            for p in partial.rglob("*")
            if PUBLIC_TRACES[0].search(p.name)
        ]
    if problems:
        shutil.rmtree(partial)
        raise ValueError("package lint failed: " + "; ".join(problems))

    identity = model_identity(partial, released)
    expected = {**decision, "max_length": None} if no_context_limit else dict(decision)
    if public:
        expected.update(D3_DECISION)
    if identity["model_sha256"] != model_identity(export, expected)["model_sha256"]:
        raise RuntimeError("package identity differs from the export's")
    counts: dict[str, int] = {}
    for name in weight_files(partial):
        if name.endswith(".safetensors"):
            for group, n in tensor_counts(partial / name).items():
                counts[group] = counts.get(group, 0) + n
    readout = 255 * config["text_config"]["hidden_size"]
    files = sorted(
        p.relative_to(partial).as_posix() for p in partial.rglob("*") if p.is_file()
    )
    manifest = {
        "schema": SCHEMA,
        "model_name": model_name,
        "repo_id": repo_id,
        "format_id": decision.get("format_id"),
        "prompt": decision.get("prompt", "pplx"),
        "attention_mode": decision.get("attention_mode", "causal"),
        "max_input_tokens": released.get("max_length"),
        "identity": identity,
        "parameters": {
            "text": counts.get("text", 0),
            "vision": counts.get("vision", 0),
            "readout": readout,
            "loaded": counts.get("text", 0) + counts.get("vision", 0) + readout,
        },
        "base_model": {
            "repo_id": decision.get("base_model"),
            "revision": decision.get("revision"),
            "relation": "finetune",
        },
        "source": {
            "export": export.name,
            "decision_config_sha256": sha256_file(export / "decision_config.json"),
            "config_sha256": sha256_file(export / "config.json"),
            "not_packaged": skipped,
        },
        "runtime": {
            "files_sha256": {
                n: sha256_file(partial / n) for n in (*code_files, format_file)
            },
            "decision_format_source": "src/training/decision2/d25/vega/common/decision_format.py",
        },
        "licence": {
            "spdx": "apache-2.0",
            "components": licence_components
            or [
                {
                    "component": decision.get("base_model") or "base model",
                    "licence": "apache-2.0",
                },
                {
                    "component": (
                        f"{model_name} (Decision 3.0) weights, readout and runtime"
                        if model_name.startswith("d3")
                        else "Decision 2.5 weights, readout and runtime"
                    ),
                    "licence": "apache-2.0",
                },
            ],
        },
        "teachers": teachers or [],
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files_sha256": {name: sha256_file(partial / name) for name in files},
        "files_bytes": {name: (partial / name).stat().st_size for name in files},
    }
    if public:
        manifest.update(
            schema=D3_SCHEMA,
            format_id=released["format_id"],
            prompt=released["prompt"],
            base_model={
                "repo_id": decision.get("base_model"),
                "revision": decision.get("revision"),
            },
            runtime={
                "files_sha256": {
                    n: sha256_file(partial / n) for n in (*code_files, format_file)
                }
            },
        )
        for key in ("source", "teachers"):
            manifest.pop(key)
        if PUBLIC_TRACES[0].search(json.dumps(manifest)):
            shutil.rmtree(partial)
            raise ValueError("MODEL_MANIFEST.json: " + PUBLIC_TRACES[1])
    (partial / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    partial.rename(out)
    return {
        "package": str(out),
        "model_name": model_name,
        "repo_id": repo_id,
        "model_sha256": identity["model_sha256"],
        "files": len(files) + 1,
        "bytes": sum(manifest["files_bytes"].values()),
        "parameters": manifest["parameters"],
        "placed": {
            k: sum(1 for v in placed.values() if v == k) for k in ("link", "copy")
        },
        "not_packaged": skipped,
        "manifest_sha256": sha256_file(out / MANIFEST),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--no-context-limit",
        action="store_true",
        help="release without the export's max_length: the runtime then refuses no input length",
    )
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument(
        "--card",
        type=Path,
        help="directory with README.md and assets/ (from d25.vega.release.card)",
    )
    ap.add_argument(
        "--teachers", type=Path, help='JSON list of {"model", "licence", "role"}'
    )
    ap.add_argument(
        "--copy", action="store_true", help="copy weights instead of hard-linking them"
    )
    ap.add_argument(
        "--identity-only",
        action="store_true",
        help="print the identity of --export and exit",
    )
    args = ap.parse_args(argv)
    if args.identity_only:
        print(json.dumps(model_identity(args.export), indent=1))
        return 0
    teachers = json.loads(args.teachers.read_text()) if args.teachers else None
    print(
        json.dumps(
            build(
                args.export,
                args.out,
                model_name=args.model_name,
                repo_id=args.repo_id,
                card=args.card,
                teachers=teachers,
                copy=args.copy,
                no_context_limit=args.no_context_limit,
            ),
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
