"""JevArena-C1 v1.1 scoring event 3: model table, plan and gold-free checks for ``event3.sh``.

    python3 -m v2.eval.sealed.event3 plan --table T [--models K,...] [--c27 f1|f2] [--peers27 K,...] \
        [--c27-package DIR --c27-manifest SHA --c27-repo ID --c27-revision REV] \
        [--c9b-package DIR --c9b-manifest SHA [--c9b-repo ID] [--c9b-revision REV] [--c9b-identity SHA] \
         [--c9b-calibration FILE|none] [--c9b-parity-stored PATH] [--c9b-tolerance X] \
         [--c9b-cache DIR --c9b-cache-sha SHA]] [--site node-a|node-b] \
        [--allow-deviation ID]... --output PLAN.json
    python3 -m v2.eval.sealed.event3 show --plan PLAN.json
    python3 -m v2.eval.sealed.event3 argv --plan P --key K --phase smoke|collect --run-dir D --gpu N \
        --src MIRROR --src-root S --lease-name owner.NAME [--shared] [--cache-dir C]
    python3 -m v2.eval.sealed.event3 verify --plan P --src-root S --output VERIFY.json
    python3 -m v2.eval.sealed.event3 parity --plan P --key K --run-dir D --output PARITY.json
    python3 -m v2.eval.sealed.event3 stored --plan P --output STORED-C1.json
    python3 -m v2.eval.sealed.event3 pairs --plan P --event-dir E
    python3 -m v2.eval.sealed.event3 summary --plan P --event-dir E --preflight-dir D --output S.json
    python3 -m v2.eval.sealed.event3 stage --plan P
    python3 -m v2.eval.sealed.event3 keys --plan P [--collected | --stored]
    python3 -m v2.eval.sealed.event3 field --plan P --key K --field cache.frozen
    python3 -m v2.eval.sealed.event3 digest DIR
    python3 -m v2.eval.sealed.event3 c27-receipt --package-dir DIR \
        (--formal-run DIR | --typed-final F --public231 F --collect F) [--seal F] [--report F] \
        [--calibration FILE] [--cache DIR] [--base DIR] --output RECEIPT.json
    python3 -m v2.eval.sealed.event3 fill-c27 --table T --key f2 --receipt R [--repo ID] \
        [--revision REV] [--include 'SUBDIR/*'] [--adopt-paths] [--output T2]

``plan`` resolves the table (``event3-models.json``) and enforces the C1 limits: per size at most one
candidate, one Decision 1.0 configuration and two open peers; internal-only peers, a second scoring of
a 1.0 model or a second candidate of a size need a recorded deviation approval. The 9B candidate's
package dir and manifest hash are parameters: the weights identity is read from that package's
MODEL_MANIFEST.json, whose SHA-256 must be ``--c9b-manifest``, and must be in the row's
``identity_allowed`` (or equal ``--c9b-identity``). ``argv`` prints the runner arguments
NUL-separated. ``verify`` checks images, mirror modules, paths, release-package manifests and pinned
tree digests. ``parity`` compares a smoke (typed FINAL and public 231, which together cover every
decision type) with the model's stored formal run. None of these reads C1 prompts or gold; ``stored``
hashes stored C1 prediction files and seals, and ``event3.sh`` calls it only in the event phase.

``c27-receipt`` (CPU, on the node that holds a ~27B candidate's formal run) records what the table
pins for that candidate with the functions ``verify`` uses: the package check, calibration, stored
predictions (against the run's SEAL.json and the manifest identity), COLLECT, v3 and the frozen
cache. ``fill-c27`` writes a passed receipt into a ``candidates_27b`` block (only that block's text
changes) and refuses while any PARENT-FILLS remains in it.

Tree digest (as ``triton_cache.py``, but following file symlinks, e.g. HF snapshots)::

    cd DIR && find -L . -type f ! -path './.cache/*' ! -path './.git/*' ! -path '*/__pycache__/*' \
        -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "dev2-c1-event3/1"
PLACEHOLDER = "PARENT-FILLS"
PROMPTS_SHA256 = "0b29686f60c980f3fbc8a03b88537fc0bf90ee967afa67d4fe0c958b1bfde16a"
TIERS = ("0.6B", "0.8B", "2B", "4B", "9B", "27B")
ROLES = ("candidate", "own1", "peer", "internal")
PARITY_MODES = ("exact", "near", "none")
SMOKE_PANELS = ("typed-final", "public231")
C1_PANEL = "sealed-c1"
EXCLUDED_TOP = (".cache", ".git")
NUMERIC_KEYS = ("noul", "score", "confidence")
SRC_ROOT = Path(__file__).resolve().parents[3]
C9B_REFUSAL = (
    "cand9b (DEV2.0-8B) needs its frozen release package on this node: pass --c9b-package DIR"
    " and --c9b-manifest SHA, or leave the 9B rows out with --models"
)
RECEIPT_ROLES = ("calibration", "typed-final", "public231", "collect")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_new(path: Path, value: Any) -> str:
    data = (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as target:
        target.write(data)
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ digests


def tree_files(root: Path) -> list[str]:
    """Relative paths of the files the documented ``find -L`` recipe lists."""
    found = []
    for folder, dirs, names in os.walk(root, followlinks=True):
        rel_folder = os.path.relpath(folder, root)
        dirs[:] = [
            d
            for d in dirs
            if d != "__pycache__" and not (rel_folder == "." and d in EXCLUDED_TOP)
        ]
        for name in names:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            rel = name if rel_folder == "." else f"{rel_folder}/{name}"
            if "\\" in rel or "\n" in rel:
                raise ValueError(f"sha256sum would escape the name {rel!r}")
            found.append(rel.replace(os.sep, "/"))
    return found


def hash_many(paths: list[Path], workers: int = 8) -> list[str]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(sha_file, paths))


def tree_digest(root: Path, workers: int = 8) -> tuple[str, int]:
    rels = tree_files(root)
    hashes = hash_many([root / rel for rel in rels], workers)
    lines = sorted(
        (f"./{rel}".encode(), f"{sha}  ./{rel}\n".encode())
        for rel, sha in zip(rels, hashes)
    )
    return hashlib.sha256(b"".join(line for _, line in lines)).hexdigest(), len(rels)


# -------------------------------------------------------------------- table


def expand(value: Any, roots: dict[str, str]) -> Any:
    if isinstance(value, str):
        out = re.sub(
            r"\$\{([A-Z0-9_]+)\}",
            lambda m: roots[m.group(1)] if m.group(1) in roots else m.group(0),
            value,
        )
        if "${" in out:
            raise ValueError(f"unknown root in {value!r}")
        return out
    if isinstance(value, list):
        return [expand(v, roots) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, roots) for k, v in value.items()}
    return value


def placeholders(value: Any, where: str = "") -> list[str]:
    if isinstance(value, str):
        return [where or "value"] if PLACEHOLDER in value else []
    if isinstance(value, list):
        return [
            p for i, v in enumerate(value) for p in placeholders(v, f"{where}[{i}]")
        ]
    if isinstance(value, dict):
        return [
            p
            for k, v in value.items()
            if k not in ("notes", "node_b")
            for p in placeholders(v, f"{where}.{k}" if where else k)
        ]
    return []


def set_dotted(row: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    target = row
    for part in parts[:-1]:
        if not isinstance(target.get(part), dict):
            raise ValueError(f"cannot set {dotted}: {part} is not an object")
        target = target[part]
    target[parts[-1]] = value


def load_table(path: Path) -> dict[str, Any]:
    table = json.loads(path.read_text(encoding="utf-8"))
    if table.get("schema") != "dev2-c1-event3-models/1":
        raise ValueError("not a C1 event-3 model table")
    return table


def resolve_rows(table: dict[str, Any], c27: str | None) -> dict[str, dict[str, Any]]:
    rows = {k: dict(v) for k, v in table["models"].items()}
    if c27 is not None:
        options = table.get("candidates_27b", {})
        if c27 not in options:
            raise ValueError(f"--c27 must be one of {sorted(options)}")
        rows["cand27"] = json.loads(json.dumps(options[c27]))
        rows["cand27"]["c27"] = c27
    return rows


def pick_peers27(
    table: dict[str, Any], rows: dict[str, dict[str, Any]], explicit: list[str] | None
) -> list[str]:
    if explicit is not None:
        chosen = explicit
    else:
        eligible = table.get("card_eligible_27b")
        if not isinstance(eligible, list):
            raise ValueError(
                "27B peers not set: pass --peers27 or fill card_eligible_27b"
                " (the parent fills in the card-eligible peers)"
            )
        unknown = [k for k in eligible if k not in rows]
        if unknown:
            raise ValueError(f"unknown 27B peers: {', '.join(unknown)}")
        chosen = sorted(eligible, key=lambda k: -float(rows[k]["v3"]))[:2]
    for key in chosen:
        if (
            key not in rows
            or rows[key].get("tier") != "27B"
            or rows[key].get("role") != "peer"
        ):
            raise ValueError(f"{key} is not a 27B peer row")
    if len(chosen) > 2:
        raise ValueError("at most two open 27B peers")
    return chosen


def check_rules(
    models: dict[str, dict[str, Any]], allowed: set[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    deviations: list[dict[str, Any]] = []
    by_tier: dict[str, dict[str, list[str]]] = {}
    exempt: set[str] = set()
    for key, row in models.items():
        tier, role = row.get("tier"), row.get("role")
        if tier not in TIERS:
            errors.append(f"{key}: unknown tier {tier!r}")
            continue
        if role not in ROLES:
            errors.append(f"{key}: unknown role {role!r}")
            continue
        if role == "own1" and tier == "27B":
            errors.append(f"{key}: no Decision 1.0 model exists at 27B")
        if not (row.get("stored") or {}).get("reference_only"):
            by_tier.setdefault(tier, {}).setdefault(role, []).append(key)
        deviation = row.get("deviation")
        if deviation:
            approved = deviation.get("approved") or (
                deviation["id"] in allowed and "--allow-deviation"
            )
            deviations.append({"key": key, **deviation, "approved": approved or None})
            if approved:
                exempt.add(key)
            else:
                errors.append(
                    f"{key}: deviation {deviation['id']} needs the coordinator's approval"
                    f" (--allow-deviation {deviation['id']})"
                )
        if role == "internal" and not deviation:
            errors.append(f"{key}: internal-only rows need a recorded deviation")
    for tier, roles in by_tier.items():
        # Rows with an approved deviation are the recorded exceptions to these limits.
        counted = {
            role: [k for k in keys if k not in exempt] for role, keys in roles.items()
        }
        if not roles.get("candidate"):
            errors.append(f"{tier}: comparators without a candidate")
        if len(counted.get("candidate", [])) > 1:
            errors.append(
                f"{tier}: at most one candidate per size ({', '.join(roles['candidate'])})"
            )
        repos: dict[str, list[str]] = {}
        for key in counted.get("own1", []):
            repos.setdefault(models[key].get("repo", key), []).append(key)
        for keys in repos.values():
            if len(keys) > 1:
                errors.append(
                    f"{tier}: one configuration per Decision 1.0 model ({', '.join(keys)})"
                )
        if len(counted.get("peer", [])) > 2:
            errors.append(
                f"{tier}: at most two open peers ({', '.join(roles['peer'])})"
            )
    return errors, deviations


def build_pairs(
    models: dict[str, dict[str, Any]], reference: list[list[str]]
) -> list[dict[str, str]]:
    pairs = []
    for key, row in models.items():
        if row["role"] != "candidate":
            continue
        for other, orow in models.items():
            if (
                other == key
                or orow["tier"] != row["tier"]
                or orow["role"] == "candidate"
            ):
                continue
            if (orow.get("stored") or {}).get("reference_only"):
                continue
            pairs.append({"left": key, "right": other, "kind": "candidate"})
    for left, right, *_ in reference:
        if left in models and right in models:
            pairs.append({"left": left, "right": right, "kind": "reference"})
    return pairs


def apply_c9b(
    row: dict[str, Any], args: argparse.Namespace, roots: dict[str, str]
) -> None:
    """The 9B candidate's frozen release package; its weights identity comes from its manifest."""
    package = getattr(args, "c9b_package", None)
    manifest = getattr(args, "c9b_manifest", None)
    if not (package and manifest):
        raise ValueError(C9B_REFUSAL)
    path = Path(package) / "MODEL_MANIFEST.json"
    if not path.is_file():
        raise ValueError(f"cand9b: no MODEL_MANIFEST.json in {package}")
    actual = sha_file(path)
    if actual != manifest:
        raise ValueError(
            f"cand9b: {path} has SHA-256 {actual}, not --c9b-manifest {manifest}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    identity = (value.get("identity") or {}).get("model_sha256")
    allowed = row.pop("identity_allowed", {})
    override = getattr(args, "c9b_identity", None)
    if override and override != identity:
        raise ValueError(
            f"cand9b: --c9b-identity {override} is not the manifest identity {identity}"
        )
    if identity not in allowed and not override:
        raise ValueError(
            f"cand9b: manifest identity {identity} is not an allowed DEV2.0-8B weights identity"
            f" ({', '.join(allowed)}); --c9b-identity accepts another one"
        )
    row["model_path"] = row["model_dir"] = package
    row["package"].update(dir=package, manifest_sha256=manifest, identity=identity)
    row["identity"] = identity
    if getattr(args, "c9b_repo", None):
        row["repo"] = args.c9b_repo
    row["extra"]["model_id"] = row["repo"]
    row["revision"] = (
        getattr(args, "c9b_revision", None) or f"manifest-sha256:{manifest}"
    )
    cache = getattr(args, "c9b_cache", None)
    cache_sha = getattr(args, "c9b_cache_sha", None)
    if bool(cache) != bool(cache_sha):
        raise ValueError("--c9b-cache DIR and --c9b-cache-sha SHA go together")
    if cache:
        row["cache"] = {"frozen": cache, "sha256": cache_sha}
    if getattr(args, "c9b_tolerance", None) is not None:
        row["parity"]["tolerance"] = args.c9b_tolerance
    calibrated = expand(row.pop("calibrated"), roots)
    calibration = getattr(args, "c9b_calibration", None)
    if calibration and calibration != "none":
        calibration = os.path.abspath(calibration)
        if not os.path.isfile(calibration):
            raise ValueError(f"cand9b: no calibration file {calibration}")
        report = json.loads(Path(calibration).read_text(encoding="utf-8"))
        bound = report.get("model_sha256") if isinstance(report, dict) else None
        if bound != identity:
            raise ValueError(
                f"cand9b: {calibration} is bound to {bound}, not to the package identity"
                f" {identity} (a BF16-storage package runs only at T = 1)"
            )
        row["adapter_spec"] = calibrated["adapter_spec"]
        row["extra"]["calibration"] = calibration
        folder = os.path.dirname(calibration)
        if folder not in row["mounts"]:
            row["mounts"].append(folder)
        row["parity"]["stored"] = calibrated["parity_stored"]
        row["files"] = [
            f
            for f in calibrated["files"]
            if "calibration" not in f["path"] or f["path"] == calibration
        ]
    if getattr(args, "c9b_parity_stored", None):
        row["parity"]["stored"] = args.c9b_parity_stored
        # The pins describe the table's parity files; another stored run is checked by existence.
        row["files"] = [
            f for f in row.get("files", []) if "predictions" not in f["path"]
        ]


def plan(args: argparse.Namespace) -> int:
    table = load_table(args.table)
    roots = table["roots"]
    selection = args.models.split(",") if args.models else list(table["default_models"])
    c27 = args.c27 if any(k == "cand27" for k in selection) else None
    if c27 is None and "cand27" in selection:
        raise ValueError("cand27 needs --c27")
    rows = resolve_rows(table, c27)
    if "cand27" in selection:
        peers = pick_peers27(
            table, rows, args.peers27.split(",") if args.peers27 is not None else None
        )
        selection = [
            k
            for k in selection
            if rows.get(k, {}).get("tier") != "27B" or k == "cand27"
        ]
        selection += [p for p in peers if p not in selection]
    if len(set(selection)) != len(selection):
        raise ValueError("duplicate model keys")
    unknown = [k for k in selection if k not in rows]
    if unknown:
        raise ValueError(f"unknown model keys: {', '.join(unknown)}")
    models: dict[str, dict[str, Any]] = {}
    for key in selection:
        row = json.loads(json.dumps(rows[key]))
        if args.site == "node-b":
            for dotted, value in (row.get("node_b") or {}).items():
                set_dotted(row, dotted, value)
        if key == "cand27":
            for flag, dotted in (
                ("c27_package", "package.dir"),
                ("c27_manifest", "package.manifest_sha256"),
                ("c27_repo", "repo"),
                ("c27_revision", "revision"),
            ):
                value = getattr(args, flag)
                if value:
                    set_dotted(row, dotted, value)
            if args.c27_package:
                row["model_path"] = row["model_dir"] = args.c27_package
        if key == "cand9b":
            apply_c9b(row, args, roots)
        row.pop("node_b", None)
        models[key] = expand(row, roots)
    errors = [
        f"{k}: {PLACEHOLDER} in {', '.join(placeholders(r))}"
        for k, r in models.items()
        if placeholders(r)
    ]
    rule_errors, deviations = check_rules(models, set(args.allow_deviation))
    errors += rule_errors
    for key, row in models.items():
        if row.get("stored") is None:
            if row["parity"]["mode"] not in PARITY_MODES:
                errors.append(f"{key}: parity mode {row['parity']['mode']!r}")
            if bool(row.get("adapter")) == bool(row.get("adapter_spec")):
                errors.append(f"{key}: give exactly one of adapter / adapter_spec")
            if row.get("image") not in table["images"]:
                errors.append(f"{key}: unknown image {row.get('image')!r}")
    if errors:
        for line in errors:
            print(f"plan error: {line}", file=sys.stderr)
        return 2
    result = {
        "schema": SCHEMA,
        "created_utc": utc(),
        "table_sha256": sha_file(args.table),
        "event": table["event"],
        "events_total": table["events_total"],
        "site": args.site,
        "c27": c27,
        "images": table["images"],
        "selection": selection,
        "models": models,
        "pairs": build_pairs(models, table.get("reference_pairs", [])),
        "deviations": deviations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "models": selection,
                "pairs": len(result["pairs"]),
                "deviations": [d["id"] for d in deviations],
            }
        )
    )
    return 0


def load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("not an event-3 plan")
    return value


def collected(plan_: dict[str, Any]) -> list[str]:
    return [k for k in plan_["selection"] if plan_["models"][k].get("stored") is None]


def show(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    print(
        f"event {p['event']} of {p['events_total']}; site {p['site']}; 27B candidate {p['c27']}"
    )
    for key in p["selection"]:
        row = p["models"][key]
        where = (
            "stored " + row["stored"]["predictions"]
            if row.get("stored")
            else (
                f"{row.get('adapter') or row.get('adapter_spec')} image={row['image']}"
                f" parity={row['parity']['mode']} cache={'frozen' if row.get('cache') else 'none'}"
            )
        )
        print(f"{key:12s} {row['tier']:5s} {row['role']:9s} {row['label']} | {where}")
    for pair in p["pairs"]:
        print(f"pair ({pair['kind']}): {pair['left']} vs {pair['right']}")
    for dev in p["deviations"]:
        print(f"deviation {dev['id']} ({dev['key']}): approved by {dev['approved']}")
    return 0


# ---------------------------------------------------------------------- argv


def runner_argv(
    row: dict[str, Any],
    images: dict[str, str],
    phase: str,
    run_dir: str,
    gpu: str,
    src: str,
    src_root: str,
    lease_name: str,
    shared: bool,
    cache_dir: str | None,
) -> list[str]:
    if row.get("stored") is not None:
        raise ValueError("stored rows are not collected")
    if bool(row.get("cache")) != bool(cache_dir):
        raise ValueError("a frozen-cache row needs --cache-dir (and only such a row)")
    purpose = f"C1 event 3 {'preflight smoke' if phase == 'smoke' else 'collection'}: {row['label']}"
    argv = ["--gpu", gpu, "--track", "eval", "--lease-name", lease_name]
    if shared:
        argv.append("--shared")
    argv += [
        "--src",
        src,
        "--run-dir",
        run_dir,
        "--purpose",
        purpose,
        "--model-dir",
        row.get("model_dir") or row["model_path"],
        "--image",
        images[row["image"]],
    ]
    for mount in row.get("mounts", []):
        argv += ["--mount", mount]
    for env in row.get("env", []):
        argv += ["--env", env]
    if cache_dir:
        argv += [
            "--env",
            "TRITON_CACHE_AUTOTUNING=1",
            "--env",
            f"TRITON_CACHE_DIR={cache_dir}",
            "--mount-rw",
            cache_dir,
        ]
    argv.append("--")
    if row.get("adapter"):
        argv += ["--adapter", row["adapter"]]
    else:
        argv += ["--adapter-spec", f"{src_root}/{row['adapter_spec']}"]
    argv += ["--model-path", row["model_path"], "--revision", row["revision"]]
    for k, v in row.get("extra", {}).items():
        argv += ["--extra", f"{k}={v}"]
    if phase == "smoke":
        argv += ["--panels", ",".join(SMOKE_PANELS)]
        if row.get("smoke_items"):
            argv += ["--max-items", str(row["smoke_items"])]
    elif phase == "collect":
        argv += ["--panels", C1_PANEL]
    else:
        raise ValueError(f"unknown phase {phase}")
    return argv


def argv_cmd(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    row = p["models"][args.key]
    out = runner_argv(
        row,
        p["images"],
        args.phase,
        str(args.run_dir),
        str(args.gpu),
        args.src,
        str(args.src_root),
        args.lease_name,
        args.shared,
        args.cache_dir,
    )
    sys.stdout.write("\0".join(out) + "\0")
    return 0


# -------------------------------------------------------------------- verify


def adapter_module(src_root: Path, row: dict[str, Any]) -> Path:
    sys.path.insert(0, str(src_root))
    from v2.eval import adapters as registry

    if row.get("adapter"):
        adapter = registry.load(row["adapter"], None)
    else:
        adapter = registry.load(None, src_root / row["adapter_spec"])
    return registry.module_path(src_root, adapter)


def image_present(image_id: str) -> bool:
    try:
        out = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image_id],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and out.stdout.strip() == image_id


def check_package(package: dict[str, Any], workers: int) -> dict[str, Any]:
    root = Path(package["dir"])
    result: dict[str, Any] = {"dir": str(root)}
    manifest_path = root / "MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        return {
            **result,
            "ok": False,
            "problems": [f"no MODEL_MANIFEST.json in {root}"],
        }
    result["manifest_sha256"] = sha_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files_sha256") or {}
    names = sorted(files)
    present = [n for n in names if (root / n).is_file()]
    actual = dict(zip(present, hash_many([root / n for n in present], workers)))
    bad = [n for n in names if actual.get(n) != files[n]]
    identity = (manifest.get("identity") or {}).get("model_sha256")
    result.update(files=len(names), mismatched=bad, identity=identity)
    problems = []
    if result["manifest_sha256"] != package["manifest_sha256"]:
        problems.append("manifest hash differs")
    if not names:
        problems.append("manifest pins no package files")
    if bad:
        problems.append(f"{len(bad)} package files differ from the manifest")
    if package.get("identity") and identity != package["identity"]:
        problems.append("manifest identity differs from the scored weights")
    base = package.get("base")
    if base:
        base_files = (manifest.get("base") or {}).get("files_sha256") or {}
        broot = Path(base)
        bnames = sorted(base_files)
        bpresent = [n for n in bnames if (broot / n).is_file()]
        bactual = dict(zip(bpresent, hash_many([broot / n for n in bpresent], workers)))
        bbad = [n for n in bnames if bactual.get(n) != base_files[n]]
        result.update(base=str(broot), base_files=len(bnames), base_mismatched=bbad)
        if not bnames:
            problems.append("manifest pins no base files")
        if bbad:
            problems.append(f"{len(bbad)} base files differ from the manifest")
    result["ok"] = not problems
    result["problems"] = problems
    return result


def cache_digest(src_root: Path, frozen: Path) -> str:
    """Tree hash of a frozen autotune cache by ``v2/27b/triton_cache.py`` (as event3.sh copies)."""
    sys.path.insert(0, str(src_root / "v2" / "27b"))
    import triton_cache

    return triton_cache.tree_digest(triton_cache.file_hashes(frozen))


def verify_row(row: dict[str, Any], src_root: Path, workers: int) -> dict[str, Any]:
    out: dict[str, Any] = {"problems": []}
    problems = out["problems"]
    stored = row.get("stored")
    if stored is not None:
        # Existence only: stored C1 predictions and seals are hashed in the event phase.
        for field in ("predictions", "seal"):
            if not os.path.isfile(stored[field]):
                problems.append(f"stored {field} missing")
        return out
    paths = [row["model_path"], row.get("model_dir") or row["model_path"]]
    paths += row.get("mounts", [])
    paths += [
        v
        for v in row.get("extra", {}).values()
        if isinstance(v, str) and v.startswith("/")
    ]
    missing = sorted({x for x in paths if not os.path.exists(x)})
    if missing:
        problems.append(f"missing paths: {', '.join(missing)}")
    module = adapter_module(src_root, row)
    out["adapter_module"] = str(module.relative_to(src_root))
    if module.is_file():
        out["adapter_module_sha256"] = sha_file(module)
    else:
        problems.append("adapter module missing in the mirror")
    stored_collect = row.get("stored_collect")
    if stored_collect and os.path.isfile(stored_collect):
        receipt = json.loads(Path(stored_collect).read_text(encoding="utf-8"))
        out["adapter_module_same_as_stored_run"] = receipt.get(
            "adapter_module_sha256"
        ) == out.get("adapter_module_sha256")
    parity = row["parity"]
    if parity["mode"] != "none":
        for panel in SMOKE_PANELS:
            if not stored_predictions(row, panel).is_file():
                problems.append(f"stored {panel} predictions for parity missing")
    if row.get("package"):
        out["package"] = check_package(row["package"], workers)
        problems += out["package"]["problems"]
    for tree in row.get("trees", []):
        if not os.path.isdir(tree["dir"]):
            problems.append(f"tree missing: {tree['dir']}")
            continue
        digest, count = tree_digest(Path(tree["dir"]), workers)
        out.setdefault("trees", []).append(
            {"dir": tree["dir"], "sha256": digest, "files": count}
        )
        if digest != tree["sha256"]:
            problems.append(f"tree digest differs: {tree['dir']}")
    for pinned in row.get("files", []):
        if not os.path.isfile(pinned["path"]):
            problems.append(f"file missing: {pinned['path']}")
        elif sha_file(Path(pinned["path"])) != pinned["sha256"]:
            problems.append(f"file hash differs: {pinned['path']}")
    cache = row.get("cache")
    if cache:
        if not os.path.isdir(cache["frozen"]):
            problems.append(f"frozen autotune cache missing: {cache['frozen']}")
        else:
            digest = cache_digest(src_root, Path(cache["frozen"]))
            out["cache_sha256"] = digest
            if digest != cache["sha256"]:
                problems.append("frozen autotune cache digest differs")
    return out


def verify(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    src_root = Path(args.src_root)
    report: dict[str, Any] = {
        "schema": SCHEMA + "/verify",
        "utc": utc(),
        "site": p["site"],
        "images": {},
        "models": {},
    }
    failures = 0
    for name, image_id in p["images"].items():
        used = any(p["models"][k].get("image") == name for k in collected(p))
        if used:
            ok = image_present(image_id)
            report["images"][name] = {"id": image_id, "present": ok}
            failures += not ok
    for key in p["selection"]:
        try:
            out = verify_row(p["models"][key], src_root, args.hash_workers)
        except (OSError, ValueError, KeyError) as exc:
            out = {"problems": [f"verification error: {exc!r}"]}
        out["ok"] = not out["problems"]
        failures += not out["ok"]
        report["models"][key] = out
    report["passed"] = failures == 0
    write_new(args.output, report)
    for key, out in report["models"].items():
        print(
            f"verify {key}: {'ok' if out['ok'] else 'FAILED: ' + '; '.join(out['problems'])}"
        )
    return 0 if report["passed"] else 1


# -------------------------------------------------------------------- parity


def point(answer: Any) -> Any:
    """Gold-free categorical answer, as ``benchmark.score.evaluate_answer`` reads it."""
    if not isinstance(answer, dict):
        return None
    kind = answer.get("type")
    if kind == "choice":
        return answer.get("choice")
    if kind == "noul":
        p = answer.get("noul")
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            return None
        return None if p == 0.5 else p > 0.5
    if kind == "score":
        s = answer.get("score")
        if (
            not isinstance(s, (int, float))
            or isinstance(s, bool)
            or not math.isfinite(s)
        ):
            return None
        nearest = round(s)
        return nearest if abs(s - nearest) < 0.5 else None
    return answer.get("value")


def drift(a: Any, b: Any) -> float:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0 if a == b else 1.0
    worst = 0.0
    for key in NUMERIC_KEYS:
        x, y = a.get(key), b.get(key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            worst = max(worst, abs(float(x) - float(y)))
        elif (x is None) != (y is None):
            worst = max(worst, 1.0)
    pa, pb = a.get("probabilities") or {}, b.get("probabilities") or {}
    for label in set(pa) | set(pb):
        x, y = pa.get(label), pb.get(label)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            worst = max(worst, abs(float(x) - float(y)))
        else:
            worst = max(worst, 1.0)
    return worst


def smoke_predictions(run_dir: Path, panel: str) -> Path | None:
    for sub in ("smoke", "output"):
        path = run_dir / sub / f"{panel}.predictions.jsonl"
        if path.is_file():
            return path
    return None


def stored_predictions(row: dict[str, Any], panel: str) -> Path:
    """The stored run's predictions for a smoke panel (named like its typed-final file)."""
    path = Path(row["parity"]["stored"])
    return path.with_name(path.name.replace(SMOKE_PANELS[0], panel))


def compare_smoke(
    smoke: list[dict[str, Any]], stored: dict[str, dict[str, Any]], row: dict[str, Any]
) -> dict[str, Any]:
    answers = changed = missing = 0
    worst = 0.0
    identity_bad = 0
    by_type: dict[str, int] = {}
    expect = row.get("identity")
    for pred in smoke:
        if expect and pred.get("model_sha256") != expect:
            identity_bad += 1
        ref = stored.get(pred.get("id"))
        if ref is None:
            missing += 1
            continue
        mine, theirs = pred.get("answers") or {}, ref.get("answers") or {}
        for qid in set(mine) | set(theirs):
            answers += 1
            kind = (theirs.get(qid) or mine.get(qid) or {}).get("type", "invalid")
            by_type[kind] = by_type.get(kind, 0) + 1
            if point(mine.get(qid)) != point(theirs.get(qid)):
                changed += 1
            worst = max(worst, drift(mine.get(qid), theirs.get(qid)))
    parity = row["parity"]
    mode = parity["mode"]
    problems = []
    if not smoke:
        problems.append("the smoke produced no predictions")
    if expect and identity_bad:
        problems.append(f"{identity_bad} rows lack the expected model_sha256")
    if mode == "exact":
        if missing or changed or worst > parity.get("tolerance", 1e-4):
            problems.append("not identical to the stored formal run")
    elif mode == "near":
        allowed = parity.get("max_changed_fraction", 0.1) * max(answers, 1)
        if missing or changed > allowed:
            problems.append("too many answers differ from the stored formal run")
    return {
        "mode": mode,
        "prompts": len(smoke),
        "missing_in_stored": missing,
        "answers": answers,
        "answers_by_type": by_type,
        "changed": changed,
        "max_probability_drift": worst,
        "identity_expected": expect,
        "identity_mismatches": identity_bad,
        "problems": problems,
        "passed": not problems,
    }


def parity_cmd(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    row = p["models"][args.key]
    panels = {}
    for panel in SMOKE_PANELS:
        path = smoke_predictions(args.run_dir, panel)
        smoke = read_jsonl(path) if path else []
        stored: dict[str, dict[str, Any]] = {}
        if row["parity"]["mode"] != "none":
            stored = {r["id"]: r for r in read_jsonl(stored_predictions(row, panel))}
        panels[panel] = compare_smoke(smoke, stored, row)
        panels[panel].update(
            preflight=str(path) if path else None,
            stored_run=str(stored_predictions(row, panel)),
        )
    result = {
        "schema": SCHEMA + "/parity",
        "key": args.key,
        "mode": row["parity"]["mode"],
        "panels": panels,
        "passed": all(v["passed"] for v in panels.values()),
    }
    write_new(args.output, result)
    for panel, v in panels.items():
        print(
            json.dumps(
                {"key": args.key, "panel": panel}
                | {
                    k: v[k]
                    for k in ("mode", "prompts", "answers_by_type", "changed")
                    + ("max_probability_drift", "passed")
                }
            )
        )
    return 0 if result["passed"] else 1


# -------------------------------------------------------------------- stored


def stored_cmd(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    report: dict[str, Any] = {"schema": SCHEMA + "/stored", "utc": utc(), "models": {}}
    ok = True
    for key in p["selection"]:
        stored = p["models"][key].get("stored")
        if stored is None:
            continue
        seal_sha = sha_file(Path(stored["seal"]))
        seal = json.loads(Path(stored["seal"]).read_text(encoding="utf-8"))
        pred_sha = sha_file(Path(stored["predictions"]))
        problems = []
        if seal_sha != stored["seal_sha256"]:
            problems.append("seal file differs from the recorded seal")
        if seal.get("prompts_sha256") != PROMPTS_SHA256:
            problems.append("seal is for other prompts")
        if seal.get("predictions_sha256") != pred_sha:
            problems.append("predictions changed after the seal")
        if seal.get("missing") != 0:
            problems.append("seal records missing predictions")
        report["models"][key] = {
            "seal_sha256": seal_sha,
            "predictions_sha256": pred_sha,
            "problems": problems,
            "ok": not problems,
        }
        ok &= not problems
    report["passed"] = ok
    write_new(args.output, report)
    for key, out in report["models"].items():
        print(
            f"stored {key}: {'ok' if out['ok'] else 'FAILED: ' + '; '.join(out['problems'])}"
        )
    return 0 if ok else 1


# --------------------------------------------------------------- pairs, summary


def prediction_file(p: dict[str, Any], key: str, event_dir: Path) -> str:
    stored = p["models"][key].get("stored")
    if stored is not None:
        return stored["predictions"]
    return str(event_dir / key / "output" / f"{C1_PANEL}.predictions.jsonl")


def pairs_cmd(args: argparse.Namespace) -> int:
    """Pairs whose sides are both sealed: left, right, left name, right name, output (NUL-separated)."""
    p = load_plan(args.plan)
    fields = []
    for pair in p["pairs"]:
        left, right = pair["left"], pair["right"]
        if any(
            p["models"][k].get("stored") is None
            and not (args.event_dir / k / "SEAL-C1.json").is_file()
            for k in (left, right)
        ):
            print(f"skipped {left} vs {right}: a side is not sealed", file=sys.stderr)
            continue
        fields += [
            prediction_file(p, left, args.event_dir),
            prediction_file(p, right, args.event_dir),
            p["models"][left]["label"],
            p["models"][right]["label"],
            str(args.event_dir / f"PAIRED-C1-{left}-vs-{right}.json"),
        ]
    sys.stdout.write("".join(f + "\0" for f in fields))
    return 0


def gpu_hours(root: Path) -> dict[str, Any]:
    runs = {}
    for record in sorted(root.glob("*/GPU-TIME.json")):
        value = json.loads(record.read_text(encoding="utf-8"))
        runs[record.parent.name] = {
            k: value.get(k)
            for k in ("gpu", "wall_seconds", "gpu_hours", "exit_code", "shared")
        }
    return {"runs": runs, "gpu_hours": sum(r["gpu_hours"] or 0 for r in runs.values())}


def summary_cmd(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    models = {}
    for key in p["selection"]:
        row = p["models"][key]
        report = args.event_dir / key / "REPORT-C1.json"
        if row.get("stored") and not report.is_file():
            report = Path(row["stored"]["report"])
        if report.is_file():
            value = json.loads(report.read_text(encoding="utf-8"))
            models[key] = {
                "label": row["label"],
                "tier": row["tier"],
                "role": row["role"],
                "stored": row.get("stored") is not None,
                "c1": value["c1"],
                "by_type": value["by_type"],
                "valid": value["valid"],
                "items": value["items"],
                "item_set": value.get("item_set"),
                "long_input": value["slices"]["long_input"]["accuracy"],
                "non_english": value["slices"]["non_english"]["accuracy"],
                "report_sha256": sha_file(report),
            }
    paired = {}
    for pair in p["pairs"]:
        path = args.event_dir / f"PAIRED-C1-{pair['left']}-vs-{pair['right']}.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            paired[f"{pair['left']} vs {pair['right']}"] = {
                "kind": pair["kind"],
                "delta": value["delta"],
                "ci95": value["ci95"],
                "by_type": value["by_type"],
                "item_set": value.get("item_set"),
                "sha256": sha_file(path),
            }
    smokes, runs = gpu_hours(args.preflight_dir), gpu_hours(args.event_dir)
    result = {
        "schema": SCHEMA + "/summary",
        "utc": utc(),
        "event": p["event"],
        "models": models,
        "paired": paired,
        "gpu": {
            "preflight": smokes,
            "collection": runs,
            "gpu_hours": smokes["gpu_hours"] + runs["gpu_hours"],
        },
    }
    write_new(args.output, result)
    for key, m in models.items():
        print(f"{key}: C1 {m['c1']:.2f} (valid {m['valid']})")
    for name, d in paired.items():
        print(f"{name}: {d['delta']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]")
    print(f"GPU-hours {result['gpu']['gpu_hours']:.3f}")
    return 0


def keys_cmd(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    if args.collected:
        keys = collected(p)
    elif args.stored:
        keys = [k for k in p["selection"] if p["models"][k].get("stored") is not None]
    else:
        keys = p["selection"]
    sys.stdout.write("".join(k + "\0" for k in keys))
    return 0


def field_cmd(args: argparse.Namespace) -> int:
    value: Any = load_plan(args.plan)["models"][args.key]
    for part in args.field.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value is None:
        return 0
    print(
        value
        if isinstance(value, (str, int, float))
        else json.dumps(value, sort_keys=True)
    )
    return 0


def stage_cmd(args: argparse.Namespace) -> int:
    """Node-A staging steps of the selected rows: kind, key, source, destination, repo, revision,
    include (an ``hf download --include`` pattern, for a package in a folder of a staging repo).
    """
    p = load_plan(args.plan)
    fields = []
    for key in collected(p):
        for step in p["models"][key].get("stage_node_a", []):
            fields += [
                step["kind"],
                key,
                step.get("from", ""),
                step["to"],
                step.get("repo", ""),
                step.get("revision", ""),
                step.get("include", ""),
            ]
    sys.stdout.write("".join(f + "\0" for f in fields))
    return 0


def digest_cmd(args: argparse.Namespace) -> int:
    digest, count = tree_digest(args.dir, args.hash_workers)
    print(f"{digest} files={count} {args.dir}")
    return 0


# ------------------------------------------------------- ~27B candidate receipt


def fingerprint_check(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """The identity recomputed from the manifest's fingerprint files, and those not pinned by it."""
    files = (manifest.get("identity") or {}).get("fingerprint_files")
    if not isinstance(files, dict):
        return None
    sys.path.insert(0, str(SRC_ROOT))
    from training.model.data import canonical

    base = (manifest.get("base") or {}).get("files_sha256") or {}
    pinned = {
        f"checkpoint/{k}": v for k, v in (manifest.get("files_sha256") or {}).items()
    }
    pinned.update({f"source/{k}": v for k, v in base.items()})
    return {
        "sha256": hashlib.sha256(canonical(files).encode("utf-8")).hexdigest(),
        "files": len(files),
        "unpinned": sorted(k for k, v in files.items() if pinned.get(k) != v),
    }


def runtime_identity(root: Path, base: Path) -> str:
    """The identity ``training.model.infer`` (the kernel collector's runtime) computes on disk."""
    sys.path.insert(0, str(SRC_ROOT))
    from training.model import infer as runtime

    return runtime.checkpoint_fingerprint(root, base)["model_sha256"]


def rows_check(
    path: Path, identity: str | None, calibration_sha: str | None
) -> dict[str, Any]:
    rows = read_jsonl(path)
    out: dict[str, Any] = {
        "sha256": sha_file(path),
        "rows": len(rows),
        "model_sha256_mismatches": sum(r.get("model_sha256") != identity for r in rows),
    }
    if calibration_sha is not None:
        out["calibration_sha256_mismatches"] = sum(
            r.get("calibration_sha256") != calibration_sha for r in rows
        )
    return out


def c27_receipt(args: argparse.Namespace) -> int:
    """What the table pins for a ~27B candidate, computed from its package and formal run (CPU)."""
    run: Path | None = args.formal_run
    paths: dict[str, Path | None] = {
        "typed_final": args.typed_final
        or (run and run / "output" / f"{SMOKE_PANELS[0]}.predictions.jsonl"),
        "public231": args.public231
        or (run and run / "output" / f"{SMOKE_PANELS[1]}.predictions.jsonl"),
        "collect": args.collect or (run and run / "COLLECT.json"),
        "seal": args.seal or (run and run / "SEAL.json"),
        "report": args.report or (run and run / "REPORT.json"),
    }
    if not (paths["typed_final"] and paths["public231"] and paths["collect"]):
        raise ValueError(
            "give --formal-run DIR, or --typed-final, --public231 and --collect"
        )
    problems: list[str] = []
    root: Path = args.package_dir
    manifest_path = root / "MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(f"no MODEL_MANIFEST.json in {root}")
    manifest_sha = sha_file(manifest_path)
    package = check_package(
        {"dir": str(root), "manifest_sha256": manifest_sha}, args.hash_workers
    )
    problems += package["problems"]
    identity = package["identity"]
    if not identity:
        problems.append("the manifest records no weights identity")
    fingerprint = fingerprint_check(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if fingerprint is not None:
        if fingerprint["sha256"] != identity:
            problems.append(
                "the identity is not the hash of the manifest's fingerprint files"
            )
        if fingerprint["unpinned"]:
            problems.append(
                f"{len(fingerprint['unpinned'])} fingerprint files are not the pinned files"
            )
    runtime = None
    if args.base:
        try:
            runtime = runtime_identity(root, args.base)
        except ValueError as exc:
            problems.append(f"the runtime refuses this package and base: {exc}")
        if runtime is not None and runtime != identity:
            problems.append(
                "the runtime computes another identity on this package and base"
            )
    calibration = None
    if args.calibration:
        value = json.loads(args.calibration.read_text(encoding="utf-8"))
        calibration = {
            "sha256": sha_file(args.calibration),
            "model_sha256": (
                value.get("model_sha256") if isinstance(value, dict) else None
            ),
        }
        if calibration["model_sha256"] != identity:
            problems.append("the calibration file is bound to other weights")
    seal = None
    if paths["seal"]:
        if paths["seal"].is_file():
            seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
        else:
            problems.append(f"no {paths['seal']}")
    predictions = {}
    for panel, key in zip(SMOKE_PANELS, ("typed_final", "public231")):
        out = rows_check(paths[key], identity, calibration and calibration["sha256"])
        if seal is not None:
            sealed = ((seal.get("panels") or {}).get(panel) or {}).get(
                "predictions_sha256"
            )
            out["sealed"] = sealed == out["sha256"]
            if not out["sealed"]:
                problems.append(f"{panel} predictions are not the ones in SEAL.json")
        if out["model_sha256_mismatches"]:
            problems.append(
                f"{panel}: {out['model_sha256_mismatches']} rows lack the identity"
            )
        if out.get("calibration_sha256_mismatches"):
            problems.append(
                f"{panel}: rows were not predicted with this calibration file"
            )
        predictions[panel] = out
    value = json.loads(paths["collect"].read_text(encoding="utf-8"))
    collect = {
        "sha256": sha_file(paths["collect"]),
        "adapter_module_sha256": value.get("adapter_module_sha256"),
        "image_id": value.get("image_id"),
    }
    v3 = report_sha = None
    if paths["report"]:
        if paths["report"].is_file():
            report_sha = sha_file(paths["report"])
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            v3 = (report.get("v3") or {}).get("score")
            if v3 is None:
                problems.append("REPORT.json has no v3")
            if seal is not None and report.get("seal_sha256") != sha_file(
                paths["seal"]
            ):
                problems.append("REPORT.json is not the report of this SEAL.json")
        else:
            problems.append(f"no {paths['report']}")
    cache = None
    if args.cache:
        cache = {"sha256": cache_digest(SRC_ROOT, args.cache)}
        post = args.cache.with_name(args.cache.name + ".post.json")
        if post.is_file():
            cache["post_sha256"] = json.loads(post.read_text(encoding="utf-8")).get(
                "post_sha256"
            )
            if cache["post_sha256"] != cache["sha256"]:
                problems.append(
                    "the cache changed after the formal run (its post.json)"
                )
    inputs = {
        "package_dir": root,
        **paths,
        "calibration": args.calibration,
        "cache": args.cache,
        "base": args.base,
    }
    receipt = {
        "schema": SCHEMA + "/c27-receipt",
        "utc": utc(),
        "inputs": {k: os.path.abspath(v) if v else None for k, v in inputs.items()},
        "manifest_sha256": manifest_sha,
        "identity": identity,
        "package": {
            "files": package["files"],
            "mismatched": package["mismatched"],
            "fingerprint": fingerprint,
            "runtime_identity": runtime,
        },
        "calibration": calibration,
        "predictions": predictions,
        "collect": collect,
        "seal_checked": seal is not None,
        "seal_sha256": sha_file(paths["seal"]) if seal is not None else None,
        "report_sha256": report_sha,
        "v3": v3,
        "cache": cache,
        "problems": problems,
        "passed": not problems,
    }
    write_new(args.output, receipt)
    print(
        json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "identity": identity,
                "calibration": calibration and calibration["sha256"],
                **{p: predictions[p]["sha256"] for p in SMOKE_PANELS},
                "collect": collect["sha256"],
                "cache": cache and cache["sha256"],
                "v3": v3,
                "passed": receipt["passed"],
            }
        )
    )
    for line in problems:
        print(f"c27-receipt problem: {line}", file=sys.stderr)
    return 0 if receipt["passed"] else 1


def template(path: str, roots: dict[str, str]) -> str:
    """``path`` with its longest table root written as ``${ROOT}``."""
    under = [
        name
        for name, root in roots.items()
        if path == root or path.startswith(root.rstrip("/") + "/")
    ]
    if not under:
        return path
    name = max(under, key=lambda n: len(roots[n]))
    return "${" + name + "}" + path[len(roots[name]) :]


def layout(value: Any, indent: int = 0) -> str:
    """JSON as the table writes it: containers of objects one entry per line, the rest inline."""
    pad = " " * (indent + 2)

    def holds_objects(v: Any) -> bool:
        return isinstance(v, dict) or (
            isinstance(v, list) and any(isinstance(x, dict) for x in v)
        )

    if isinstance(value, dict) and any(holds_objects(v) for v in value.values()):
        body = ",\n".join(
            f"{pad}{json.dumps(k)}: {layout(v, indent + 2)}" for k, v in value.items()
        )
        return "{\n" + body + "\n" + " " * indent + "}"
    if isinstance(value, list) and any(isinstance(x, dict) for x in value):
        body = ",\n".join(pad + layout(x, indent + 2) for x in value)
        return "[\n" + body + "\n" + " " * indent + "]"
    return json.dumps(value, ensure_ascii=False)


def splice_block(text: str, key: str, block: dict[str, Any]) -> str:
    """The table text with one ``candidates_27b`` block rewritten; the rest byte-identical."""
    start = text.index('"candidates_27b"')
    match = re.compile(r'\n( *)"' + re.escape(key) + r'": (\{)').search(text, start)
    if match is None:
        raise ValueError(f"no candidates_27b block {key!r} in the table text")
    _, end = json.JSONDecoder().raw_decode(text, match.start(2))
    return text[: match.start(2)] + layout(block, len(match.group(1))) + text[end:]


def fill_c27(args: argparse.Namespace) -> int:
    """Write a passed ``c27-receipt`` into a ``candidates_27b`` block of the model table."""
    text = args.table.read_text(encoding="utf-8")
    table = load_table(args.table)
    roots = table["roots"]
    if args.key not in table.get("candidates_27b", {}):
        raise ValueError(
            f"--key must be one of {sorted(table.get('candidates_27b', {}))}"
        )
    block = json.loads(json.dumps(table["candidates_27b"][args.key]))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA + "/c27-receipt":
        raise ValueError("not a c27-receipt")
    if not receipt.get("passed"):
        raise ValueError(
            "the receipt did not pass: " + "; ".join(receipt.get("problems", []))
        )
    if not (receipt.get("seal_checked") and receipt.get("v3") is not None):
        raise ValueError("the receipt needs the formal run's SEAL.json and REPORT.json")
    if not (receipt.get("calibration") and receipt.get("cache")):
        raise ValueError("the receipt needs --calibration and --cache")
    block["v3"] = round(receipt["v3"], 3)
    block["identity"] = block["package"]["identity"] = receipt["identity"]
    block["package"]["manifest_sha256"] = receipt["manifest_sha256"]
    block["cache"]["sha256"] = receipt["cache"]["sha256"]
    if args.repo:
        block["repo"] = args.repo
    if args.revision:
        block["revision"] = args.revision
    hf = [s for s in block.get("stage_node_a", []) if s["kind"] == "hf"]
    if len(hf) != 1:
        raise ValueError(f"{args.key}: the package must come from one hf staging step")
    hf[0].update(repo=block["repo"], revision=block["revision"])
    if args.include:
        folder = re.fullmatch(r"([^*?\[\]]+?)/\*\*?", args.include)
        if folder is None:
            raise ValueError(
                "--include must be SUBDIR/* (the package's folder in the repo)"
            )
        hf[0]["include"] = args.include
        block["model_path"] = block["package"]["dir"] = (
            f"{hf[0]['to']}/{folder.group(1)}"
        )
    sha = {
        "calibration": receipt["calibration"]["sha256"],
        "typed-final": receipt["predictions"]["typed-final"]["sha256"],
        "public231": receipt["predictions"]["public231"]["sha256"],
        "collect": receipt["collect"]["sha256"],
    }
    here = {
        "calibration": block["extra"]["calibration"],
        "typed-final": block["parity"]["stored"],
        "public231": str(stored_predictions(block, "public231")),
        "collect": block["stored_collect"],
    }
    block["files"] = [{"path": here[r], "sha256": sha[r]} for r in RECEIPT_ROLES]
    hashed = receipt["inputs"]
    wanted = {
        "package": hashed["package_dir"],
        "calibration": hashed["calibration"],
        "cache": hashed["cache"],
        "typed-final": hashed["typed_final"],
        "public231": hashed["public231"],
        "collect": hashed["collect"],
    }
    # Node-B sources: the stage steps' "from" and the node_b overrides must be the files hashed.
    node_b = block.setdefault("node_b", {})
    copies = {s["to"]: s for s in block["stage_node_a"] if s["kind"] == "copy"}
    refs: list[tuple[str, dict[str, Any], str]] = [
        ("package", node_b, "model_path"),
        ("package", node_b, "package.dir"),
        ("calibration", node_b, "extra.calibration"),
        ("cache", node_b, "cache.frozen"),
        ("typed-final", node_b, "parity.stored"),
        ("collect", node_b, "stored_collect"),
    ]
    for role, dest in (*here.items(), ("cache", block["cache"]["frozen"])):
        if dest not in copies:
            raise ValueError(f"{args.key}: no copy step stages {dest}")
        refs.append((role, copies[dest], "from"))
    mismatches = []
    for role, holder, field in refs:
        current = holder.get(field)
        filled = isinstance(current, str) and PLACEHOLDER not in current
        if filled and not args.adopt_paths:
            if expand(current, roots) != wanted[role]:
                mismatches.append(
                    f"{role}: the table has {current}, the receipt hashed {wanted[role]}"
                )
            continue
        holder[field] = template(wanted[role], roots)
    public = str(
        stored_predictions({"parity": {"stored": node_b["parity.stored"]}}, "public231")
    )
    if expand(public, roots) != wanted["public231"]:
        mismatches.append(
            f"public231: node-B parity reads {public}, the receipt hashed {wanted['public231']}"
        )
    if mismatches:
        raise ValueError(
            "node-B paths differ from the receipt (run c27-receipt on the table's paths, or pass"
            " --adopt-paths): " + "; ".join(mismatches)
        )
    there = {
        "calibration": node_b["extra.calibration"],
        "typed-final": node_b["parity.stored"],
        "public231": public,
        "collect": node_b["stored_collect"],
    }
    node_b["files"] = [{"path": there[r], "sha256": sha[r]} for r in RECEIPT_ROLES]
    staged = [expand(s["to"], roots) for s in block["stage_node_a"]]
    node_b["mounts"] = [
        m
        for m in block.get("mounts", [])
        if not any(t.startswith(expand(m, roots).rstrip("/") + "/") for t in staged)
    ] + [template(os.path.dirname(wanted["calibration"]), roots)]
    receipt_sha = sha_file(args.receipt)
    block["notes"] = (
        f"{block.get('notes', '')} Values from c27-receipt {receipt_sha[:12]}"
        f" ({receipt['utc']}).".strip()
    )
    left = placeholders(block) + placeholders(node_b, "node_b")
    if left:
        raise ValueError(f"{args.key}: PARENT-FILLS remains in {', '.join(left)}")
    new_text = splice_block(text, args.key, block)
    expected = json.loads(text)
    expected["candidates_27b"][args.key] = block
    if json.loads(new_text) != expected:
        raise ValueError("the rewritten table does not parse back to the filled table")
    out = args.output or args.table
    pending = out.with_name(out.name + ".pending")
    pending.write_text(new_text, encoding="utf-8")
    os.replace(pending, out)
    print(
        json.dumps(
            {
                "key": args.key,
                "v3": block["v3"],
                "identity": block["identity"],
                "manifest_sha256": block["package"]["manifest_sha256"],
                "revision": block["revision"],
                "table": str(out),
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--table", type=Path, required=True)
    a.add_argument("--models")
    a.add_argument("--c27", choices=("f1", "f2"))
    a.add_argument("--peers27")
    for flag in ("--c27-package", "--c27-manifest", "--c27-repo", "--c27-revision"):
        a.add_argument(flag)
    for flag in (
        "--c9b-package",
        "--c9b-manifest",
        "--c9b-repo",
        "--c9b-revision",
        "--c9b-identity",
        "--c9b-calibration",
        "--c9b-parity-stored",
        "--c9b-cache",
        "--c9b-cache-sha",
    ):
        a.add_argument(flag)
    a.add_argument("--c9b-tolerance", type=float)
    a.add_argument("--site", choices=("node-a", "node-b"), default="node-a")
    a.add_argument("--allow-deviation", action="append", default=[])
    a.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("show")
    b.add_argument("--plan", type=Path, required=True)
    c = sub.add_parser("argv")
    c.add_argument("--plan", type=Path, required=True)
    c.add_argument("--key", required=True)
    c.add_argument("--phase", choices=("smoke", "collect"), required=True)
    c.add_argument("--run-dir", type=Path, required=True)
    c.add_argument("--gpu", required=True)
    c.add_argument("--src", required=True)
    c.add_argument("--src-root", type=Path, required=True)
    c.add_argument("--lease-name", required=True)
    c.add_argument("--shared", action="store_true")
    c.add_argument("--cache-dir")
    d = sub.add_parser("verify")
    d.add_argument("--plan", type=Path, required=True)
    d.add_argument("--src-root", type=Path, required=True)
    d.add_argument("--hash-workers", type=int, default=8)
    d.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("parity")
    e.add_argument("--plan", type=Path, required=True)
    e.add_argument("--key", required=True)
    e.add_argument("--run-dir", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    f = sub.add_parser("stored")
    f.add_argument("--plan", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    g = sub.add_parser("pairs")
    g.add_argument("--plan", type=Path, required=True)
    g.add_argument("--event-dir", type=Path, required=True)
    h = sub.add_parser("summary")
    h.add_argument("--plan", type=Path, required=True)
    h.add_argument("--event-dir", type=Path, required=True)
    h.add_argument("--preflight-dir", type=Path, required=True)
    h.add_argument("--output", type=Path, required=True)
    i = sub.add_parser("digest")
    i.add_argument("dir", type=Path)
    i.add_argument("--hash-workers", type=int, default=8)
    j = sub.add_parser("stage")
    j.add_argument("--plan", type=Path, required=True)
    k = sub.add_parser("keys")
    k.add_argument("--plan", type=Path, required=True)
    only = k.add_mutually_exclusive_group()
    only.add_argument("--collected", action="store_true")
    only.add_argument("--stored", action="store_true")
    m = sub.add_parser("field")
    m.add_argument("--plan", type=Path, required=True)
    m.add_argument("--key", required=True)
    m.add_argument("--field", required=True)
    n = sub.add_parser("c27-receipt")
    n.add_argument("--package-dir", type=Path, required=True)
    n.add_argument("--formal-run", type=Path)
    for flag in (
        "--typed-final",
        "--public231",
        "--collect",
        "--seal",
        "--report",
        "--calibration",
        "--cache",
        "--base",
    ):
        n.add_argument(flag, type=Path)
    n.add_argument("--hash-workers", type=int, default=8)
    n.add_argument("--output", type=Path, required=True)
    o = sub.add_parser("fill-c27")
    o.add_argument("--table", type=Path, required=True)
    o.add_argument("--key", required=True)
    o.add_argument("--receipt", type=Path, required=True)
    o.add_argument("--repo")
    o.add_argument("--revision")
    o.add_argument("--include")
    o.add_argument("--adopt-paths", action="store_true")
    o.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    handler = {
        "plan": plan,
        "show": show,
        "argv": argv_cmd,
        "verify": verify,
        "parity": parity_cmd,
        "stored": stored_cmd,
        "pairs": pairs_cmd,
        "summary": summary_cmd,
        "digest": digest_cmd,
        "stage": stage_cmd,
        "keys": keys_cmd,
        "field": field_cmd,
        "c27-receipt": c27_receipt,
        "fill-c27": fill_c27,
    }
    try:
        return handler[args.command](args)
    except (ValueError, KeyError, OSError) as exc:
        print(f"event3 {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
