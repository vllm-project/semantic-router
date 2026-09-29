"""JevArena-C1 v1.2 post-key successor guard: plan, checks and ledger for ``c1-postkey.sh``.

    python3 -m v2.eval.sealed.postkey plan --spec SPEC --registry R --output PLAN.json
    python3 -m v2.eval.sealed.postkey show --plan P
    python3 -m v2.eval.sealed.postkey field --plan P --field model.cache.frozen
    python3 -m v2.eval.sealed.postkey verify --plan P --src-root S --output VERIFY.json
    python3 -m v2.eval.sealed.postkey argv --plan P --phase smoke|collect --run-dir D --gpu N \
        --src MIRROR --src-root S --lease-name owner.NAME [--shared] [--cache-dir C]
    python3 -m v2.eval.sealed.postkey parity --plan P --run-dir D --output PARITY.json
    python3 -m v2.eval.sealed.postkey ledger-check --plan P --ledger L [--approval TEXT]
    python3 -m v2.eval.sealed.postkey gates --plan P --job-dir J
    python3 -m v2.eval.sealed.postkey finish --plan P --job-dir J --ledger L [--approval TEXT] \
        --output SUMMARY.json
    python3 -m v2.eval.sealed.postkey reproduce --event-dir E --gold G --output-dir D

JevArena-C1 is post-key after its three scoring events. Its only use is successor-rule item 8:
a successor must not show a significant C1 regression against the current revision
(``v2.eval.gates c1``). It is never training data and never a selection criterion, in
development or among siblings, and a card may report it only as "JevArena-C1 v1.2, post-key
(not an independent validation)".

``plan`` resolves one spec (``dev2-c1-postkey-spec/1``: one frozen release package with its
formal runtime, written like a row of the event-3 table) against the baseline registry
(``c1-postkey-baselines.json``). A ``successor`` is gated against its tier's registered
baseline, which must hold other weights; a ``current`` revision builds its tier's baseline.
A spec's ``compare`` runs are gated for information only. ``verify`` checks the image, the
mirror's adapter module, paths, the package manifest and identity, pinned files, the frozen
cache and each comparison run's seal file (CPU). ``parity`` compares the smoke with the stored
formal run and must be exact. ``ledger-check`` enforces non-selection before the key is read:
one successor per tier baseline and no second scoring of the same weights, unless
``--approval`` records the coordinator's approval. ``gates`` prints the gate calls
(NUL-separated: right run, right name, output). ``finish`` writes the summary and appends the
custodial ledger. ``reproduce`` repeats the event-3 paired comparisons with the gate and
compares them with the event's paired files. Only aggregates are written; none of these reads
C1 prompts, and only ``reproduce`` reads the gold (for ``v2.eval.gates c1``).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from v2.eval.sealed import event3
from v2.eval.sealed import score as c1score

SCHEMA = "dev2-c1-postkey/1"
SPEC_SCHEMA = "dev2-c1-postkey-spec/1"
REGISTRY_SCHEMA = "dev2-c1-postkey-baselines/1"
ROLES = ("current", "successor")
RUN = "cand"
USE = (
    "successor-rule item 8 only: never training data, never a selection criterion"
    " (development or siblings)"
)
REQUIRED = (
    "name",
    "tier",
    "role",
    "label",
    "repo",
    "revision",
    "image",
    "adapter_spec",
    "model_path",
    "limit",
    "parity",
    "identity",
    "package",
)
SHA = re.compile(r"^[0-9a-f]{64}$")
NAME = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
KEY = re.compile(r"^[a-z0-9][a-z0-9-]*$")
COMPARE_FIELDS = ("key", "name", "run", "seal_sha256")


def load(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{path} is not {schema}")
    return value


def resolve(spec: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    """The plan for one spec, or ValueError listing every problem."""
    errors = [f"missing {k}" for k in REQUIRED if spec.get(k) in (None, "")]
    errors += [
        f"{event3.PLACEHOLDER} in {where}" for where in event3.placeholders(spec)
    ]
    if (registry.get("item_set") or {}).get(
        "retired_sha256"
    ) != c1score.POSTKEY_RETIRED_SHA256:
        errors.append(f"the registry is not for item set {c1score.POSTKEY_ITEM_SET}")
    tier, role, identity = spec.get("tier"), spec.get("role"), spec.get("identity")
    if tier not in event3.TIERS:
        errors.append(f"unknown tier {tier!r}")
    if role not in ROLES:
        errors.append(f"role must be one of {', '.join(ROLES)}")
    if not NAME.match(str(spec.get("name", ""))):
        errors.append("name must be lowercase letters, digits, '.' and '-'")
    if spec.get("image") not in registry.get("images", {}):
        errors.append(f"unknown image {spec.get('image')!r}")
    if spec.get("adapter"):
        errors.append(
            "give adapter_spec (the formal runtime's adapter file), not adapter"
        )
    parity = spec.get("parity") or {}
    if parity.get("mode") != "exact" or not parity.get("stored"):
        errors.append("parity must be exact against the stored formal run")
    package = spec.get("package") or {}
    for field in ("manifest_sha256", "identity"):
        if not SHA.match(str(package.get(field, ""))):
            errors.append(f"package.{field} must be a SHA-256 (a frozen package)")
    if package.get("identity") != identity:
        errors.append("identity differs from the package's weights identity")
    if package.get("dir") != spec.get("model_path"):
        errors.append("model_path must be the frozen package dir")
    smoke = spec.get("smoke_items")
    if smoke is not None and (not isinstance(smoke, int) or smoke < 1):
        errors.append("smoke_items must be a positive integer or null (whole panels)")
    comparisons: list[dict[str, Any]] = []
    baseline = (registry.get("tiers") or {}).get(tier)
    if role == "successor":
        if baseline is None:
            errors.append(
                f"no registered C1 baseline for {tier}: collect the current revision first"
            )
        elif baseline.get("identity") == identity:
            errors.append(
                "same weights as the registered baseline: nothing for item 8 to gate"
            )
        else:
            comparisons.append(
                {
                    "key": "baseline",
                    "kind": "item8",
                    "name": f"{baseline['model']} {baseline['revision'][:8]} (current revision)",
                    "run": baseline["run"],
                    "seal_sha256": baseline["seal_sha256"],
                }
            )
    elif role == "current" and baseline and baseline.get("identity") == identity:
        errors.append(
            f"{tier} already has a baseline with these weights ({baseline['run']})"
        )
    for entry in spec.get("compare", []):
        missing = [f for f in COMPARE_FIELDS if not entry.get(f)]
        if missing:
            errors.append(f"compare entry lacks {', '.join(missing)}")
            continue
        if not KEY.match(entry["key"]) or entry["key"] == "baseline":
            errors.append(f"compare key {entry['key']!r} is not allowed")
        if not SHA.match(entry["seal_sha256"]):
            errors.append(f"compare {entry['key']}: seal_sha256 must be a SHA-256")
        comparisons.append({**{f: entry[f] for f in COMPARE_FIELDS}, "kind": "info"})
    keys = [c["key"] for c in comparisons]
    if len(set(keys)) != len(keys):
        errors.append("duplicate compare keys")
    if errors:
        raise ValueError("; ".join(errors))
    model = {
        k: v
        for k, v in spec.items()
        if k not in ("schema", "name", "role", "compare", "notes")
    }
    return {
        "schema": SCHEMA,
        "label": c1score.POSTKEY_LABEL,
        "use": USE,
        "name": spec["name"],
        "tier": tier,
        "role": role,
        "images": registry["images"],
        "item_set": registry["item_set"],
        "model": model,
        "comparisons": comparisons,
    }


def plan(args: argparse.Namespace) -> int:
    try:
        result = resolve(
            load(args.spec, SPEC_SCHEMA), load(args.registry, REGISTRY_SCHEMA)
        )
    except ValueError as exc:
        print(f"plan refused: {exc}", file=sys.stderr)
        return 2
    result.update(
        created_utc=event3.utc(),
        spec_sha256=event3.sha_file(args.spec),
        registry_sha256=event3.sha_file(args.registry),
    )
    event3.write_new(args.output, result)
    print(
        json.dumps(
            {
                "name": result["name"],
                "tier": result["tier"],
                "role": result["role"],
                "comparisons": [c["key"] for c in result["comparisons"]],
            }
        )
    )
    return 0


def load_plan(path: Path) -> dict[str, Any]:
    return load(path, SCHEMA)


def show(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    row = p["model"]
    print(f"{p['name']}: {p['tier']} {p['role']} {row['label']} {row['revision']}")
    print(
        f"  {row['adapter_spec']} image={row['image']} identity={row['identity'][:12]}"
        f" cache={'frozen' if row.get('cache') else 'none'} limit={row['limit']}"
    )
    for c in p["comparisons"]:
        print(f"  gate ({c['kind']}) vs {c['name']}: {c['run']}")
    return 0


def field_cmd(args: argparse.Namespace) -> int:
    value: Any = load_plan(args.plan)
    for part in args.field.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value is not None:
        print(value if isinstance(value, (str, int, float)) else json.dumps(value))
    return 0


def verify(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    row = p["model"]
    image_id = p["images"][row["image"]]
    try:
        model = event3.verify_row(row, Path(args.src_root), args.hash_workers)
    except (OSError, ValueError, KeyError) as exc:
        model = {"problems": [f"verification error: {exc!r}"]}
    comparisons = {}
    for c in p["comparisons"]:
        seal = Path(c["run"]) / "SEAL-C1.json"
        problems = []
        if not seal.is_file():
            problems.append("no SEAL-C1.json")
        elif event3.sha_file(seal) != c["seal_sha256"]:
            problems.append("seal file differs from the pinned seal")
        if not (Path(c["run"]) / "output" / "sealed-c1.predictions.jsonl").is_file():
            problems.append("no sealed C1 predictions")
        comparisons[c["key"]] = {"run": c["run"], "problems": problems}
    report = {
        "schema": SCHEMA + "/verify",
        "utc": event3.utc(),
        "image": {"id": image_id, "present": event3.image_present(image_id)},
        "model": model,
        "comparisons": comparisons,
    }
    report["passed"] = (
        report["image"]["present"]
        and not model["problems"]
        and not any(v["problems"] for v in comparisons.values())
    )
    event3.write_new(args.output, report)
    print(f"image {image_id[:19]}: {'ok' if report['image']['present'] else 'MISSING'}")
    print(
        f"model: {'ok' if not model['problems'] else 'FAILED: ' + '; '.join(model['problems'])}"
    )
    for key, v in comparisons.items():
        print(
            f"comparison {key}: {'ok' if not v['problems'] else 'FAILED: ' + '; '.join(v['problems'])}"
        )
    return 0 if report["passed"] else 1


def argv_cmd(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    row = p["model"]
    out = event3.runner_argv(
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
    what = "preflight smoke" if args.phase == "smoke" else "collection"
    out[out.index("--purpose") + 1] = (
        f"C1 v1.2 post-key {what}: {row['label']} {row['revision'][:8]}"
    )
    sys.stdout.write("\0".join(out) + "\0")
    return 0


def parity(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    row = p["model"]
    panels = {}
    for panel in event3.SMOKE_PANELS:
        path = event3.smoke_predictions(args.run_dir, panel)
        stored_path = event3.stored_predictions(row, panel)
        stored = {r["id"]: r for r in event3.read_jsonl(stored_path)}
        panels[panel] = event3.compare_smoke(
            event3.read_jsonl(path) if path else [], stored, row
        )
        panels[panel].update(
            preflight=str(path) if path else None, stored_run=str(stored_path)
        )
    result = {
        "schema": SCHEMA + "/parity",
        "mode": row["parity"]["mode"],
        "panels": panels,
        "passed": all(v["passed"] for v in panels.values()),
    }
    event3.write_new(args.output, result)
    for panel, v in panels.items():
        print(
            json.dumps(
                {"panel": panel}
                | {
                    k: v[k]
                    for k in ("prompts", "answers_by_type", "changed")
                    + ("max_probability_drift", "passed")
                }
            )
        )
    return 0 if result["passed"] else 1


def ledger_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return event3.read_jsonl(path)


def item8(p: dict[str, Any]) -> dict[str, Any] | None:
    return next((c for c in p["comparisons"] if c["kind"] == "item8"), None)


def ledger_problems(p: dict[str, Any], entries: list[dict[str, Any]]) -> list[str]:
    identity = p["model"]["identity"]
    baseline = item8(p)
    problems = []
    for entry in entries:
        if entry.get("tier") != p["tier"]:
            continue
        if entry.get("identity") == identity:
            problems.append(
                f"these weights were already scored post-key ({entry['job_dir']}); reuse that run"
            )
        elif (
            baseline is not None
            and entry.get("role") == "successor"
            and entry.get("baseline_seal_sha256") == baseline["seal_sha256"]
        ):
            problems.append(
                f"a successor of the same baseline was already scored ({entry['job_dir']}):"
                " C1 is never a selection criterion among siblings"
            )
    return problems


def ledger_check(args: argparse.Namespace) -> int:
    problems = ledger_problems(load_plan(args.plan), ledger_entries(args.ledger))
    for line in problems:
        print(f"ledger: {line}", file=sys.stderr)
    if problems and not args.approval:
        print(
            "ledger: refused; a recorded coordinator approval (--approval) is required",
            file=sys.stderr,
        )
        return 1
    if problems:
        print(f"ledger: proceeding under the recorded approval: {args.approval}")
    return 0


def gate_output(job_dir: Path, key: str) -> Path:
    return job_dir / f"GATE-C1-vs-{key}.json"


def gates_cmd(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    fields = []
    for c in p["comparisons"]:
        fields += [c["run"], c["name"], str(gate_output(args.job_dir, c["key"]))]
    sys.stdout.write("".join(f + "\0" for f in fields))
    return 0


def gpu_time(run: Path) -> dict[str, Any] | None:
    path = run / "GPU-TIME.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        k: value.get(k)
        for k in ("gpu", "wall_seconds", "gpu_hours", "exit_code", "shared")
    }


def finish(args: argparse.Namespace) -> int:
    p = load_plan(args.plan)
    row = p["model"]
    run = args.job_dir / RUN
    report_path, seal_path = run / "REPORT-C1.json", run / "SEAL-C1.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sealed = json.loads(seal_path.read_text(encoding="utf-8"))
    comparisons = []
    for c in p["comparisons"]:
        path = gate_output(args.job_dir, c["key"])
        entry: dict[str, Any] = {k: c[k] for k in ("key", "kind", "name", "run")}
        if path.is_file():
            gate = json.loads(path.read_text(encoding="utf-8"))
            entry.update(
                verdict=gate["verdict"],
                delta=gate["delta"],
                ci95=gate["ci95"],
                p=gate["p"],
                right_c1=gate["right"]["c1"],
                by_type={
                    k: {f: v[f] for f in ("delta", "ci95", "p")}
                    for k, v in gate["by_type"].items()
                },
                sha256=event3.sha_file(path),
            )
        else:
            entry.update(verdict=None, problem="gate output missing")
        comparisons.append(entry)
    gpu = {name: gpu_time(args.job_dir / name) for name in ("smoke", RUN)}
    rule = next((c for c in comparisons if c["kind"] == "item8"), None)
    summary = {
        "schema": SCHEMA + "/summary",
        "utc": event3.utc(),
        "label": c1score.POSTKEY_LABEL,
        "use": USE,
        "name": p["name"],
        "tier": p["tier"],
        "role": p["role"],
        "model": {
            k: row.get(k)
            for k in ("label", "repo", "revision", "identity", "adapter_spec", "limit")
        }
        | {"manifest_sha256": row["package"]["manifest_sha256"]},
        "c1": report["c1"],
        "by_type": report["by_type"],
        "items": report["items"],
        "valid": report["valid"],
        "item_set": report.get("item_set"),
        "report_sha256": event3.sha_file(report_path),
        "seal_sha256": event3.sha_file(seal_path),
        "predictions_sha256": sealed["predictions_sha256"],
        "comparisons": comparisons,
        "item8": (
            None
            if rule is None
            else {k: rule.get(k) for k in ("verdict", "delta", "ci95", "p", "name")}
        ),
        "gpu": gpu,
        "gpu_hours": sum((v or {}).get("gpu_hours") or 0 for v in gpu.values()),
        "baseline_entry": {
            "model": row["label"],
            "repo": row["repo"],
            "revision": row["revision"],
            "identity": row["identity"],
            "run": str(run),
            "seal_sha256": event3.sha_file(seal_path),
            "predictions_sha256": sealed["predictions_sha256"],
            "c1": report["c1"],
            "source": f"post-key C1 {c1score.POSTKEY_ITEM_SET} collection {args.job_dir.name}",
        },
        "approval": args.approval,
    }
    digest = event3.write_new(args.output, summary)
    baseline = item8(p)
    line = {
        "utc": summary["utc"],
        "tier": p["tier"],
        "role": p["role"],
        "name": p["name"],
        "label": row["label"],
        "revision": row["revision"],
        "identity": row["identity"],
        "manifest_sha256": row["package"]["manifest_sha256"],
        "job_dir": str(args.job_dir),
        "seal_sha256": summary["seal_sha256"],
        "report_sha256": summary["report_sha256"],
        "summary_sha256": digest,
        "c1": report["c1"],
        "baseline_seal_sha256": baseline["seal_sha256"] if baseline else None,
        "item8_verdict": (summary["item8"] or {}).get("verdict"),
        "approval": args.approval,
    }
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(args.ledger, "a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(line, sort_keys=True) + "\n")
    print(
        f"{row['label']} {row['revision'][:8]}: C1 {report['c1']:.2f} (valid {report['valid']})"
    )
    for c in comparisons:
        if c["verdict"] is None:
            print(f"  vs {c['name']}: GATE MISSING")
        else:
            print(
                f"  vs {c['name']} ({c['kind']}): {c['delta']:+.2f}"
                f" [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] p={c['p']:.3f} {c['verdict']}"
            )
    print(f"GPU-hours {summary['gpu_hours']:.3f}")
    return 0 if all(c["verdict"] for c in comparisons) else 1


def same_pair(got: dict[str, Any], want: dict[str, Any]) -> bool:
    return (
        got["delta"] == want["delta"]
        and got["ci95"] == want["ci95"]
        and got["item_set"] == want.get("item_set")
        and set(got["by_type"]) == set(want["by_type"])
        and all(
            got["by_type"][k]["delta"] == v["delta"]
            and got["by_type"][k]["ci95"] == v["ci95"]
            for k, v in want["by_type"].items()
        )
    )


def reproduce_one(job: tuple[str, ...]) -> dict[str, Any]:
    from v2.eval import gates

    name, left, right, left_name, right_name, gold, output, expected = job
    with contextlib.redirect_stdout(io.StringIO()):
        gates.main(
            ["c1", "--left", left, "--right", right, "--left-name", left_name]
            + ["--right-name", right_name, "--gold", gold, "--output", output]
        )
    got = json.loads(Path(output).read_text(encoding="utf-8"))
    want = json.loads(Path(expected).read_text(encoding="utf-8"))
    return {
        "pair": name,
        "left": left_name,
        "right": right_name,
        "delta": got["delta"],
        "ci95": got["ci95"],
        "p": got["p"],
        "verdict": got["verdict"],
        "by_type": {
            k: {f: v[f] for f in ("delta", "ci95", "p")}
            for k, v in got["by_type"].items()
        },
        "matches_event": same_pair(got, want),
        "gate_sha256": event3.sha_file(Path(output)),
        "event_paired_sha256": event3.sha_file(Path(expected)),
    }


def reproduce(args: argparse.Namespace) -> int:
    event_plan = json.loads((args.event_dir / "PLAN.json").read_text(encoding="utf-8"))
    runs = {
        key: (
            Path(row["stored"]["seal"]).parent
            if row.get("stored")
            else args.event_dir / key
        )
        for key, row in event_plan["models"].items()
    }
    jobs = []
    for pair in event_plan["pairs"]:
        left, right = pair["left"], pair["right"]
        jobs.append(
            (
                f"{left} vs {right}",
                str(runs[left]),
                str(runs[right]),
                event_plan["models"][left]["label"],
                event_plan["models"][right]["label"],
                str(args.gold),
                str(args.output_dir / f"GATE-C1-{left}-vs-{right}.json"),
                str(args.event_dir / f"PAIRED-C1-{left}-vs-{right}.json"),
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers <= 1:
        pairs = [reproduce_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(len(jobs), args.workers)) as pool:
            pairs = list(pool.map(reproduce_one, jobs))
    result = {
        "schema": SCHEMA + "/reproduce",
        "utc": event3.utc(),
        "event_dir": str(args.event_dir),
        "event_plan_sha256": event3.sha_file(args.event_dir / "PLAN.json"),
        "pairs": pairs,
        "all_match": all(p["matches_event"] for p in pairs),
    }
    event3.write_new(args.output_dir / "REPRODUCE.json", result)
    for p in pairs:
        print(
            f"{p['pair']}: {p['delta']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]"
            f" p={p['p']:.4f} {p['verdict']} matches_event={p['matches_event']}"
        )
    return 0 if result["all_match"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--spec", type=Path, required=True)
    a.add_argument("--registry", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("show")
    b.add_argument("--plan", type=Path, required=True)
    c = sub.add_parser("field")
    c.add_argument("--plan", type=Path, required=True)
    c.add_argument("--field", required=True)
    d = sub.add_parser("verify")
    d.add_argument("--plan", type=Path, required=True)
    d.add_argument("--src-root", type=Path, required=True)
    d.add_argument("--hash-workers", type=int, default=8)
    d.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("argv")
    e.add_argument("--plan", type=Path, required=True)
    e.add_argument("--phase", choices=("smoke", "collect"), required=True)
    e.add_argument("--run-dir", type=Path, required=True)
    e.add_argument("--gpu", required=True)
    e.add_argument("--src", required=True)
    e.add_argument("--src-root", type=Path, required=True)
    e.add_argument("--lease-name", required=True)
    e.add_argument("--shared", action="store_true")
    e.add_argument("--cache-dir")
    f = sub.add_parser("parity")
    f.add_argument("--plan", type=Path, required=True)
    f.add_argument("--run-dir", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    g = sub.add_parser("ledger-check")
    g.add_argument("--plan", type=Path, required=True)
    g.add_argument("--ledger", type=Path, required=True)
    g.add_argument("--approval")
    h = sub.add_parser("gates")
    h.add_argument("--plan", type=Path, required=True)
    h.add_argument("--job-dir", type=Path, required=True)
    i = sub.add_parser("finish")
    i.add_argument("--plan", type=Path, required=True)
    i.add_argument("--job-dir", type=Path, required=True)
    i.add_argument("--ledger", type=Path, required=True)
    i.add_argument("--approval")
    i.add_argument("--output", type=Path, required=True)
    j = sub.add_parser("reproduce")
    j.add_argument("--event-dir", type=Path, required=True)
    j.add_argument("--gold", type=Path, required=True)
    j.add_argument("--output-dir", type=Path, required=True)
    j.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    return {
        "plan": plan,
        "show": show,
        "field": field_cmd,
        "verify": verify,
        "argv": argv_cmd,
        "parity": parity,
        "ledger-check": ledger_check,
        "gates": gates_cmd,
        "finish": finish,
        "reproduce": reproduce,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
