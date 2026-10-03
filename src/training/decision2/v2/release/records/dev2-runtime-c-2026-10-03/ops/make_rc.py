"""Specs and decisions of the runtime C revisions of the six Decision 2.0 repositories.

    python3 make_rc.py preview --base SPEC --runtime-source DIR --key KEY --output OUT
    python3 make_rc.py release --tiers 0.6B ... (--out DIR | --check)

``preview``: a staging spec that builds the tier's released package with this runtime and remote code: the
released spec (Hub IDs made current), kind ``staging`` under ``vllm-sr/dev2-release-staging-rc<key>``, no
release decision, ``runtime_source`` and ``automap_source`` set to the new mirror. Weights, tokenizer and
vendored sources stay those of the released spec.

``release`` (inference owner 885d85cc, track=runtime-c; COORDINATION 2026-10-03 11:35 UTC+8): the runtime-only
revision of each repository. The spec is the exact spec that built the repository's current main (receipts/spec.json
of that release's work directory on node A) with runtime_source and automap_source = the mirror of RC_COMMIT,
card.speed = this record's single-question bench of the new runtime (<key>/bench/bench-new.json), card.speed_shared =
its many-question bench (<key>/shared/shared-new.json, 128 questions), one runtime sentence in runtime_equivalence
and gate_receipt = the new decision. The decision carries the superseded final decision's judgement forward and
changes only the action, the rationale, the runtime evidence and the supersedes chain. A tier is derived only when
its committed parity comparisons under Transformers 5.17 and 5.18 (<key>/parity, <key>/parity-tf518: every prompt of
the four scored panels, old runtime vs new) have 0 answer changes and 0.0 drift, its bench comparison has every
answer bit-identical and its shared bench shares the 128-question request. Run on node A from an exact mirror
(cwd src/training/decision2).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from v2.release import layout

RECORDS = Path("v2/release/records")
SPECS = Path("v2/release/specs")
OUT = RECORDS / "dev2-runtime-c-2026-10-03"
DECISIONS = "/data/dev2/runs/release/decisions"
RELEASES = "/data/dev2/runs/release"
# The runtime C commit every revision ships (set when the runtime is frozen; evidence must be of this commit).
RC_COMMIT = "dcd15f5ceda29947178781c7779b521ff4a20b76"
PANELS = {"typed-final": 1600, "css15": 6547, "public231": 231, "mlx-diag": 2275}
SHARED_QUESTIONS = 128
MARKER = " Checked on one GPU"
PREPARED_BY = "Decision 2.0 inference owner 885d85cc (track=runtime-c, worktree vllm-sr-dev2-runtime-c)"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: COORDINATION 2026-10-03 11:00 / 11:12 UTC+8 (the user stopped "
    "training; the focus is inference optimization; the inference owner ships runtime-only revisions through the "
    "parity rollout, the default path byte-identical to the released runtime)"
)
DECIDED_UTC = "2026-10-03T03:12:00Z"
# The current main of each repository and the release work directory on node A that built and sealed it.
TIERS: dict[str, tuple[str, str, str, str]] = {
    "0.6B": (
        "0p6b",
        "Kai",
        "881bee413681d80ebeac86afcda8b4138dae516e",
        "dev2-ras-0.6B-20261002T235739Z",
    ),
    "0.8B": (
        "0p8b",
        "Eos",
        "ad0aa724c924f7c4194be94b1b8441caf2d61c01",
        "dev2-ras-0.8B-20261003T000237Z",
    ),
    "2B": (
        "2b",
        "Sol",
        "4b75b52114583b4519001492e8dfb0926c89cfe1",
        "dev2-ras-2B-20261003T000946Z",
    ),
    "4B": (
        "4b",
        "Nox",
        "ce1bdc9d91333aae2bf496ec48c66e1a913eb0a0",
        "dev2-4b-lrhxall-release-20261003T003955Z",
    ),
    "9B": (
        "9b",
        "Lux",
        "214ffa4322bc1bce3215c1bd5de6168402c76969",
        "dev2-ras-9B-20261003T022358Z",
    ),
    "27B": (
        "27b",
        "Vega",
        "9b067a95560284dac8c98ef4130fd5a2c5a92ff9",
        "dev2-ras-27B-20261002T213625Z",
    ),
}
SENTENCE = (
    " From this revision the runtime also adds the MLP residual into the residual stream and prepares attention "
    "queries and keys with faster kernels that round exactly as before, runs the opt-in shared-context switch through "
    "the same kernels (its answers are unchanged), enables the fast path under Transformers 5.18 as well as 5.17, and "
    "accepts share_context through AutoModel and the pipeline; with the switch off it was checked with 0 answer "
    "changes and 0.0 drift on every scored prompt and on mlx-diag (10,653 prompts) against the previous runtime under "
    "both Transformers versions."
)


def mirror(commit: str) -> str:
    return f"/data/dev2/src/{commit}-src_training_decision2/src/training/decision2"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preview(base: dict, runtime_source: str, key: str) -> dict:
    spec = layout.current_ids(base)
    for name in ("_release", "gate_receipt"):
        spec.pop(name, None)
    spec["kind"] = "staging"
    spec["repo_id"] = f"{layout.ORG}/dev2-release-staging-rc{key}"
    spec["runtime_source"] = runtime_source
    spec["automap_source"] = runtime_source
    return spec


def _parity_ok(parity: dict) -> bool:
    return (
        parity["passed"] is True
        and parity["max_abs_drift"] == 0.0
        and {
            panel: (p["prompts"], p["identical_prompts"], p["category_changes"])
            for panel, p in parity["panels"].items()
        }
        == {panel: (n, n, 0) for panel, n in PANELS.items()}
    )


def sources(tier: str) -> dict:
    """The release that built the current main (spec, sealed gate, final decision) and this record's evidence."""
    key, codename, revision, work = TIERS[tier]
    name = f"Decision-2.0-{codename}-{tier}"
    receipts = Path(RELEASES) / work / "receipts"
    spec_path, gate_path = receipts / "spec.json", receipts / "gate.json"
    spec, gate, build = (
        _json(spec_path),
        _json(gate_path),
        _json(receipts / "build.json"),
    )
    decision_path = Path(spec["gate_receipt"])
    paths = {
        "parity": OUT / key / "parity" / "answers-compare.json",
        "parity_tf518": OUT / key / "parity-tf518" / "answers-compare.json",
        "bench": OUT / key / "bench" / "compare.json",
        "shared": OUT / key / "shared" / "shared-new.json",
    }
    loaded = {k: _json(p) for k, p in paths.items()}
    bench, shared = loaded["bench"], loaded["shared"]
    entry = (shared.get("per_n") or {}).get(str(SHARED_QUESTIONS)) or {}
    problems = [
        message
        for ok, message in (
            (spec["model_name"] == name, "the spec names another model"),
            (gate["revision"] == revision, "the gate seals another revision"),
            (gate["model_name"] == name, "the gate names another model"),
            (
                gate["decision_sha256"] == sha(decision_path),
                "the gate seals another decision",
            ),
            (build["spec_sha256"] == sha(spec_path), "the build used another spec"),
            (_json(decision_path)["status"] == "final", "the decision is not final"),
            (
                _parity_ok(loaded["parity"]),
                "the Transformers 5.17 parity did not pass at 0.0 drift on every prompt",
            ),
            (
                _parity_ok(loaded["parity_tf518"]),
                "the Transformers 5.18 parity did not pass at 0.0 drift on every prompt",
            ),
            (
                bench["passed"] is True
                and bench["bit_identical_items"] == bench["items"],
                "the bench answers differ",
            ),
            (
                sha(paths["bench"].parent / "bench-new.json") == bench["new"]["sha256"],
                "bench-new.json is not the compared run",
            ),
            (
                shared.get("mode") == "shared"
                and shared.get("passed") is True
                and bool((entry.get("on") or {}).get("shared")),
                "the shared bench did not share the many-question request",
            ),
        )
        if not ok
    ]
    if problems:
        raise ValueError(f"{tier}: {'; '.join(problems)}")
    return {
        "key": key,
        "name": name,
        "revision": revision,
        "spec_path": spec_path,
        "spec": spec,
        "gate_path": gate_path,
        "gate": gate,
        "decision_path": decision_path,
        "decision": _json(decision_path),
        "paths": paths,
        **loaded,
    }


def spec_for(src: dict) -> dict:
    old = src["spec"]
    spec = layout.current_ids({k: v for k, v in old.items() if k != "_release"})
    spec["runtime_source"] = mirror(RC_COMMIT)
    spec["automap_source"] = mirror(RC_COMMIT)
    bench_new = src["paths"]["bench"].parent / "bench-new.json"
    spec["card"]["speed"] = {"evidence": bench_new.as_posix(), "sha256": sha(bench_new)}
    spec["card"]["speed_shared"] = {
        "evidence": src["paths"]["shared"].as_posix(),
        "sha256": sha(src["paths"]["shared"]),
        "questions": SHARED_QUESTIONS,
    }
    text = spec["runtime_equivalence"]
    if text.count(MARKER) == 1:
        spec["runtime_equivalence"] = text.replace(MARKER, SENTENCE + MARKER)
    else:
        spec["runtime_equivalence"] = text + SENTENCE
    spec["gate_receipt"] = f"{DECISIONS}/{src['name']}.decision.rc.json"
    lat = src["bench"]["latency_ms"]
    spec["_release"] = {
        "runtime_c": (
            "Runtime-only revision (COORDINATION 2026-10-03 11:00 / 11:12 UTC+8; inference owner 885d85cc): the "
            f"package runtime and the root Transformers remote code come from commit {RC_COMMIT[:9]} "
            "(runtime_source, automap_source). Weights, tokenizer, configs, the vendored training/model sources, "
            "card index and assets are those of the spec that built "
            f"{spec['repo_id']}@{src['revision'][:8]}. card.speed is the new runtime's bench (p50 "
            f"{lat['p50']['old']:.1f} -> {lat['p50']['new']:.1f} ms), card.speed_shared its many-question bench, and "
            "runtime_equivalence gains one runtime sentence."
        ),
        "replaces_spec": {
            "spec": str(src["spec_path"]),
            "sha256": sha(src["spec_path"]),
        },
        "previous": old.get("_release"),
    }
    return spec


def runtime_evidence(src: dict) -> dict:
    bench, shared = src["bench"], src["shared"]
    lat, mem = bench["latency_ms"], bench["memory_gib"]["request_peak"]
    entry = shared["per_n"][str(SHARED_QUESTIONS)]

    def parity(name: str) -> dict:
        p = src[name]
        return {
            "compare": src["paths"][name].as_posix(),
            "sha256": sha(src["paths"][name]),
            "panels": {
                k: [v["prompts"], v["identical_prompts"], v["category_changes"]]
                for k, v in p["panels"].items()
            },
            "max_abs_drift": p["max_abs_drift"],
        }

    return {
        "parity_transformers_5_17": parity("parity"),
        "parity_transformers_5_18": parity("parity_tf518"),
        "bench": {
            "compare": src["paths"]["bench"].as_posix(),
            "sha256": sha(src["paths"]["bench"]),
            "items": bench["items"],
            "bit_identical_items": bench["bit_identical_items"],
            "p50_ms": [lat["p50"]["old"], lat["p50"]["new"]],
            "p95_ms": [lat["p95"]["old"], lat["p95"]["new"]],
            "request_peak_gib": [mem["old"], mem["new"]],
        },
        "shared": {
            "receipt": src["paths"]["shared"].as_posix(),
            "sha256": sha(src["paths"]["shared"]),
            "questions": SHARED_QUESTIONS,
            "p50_ms_off_on": [
                entry["off"]["latency_ms"]["p50"],
                entry["on"]["latency_ms"]["p50"],
            ],
            "on_vs_off_category_changes": entry["on_vs_off"]["category_changes"],
        },
    }


def decision_for(src: dict, spec_sha: str) -> dict:
    old, gate, bench = src["decision"], src["gate"], src["bench"]
    old_sha = sha(src["decision_path"])
    repo = layout.current_repo(old["repo_id"])
    lat = bench["latency_ms"]
    new = copy.deepcopy(old)
    new.pop("card_revision", None)
    new.update(
        {
            "repo_id": repo,
            "decided_by": DECIDED_BY,
            "prepared_by": PREPARED_BY,
            "decided_utc": DECIDED_UTC,
            "action": (
                f"Runtime-only revision of the public repository {repo}: the package runtime decision2/*.py and the "
                f"root Transformers remote code (modeling_decision2.py, pipeline_decision2.py) come from commit "
                f"{RC_COMMIT[:9]}, which adds the MLP residual and prepares attention queries and keys with faster "
                "kernels that round exactly as before, runs the opt-in shared-context switch through the fused "
                "kernels, enables the fast path under Transformers 5.17 and 5.18 and passes share_context through "
                "AutoModel and the pipeline. Weights, tokenizer, configs, the vendored training/model sources, "
                f"calibration, any Score offsets, the banner and the charts are byte-identical to the released revision "
                f"{gate['revision']}; README.md states the new speed and the share_context call, and "
                "MODEL_MANIFEST.json the new runtime hashes and the runtime sentence. The collection is not changed."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision. Only the package runtime and the remote code change, and with the switch off (the "
                "default) they compute the same values bit for bit: every prompt of the four scored panels "
                "(typed-final 1,600, css15 6,547, public231 231, mlx-diag 2,275) answered through the released package "
                "and through this package on the released weights gave 0 answer changes and 0.0 drift under "
                "Transformers 5.17 and under 5.18; 400 single requests: p50 "
                f"{lat['p50']['old']:.2f} -> {lat['p50']['new']:.2f} ms, p95 {lat['p95']['old']:.2f} -> "
                f"{lat['p95']['new']:.2f} ms, all bit-identical. release.sh checks the native examples, the card's "
                "Transformers example before upload, after the real download and from the Hub in fresh environments "
                "under Transformers 5.17 and 5.18, the card structure and every card link; then the downloaded "
                "package must pass the Index harness's 86-request parity gate."
            ),
            "previous_rationale": old["rationale"],
            "runtime_revision": {
                "kind": "runtime-c",
                "runtime_commit": RC_COMMIT,
                "spec": f"v2/release/specs/dev2-{src['key']}-rc.json",
                "spec_sha256": spec_sha,
                **runtime_evidence(src),
            },
            "supersedes": {
                "final_sha256": old_sha,
                "released_as": f"{gate['repo_id']}@{gate['revision']}",
                "released_manifest_sha256": gate["manifest_sha256"],
                "released_gate_sha256": sha(src["gate_path"]),
                "card_revision": old.get("card_revision"),
                "earlier": old.get("supersedes"),
            },
        }
    )
    return new


def texts(tier: str) -> list[tuple[Path, str]]:
    src = sources(tier)
    spec_text = json.dumps(spec_for(src), ensure_ascii=False, indent=2) + "\n"
    spec_sha = hashlib.sha256(spec_text.encode()).hexdigest()
    decision_text = (
        json.dumps(decision_for(src, spec_sha), ensure_ascii=False, indent=2) + "\n"
    )
    return [
        (SPECS / f"dev2-{src['key']}-rc.json", spec_text),
        (OUT / f"{src['name']}.decision.rc.json", decision_text),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preview")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--runtime-source", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("release")
    r.add_argument("--tiers", nargs="+", required=True, choices=sorted(TIERS))
    mode = r.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--out", type=Path, help="write the derived files under this directory"
    )
    mode.add_argument(
        "--check", action="store_true", help="compare with the files in this tree"
    )
    args = parser.parse_args()
    if args.command == "preview":
        base = json.loads(args.base.read_text(encoding="utf-8"))
        source = Path(args.runtime_source)
        if not (source / "v2/release/runtime/fast.py").is_file():
            raise SystemExit(f"{source} has no fast-path runtime")
        spec = preview(base, str(source), args.key)
        with args.output.open("x", encoding="utf-8") as sink:
            json.dump(spec, sink, indent=2, ensure_ascii=False)
            sink.write("\n")
        print(json.dumps({"spec": str(args.output), "repo_id": spec["repo_id"]}))
        return 0
    problems = []
    for tier in args.tiers:
        for path, text in texts(tier):
            if args.check:
                if not path.is_file() or path.read_text(encoding="utf-8") != text:
                    problems.append(f"{path}: differs from the derivation")
            else:
                target = args.out / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            print(path, hashlib.sha256(text.encode()).hexdigest())
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
