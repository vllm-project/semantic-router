"""Specs and decisions of the speed-up phase A runtime (``runtime/fast.py``) for the Decision 2.0 tiers.

    python3 make_fast.py preview --base SPEC --runtime-source DIR --key KEY --output OUT
    python3 make_fast.py release --tiers 0.6B ... (--out DIR | --check)

``preview``: a staging spec that builds the tier's released package with the new
runtime: the released spec (Hub IDs made current), kind ``staging`` under
``vllm-sr/dev2-release-staging-ra<key>``, no release decision (the gate profile stays: the card's comparison follows it),
and ``runtime_source`` set to the new runtime's mirror. Weights, tokenizer,
vendored sources and remote code stay those of the released spec; only the
package runtime (and the staging name in config / manifest / card) differs.

``release`` (user approval 2026-10-02 18:00 UTC+8, COORDINATION 18:00; worker 2d541b40, track=runtime-a):
the runtime-only revision of each repository. The spec is the exact spec that built the repository's current
main (receipts/spec.json of that release's work directory on node A) with runtime_source = the mirror of the
phase A runtime commit, card.speed = this record's bench of the new runtime (<key>/bench/bench-new.json), one
runtime sentence in runtime_equivalence and gate_receipt = the new decision. The decision carries the superseded
final decision's judgement forward and changes only the action, the rationale, the runtime evidence and the
supersedes chain. A tier is derived only when its committed parity comparison (<key>/parity/answers-compare.json:
every prompt of the four scored panels, old runtime vs new) has 0 answer changes and 0.0 drift and its bench
comparison has every answer bit-identical. Run on node A from an exact mirror (cwd src/training/decision2).
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
OUT = RECORDS / "dev2-runtime-a-2026-10-02"
DECISIONS = "/data/dev2/runs/release/decisions"
RELEASES = "/data/dev2/runs/release"
RUNTIME_COMMIT = "54303117be355595ddf5bb184905be6c6d2ecae9"
PANELS = {"typed-final": 1600, "css15": 6547, "public231": 231, "mlx-diag": 2275}
MARKER = " Checked on one GPU"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's approval of speed-up phase A (2026-10-02 18:00 "
    "UTC+8, COORDINATION 18:00): runtime-only revisions of the six Decision 2.0 repositories with the exact GPU "
    "fast path, each only with 0 answer changes and 0.0 drift on all four scored panels against the released runtime"
)
PREPARED_BY = "Decision 2.0 runtime phase A worker 2d541b40 (track=runtime-a, worktree vllm-sr-dev2-runtime-a)"
DECIDED_UTC = "2026-10-02T10:00:00Z"
# The current main of each repository and the release work directory on node A that built and sealed it (the org
# worker's card-only revisions; Kai's was built on node E and its receipts copied to the same path on node A).
TIERS: dict[str, tuple[str, str, str | None, str | None]] = {
    "0.6B": (
        "0p6b",
        "Kai",
        "c441862b8c3021957c9b5cc2080a3ed5351c1f67",
        "dev2-org-0.6B-20261002T102509Z",
    ),
    "0.8B": (
        "0p8b",
        "Eos",
        "25f0914a188e5449c7e107f13e1376ddd47a43bc",
        "dev2-org-0.8B-20261002T104330Z",
    ),
    "2B": (
        "2b",
        "Sol",
        "951e7f7ff04237021793da50a959955b1fe35635",
        "dev2-org-2B-20261002T105043Z",
    ),
    "4B": (
        "4b",
        "Nox",
        "36596d27503796f94c41e66999ba71bb3fa5c8e4",
        "dev2-org-4B-20261002T103137Z",
    ),
    "9B": (
        "9b",
        "Lux",
        "6af07f3684132ab684d4f064a723ae623ec3956f",
        "dev2-org-9B-20261002T105801Z",
    ),
    "27B": (
        "27b",
        "Vega",
        "e60bd8e346110b3348b6e5a25c07329d350c4b21",
        "dev2-org-27B-20261002T113747Z",
    ),
}


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
    spec["repo_id"] = f"{layout.ORG}/dev2-release-staging-ra{key}"
    spec["runtime_source"] = runtime_source
    return spec


def sources(tier: str) -> dict:
    """The release that built the current main (spec, sealed gate, final decision) and this record's evidence."""
    key, codename, revision, work = TIERS[tier]
    if revision is None or work is None:
        raise ValueError(f"{tier}: the current main is not pinned yet")
    name = f"Decision-2.0-{codename}-{tier}"
    receipts = Path(RELEASES) / work / "receipts"
    spec_path, gate_path = receipts / "spec.json", receipts / "gate.json"
    spec, gate, build = (
        _json(spec_path),
        _json(gate_path),
        _json(receipts / "build.json"),
    )
    decision_path = Path(spec["gate_receipt"])
    parity_path = OUT / key / "parity" / "answers-compare.json"
    bench_path = OUT / key / "bench" / "compare.json"
    parity, bench = _json(parity_path), _json(bench_path)
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
                parity["passed"] is True and parity["max_abs_drift"] == 0.0,
                "the parity comparison did not pass at 0.0 drift",
            ),
            (
                {
                    panel: (p["prompts"], p["identical_prompts"], p["category_changes"])
                    for panel, p in parity["panels"].items()
                }
                == {panel: (n, n, 0) for panel, n in PANELS.items()},
                "the parity comparison does not cover every scored prompt identically",
            ),
            (
                bench["passed"] is True
                and bench["bit_identical_items"] == bench["items"],
                "the bench answers differ",
            ),
            (
                sha(bench_path.parent / "bench-new.json") == bench["new"]["sha256"],
                "bench-new.json is not the compared run",
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
        "parity_path": parity_path,
        "parity": parity,
        "bench_path": bench_path,
        "bench": bench,
    }


def runtime_sentence(lora: bool) -> str:
    lean = (
        "; the unmerged LoRA layers multiply their BF16 factors with one input cast"
        if lora
        else ""
    )
    return (
        " From this revision the runtime replays the backbone of each exact padded input shape as a HIP graph, "
        "casts a Linear input shared by several layers to BF16 once and, on MI300-class (gfx942) GPUs, fuses the "
        f"element-wise ops of each decoder layer into Triton kernels that round exactly as the ops they replace{lean}; "
        "it was checked with 0 answer changes and 0.0 drift on every scored prompt and on mlx-diag (10,653 prompts) "
        "against the previous runtime."
    )


def spec_for(src: dict) -> dict:
    old = src["spec"]
    spec = layout.current_ids({k: v for k, v in old.items() if k != "_release"})
    spec["runtime_source"] = mirror(RUNTIME_COMMIT)
    spec["card"]["speed"] = {
        "evidence": (src["bench_path"].parent / "bench-new.json").as_posix(),
        "sha256": src["bench"]["new"]["sha256"],
    }
    text = spec["runtime_equivalence"]
    sentence = runtime_sentence(src["key"] == "27b")
    if text.count(MARKER) == 1:
        spec["runtime_equivalence"] = text.replace(MARKER, sentence + MARKER)
    else:
        spec["runtime_equivalence"] = text + sentence
    spec["gate_receipt"] = f"{DECISIONS}/{src['name']}.decision.ra.json"
    lat = src["bench"]["latency_ms"]
    spec["_release"] = {
        "runtime_a": (
            "Runtime-only revision (user approval of speed-up phase A, 2026-10-02 18:00 UTC+8; worker 2d541b40): "
            f"the package runtime comes from the phase A commit {RUNTIME_COMMIT[:9]} (runtime_source), the exact "
            "GPU fast path of runtime/fast.py. Weights, tokenizer, configs, the vendored training/model sources, "
            "card index, assets and remote code are those of the spec that built "
            f"{spec['repo_id']}@{src['revision'][:8]}. card.speed is the new runtime's bench (p50 "
            f"{lat['p50']['old']:.1f} -> {lat['p50']['new']:.1f} ms) and runtime_equivalence gains one runtime "
            "sentence."
        ),
        "replaces_spec": {
            "spec": str(src["spec_path"]),
            "sha256": sha(src["spec_path"]),
        },
        "previous": old.get("_release"),
    }
    return spec


def decision_for(src: dict, spec_sha: str) -> dict:
    old, gate, bench, parity = src["decision"], src["gate"], src["bench"], src["parity"]
    old_sha = sha(src["decision_path"])
    repo = layout.current_repo(old["repo_id"])
    lat, mem = bench["latency_ms"], bench["memory_gib"]["request_peak"]
    new = copy.deepcopy(old)
    new.pop("card_revision", None)
    new.update(
        {
            "repo_id": repo,
            "decided_by": DECIDED_BY,
            "prepared_by": PREPARED_BY,
            "decided_utc": DECIDED_UTC,
            "action": (
                f"Runtime-only revision of the private repository {repo}: the package runtime decision2/*.py comes "
                f"from the phase A runtime commit {RUNTIME_COMMIT[:9]}, which on a ROCm GPU under Transformers "
                "5.17 replays the backbone of each exact padded shape as a HIP graph, casts shared Linear inputs to "
                "BF16 once and (on gfx942) runs fused Triton element-wise kernels that round exactly as the ops they "
                "replace. Weights, tokenizer, configs, the vendored training/model sources, calibration, any Score "
                f"offsets, the banner and the charts are byte-identical to the released revision {gate['revision']}; "
                "README.md states the new median latency and MODEL_MANIFEST.json the new runtime hashes and one "
                "runtime sentence. The collection is not changed; everything stays private."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision. Only the package runtime changes, and it computes the same values bit for bit: "
                "every prompt of the four scored panels (typed-final 1,600, css15 6,547, public231 231, mlx-diag "
                "2,275) answered through the released package and through this package on the released weights "
                f"gave 0 answer changes and 0.0 drift; 400 single requests: p50 {lat['p50']['old']:.2f} -> "
                f"{lat['p50']['new']:.2f} ms, p95 {lat['p95']['old']:.2f} -> {lat['p95']['new']:.2f} ms, all "
                "bit-identical. release.sh checks the native examples, the card's Transformers example before "
                "upload, after the real download and from the Hub in fresh environments under Transformers 5.17 "
                "and 5.18, the card structure and every card link."
            ),
            "previous_rationale": old["rationale"],
            "runtime_revision": {
                "kind": "fast-path-runtime",
                "runtime_commit": RUNTIME_COMMIT,
                "spec": f"v2/release/specs/dev2-{src['key']}-ra.json",
                "spec_sha256": spec_sha,
                "parity": {
                    "compare": src["parity_path"].as_posix(),
                    "sha256": sha(src["parity_path"]),
                    "panels": {
                        panel: [
                            p["prompts"],
                            p["identical_prompts"],
                            p["category_changes"],
                        ]
                        for panel, p in parity["panels"].items()
                    },
                    "max_abs_drift": parity["max_abs_drift"],
                },
                "bench": {
                    "compare": src["bench_path"].as_posix(),
                    "sha256": sha(src["bench_path"]),
                    "items": bench["items"],
                    "bit_identical_items": bench["bit_identical_items"],
                    "p50_ms": [lat["p50"]["old"], lat["p50"]["new"]],
                    "p95_ms": [lat["p95"]["old"], lat["p95"]["new"]],
                    "request_peak_gib": [mem["old"], mem["new"]],
                },
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
        (SPECS / f"dev2-{src['key']}-ra.json", spec_text),
        (OUT / f"{src['name']}.decision.ra.json", decision_text),
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
