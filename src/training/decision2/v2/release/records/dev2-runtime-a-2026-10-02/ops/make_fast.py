"""Specs and decisions of the speed-up phase A runtime (``runtime/fast.py``) for the Decision 2.0 tiers.

    python3 make_fast.py preview --base SPEC --runtime-source DIR --key KEY --output OUT
    python3 make_fast.py release --tiers 0.6B ... [--kind ra|switch|hotfix] (--out DIR | --check)

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

``--kind switch`` (COORDINATION 2026-10-03 02:38 / 03:18 UTC+8): the runtime-only revision that ships the opt-in
shared-context switch (``runtime/shared_ctx.py``, off by default) with the phase A runtime, from SWITCH_COMMIT, on
top of the current main of SWITCH_TIERS (Kai, Eos, Sol: their phase A revisions; Vega: the 27B release, which
also gains the phase A sentence). Evidence <key>/switch/{parity,bench}; outputs specs/dev2-<key>-ras.json and
<name>.decision.ras.json.

``--kind hotfix`` (COORDINATION 2026-10-03 12:03 UTC+8; ROCm hotfix owner 490b6f72): the runtime-only revision with
the runtime of HOTFIX_COMMIT on top of the current main of HOTFIX_TIERS: ``attn_prep`` launches its kernel once per
tensor (a query projection above 2 GiB failed to compile on ROCm) and a fused decoder layer whose kernels fail runs
its eager forward. Evidence <key>/hotfix/{parity,bench}; the parity carries a third side with every fused kernel
raising (answers-compare-fallback.json, 0 answer changes and 0.0 drift against the old side too). card.speed stays
the current main's: the fix leaves the replayed path unchanged but one launch per attention layer, and the hotfix
bench (old then new on one GPU) is recorded in the decision. Outputs specs/dev2-<key>-rah.json and
<name>.decision.rah.json.

``--kind evict`` (COORDINATION 2026-10-03 15:30 UTC+8; eviction-fix owner 843eb8f6): the runtime-only revision with
the runtime of EVICT_COMMIT on top of the current main of EVICT_TIERS (the hotfix revisions): the HIP graph cache
never destroys a captured graph; once it is full (512 graphs or 4 GiB of outputs), new input shapes run the eager
forward. Evidence <key>/evict/{parity,bench}; the parity carries a third side with a graph cache of 8 graphs
(answers-compare-fullcache.json, 0 answer changes and 0.0 drift against the old side too; full-cache.json shows
the cache full). card.speed stays the current main's and the bench is recorded in the decision. Outputs
specs/dev2-<key>-rae.json and <name>.decision.rae.json.
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
# Phase A with the shared-context switch (9d90afd10) and the integration branch merged.
SWITCH_COMMIT = "9cffe606ce89e19974c7def780a48b64583039ac"
# The switch runtime with the attn_prep fix and the eager fallback of fused layers (fast.py, fast_kernels.py).
HOTFIX_COMMIT = "3c8cd1a84e1cbf9ee8cd3fd44cc0ac4f5f07a987"
HOTFIX_WORKER = "ROCm hotfix owner 490b6f72"
HOTFIX_PREPARED_BY = (
    "Decision 2.0 ROCm hotfix owner 490b6f72 (worktree vllm-sr-dev2-runtime-hotfix, branch "
    "xunzhuo/decision-2-runtime-hotfix) with the runtime phase A tooling of this record"
)
# The hotfix runtime whose HIP graph cache never destroys a captured graph (fast.py).
EVICT_COMMIT = "fbedfded8ce33c7c25a99b8fb1f71f733b2c2c44"
EVICT_WORKER = "eviction-fix owner 843eb8f6"
EVICT_PREPARED_BY = (
    "Decision 2.0 eviction-fix owner 843eb8f6 (worktree vllm-sr-dev2-evict, branch "
    "xunzhuo/decision-2-runtime-evict) with the runtime phase A tooling of this record"
)
EVICT_CACHE = 8
PANELS = {"typed-final": 1600, "css15": 6547, "public231": 231, "mlx-diag": 2275}
MARKER = " Checked on one GPU"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's approval of speed-up phase A (2026-10-02 18:00 "
    "UTC+8, COORDINATION 18:00): runtime-only revisions of the six Decision 2.0 repositories with the exact GPU "
    "fast path, each only with 0 answer changes and 0.0 drift on all four scored panels against the released runtime"
)
PREPARED_BY = "Decision 2.0 runtime phase A worker 2d541b40 (track=runtime-a, worktree vllm-sr-dev2-runtime-a)"
# Lux's switch revision is the 9B owner's (COORDINATION 2026-10-03 08:30), run with this record's tooling.
LUX_SWITCH_WORKER = "9B owner e28aa509"
LUX_SWITCH_PREPARED_BY = (
    "Decision 2.0 9B owner e28aa509 (M10 continuation, Lux-9B publisher; worktree vllm-sr-dev2-9b-m10) with the "
    "runtime phase A tooling of this record"
)
DECIDED_UTC = "2026-10-02T10:00:00Z"
# The current main of each repository and the release work directory on node A that built and sealed it (the org
# worker's card-only revisions; Kai's was built on node E and its receipts copied to the same path on node A; Vega's
# is the 27B release M6-IBxIB2-m50 of 2026-10-02 18:20Z, which superseded the org revision e60bd8e3).
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
        "5c85c127828f4b5dfe0ca95d933be033a8a4caa3",
        "dev2-27b-27bx-M6-IBxIB2-m50-release-20261002T172034Z",
    ),
}
# The current main of the switch tiers and the release work directory on node A that built and sealed it.
SWITCH_TIERS: dict[str, tuple[str, str, str, str]] = {
    "0.6B": (
        "0p6b",
        "Kai",
        "51b7b4740c8c70282648d3849d233a962234e4ec",
        "dev2-ra-0.6B-20261002T112747Z",
    ),
    "0.8B": (
        "0p8b",
        "Eos",
        "1d380452ef703b283438732ec6393dd11d38ff24",
        "dev2-ra-0.8B-20261002T122005Z",
    ),
    "2B": (
        "2b",
        "Sol",
        "6a62b3198f3bcf87259cfc2d72185b4861884394",
        "dev2-ra-2B-20261002T114630Z",
    ),
    # Lux: the 9B M10 release KIB4-a40 (phase A runtime 54303117b); its switch revision is the 9B owner's
    # (COORDINATION 2026-10-03 08:30).
    "9B": (
        "9b",
        "Lux",
        "f3122c7c8abd302326c22220aac4095eb1799f37",
        "dev2-9b-m10-KIB4-a40-release-20261002T151423Z",
    ),
    "27B": TIERS["27B"],
}
# The current main of every repository on 2026-10-03 12:03 UTC+8 and the release work directory on node A that built
# and sealed it (Nox: the 4B owner's wave-7 release; the others: this record's switch revisions, Lux the 9B owner's).
HOTFIX_TIERS: dict[str, tuple[str, str, str, str]] = {
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
# The current main of every repository on 2026-10-03 15:30 UTC+8 (the hotfix revisions) and the release work
# directory on node A that built and sealed it.
EVICT_TIERS: dict[str, tuple[str, str, str, str]] = {
    "0.6B": (
        "0p6b",
        "Kai",
        "d06cf74b1304c6454b9de6e0adb8eaad5ac719fa",
        "dev2-rah-0.6B-20261003T055703Z",
    ),
    "0.8B": (
        "0p8b",
        "Eos",
        "34e2db970f2f49ef431237aa3218e0688a1fe992",
        "dev2-rah-0.8B-20261003T055607Z",
    ),
    "2B": (
        "2b",
        "Sol",
        "23cbe9f96dde4a7e1e6a238d576128bf73c5b09e",
        "dev2-rah-2B-20261003T055610Z",
    ),
    "4B": (
        "4b",
        "Nox",
        "7fc0023a8c51ffaf0f4d4a8f1eb1fb54fa2451cc",
        "dev2-rah-4B-20261003T050415Z",
    ),
    "9B": (
        "9b",
        "Lux",
        "7c6792f7e59dce7a64115bfe750ee636e7a63bd4",
        "dev2-rah-9B-20261003T051039Z",
    ),
    "27B": (
        "27b",
        "Vega",
        "477e90f537eb5bd62e90d5e5361c7b654e69cf45",
        "dev2-rah-27B-20261003T054541Z",
    ),
}
KINDS = {
    "ra": (TIERS, RUNTIME_COMMIT, "", "ra"),
    "switch": (SWITCH_TIERS, SWITCH_COMMIT, "switch", "ras"),
    "hotfix": (HOTFIX_TIERS, HOTFIX_COMMIT, "hotfix", "rah"),
    "evict": (EVICT_TIERS, EVICT_COMMIT, "evict", "rae"),
}
# The third parity side of a kind: the answers it must give identically to the old side.
THIRD_SIDE = {"hotfix": "fallback", "evict": "fullcache"}


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


def sources(tier: str, kind: str = "ra") -> dict:
    """The release that built the current main (spec, sealed gate, final decision) and this record's evidence."""
    tiers, _, evidence, _ = KINDS[kind]
    key, codename, revision, work = tiers[tier]
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
    parity_path = OUT / key / evidence / "parity" / "answers-compare.json"
    bench_path = OUT / key / evidence / "bench" / "compare.json"
    parity, bench = _json(parity_path), _json(bench_path)
    third = THIRD_SIDE.get(kind)
    fallback_path = parity_path.with_name(f"answers-compare-{third or 'fallback'}.json")
    fallback = _json(fallback_path) if third else None
    full_cache_path = parity_path.with_name("full-cache.json")
    full_cache = _json(full_cache_path) if kind == "evict" else None
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
            (
                fallback is None
                or (
                    fallback["passed"] is True
                    and fallback["max_abs_drift"] == 0.0
                    and {
                        panel: (
                            p["prompts"],
                            p["identical_prompts"],
                            p["category_changes"],
                        )
                        for panel, p in fallback["panels"].items()
                    }
                    == {panel: (n, n, 0) for panel, n in PANELS.items()}
                ),
                f"the {third} parity does not cover every scored prompt identically",
            ),
            (
                full_cache is None
                or (
                    full_cache["max_graphs"] == EVICT_CACHE
                    and any(
                        c["cached"] == EVICT_CACHE and c["full"] > 0
                        for c in full_cache["caches"]
                    )
                    and all("evicted" not in c for c in full_cache["caches"])
                ),
                f"the {third} parity did not run with a full cache of {EVICT_CACHE} graphs",
            ),
        )
        if not ok
    ]
    if problems:
        raise ValueError(f"{tier}: {'; '.join(problems)}")
    return {
        "kind": kind,
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
        "fallback_path": fallback_path,
        "fallback": fallback,
        "full_cache_path": full_cache_path,
        "full_cache": full_cache,
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


SWITCH_SENTENCE = (
    " From this revision the runtime also ships an opt-in shared-context switch (share_context, off by default) "
    "that runs the shared input of a multi-question request once instead of once per question, so its answers can "
    "differ slightly from the exact path; with the switch off it was checked with 0 answer changes and 0.0 drift on "
    "every scored prompt and on mlx-diag (10,653 prompts) against the previous runtime."
)


HOTFIX_SENTENCE = (
    " From this revision the fused attention kernel also compiles for very long multi-question inputs on ROCm (a "
    "query projection above 2 GiB, which returned an error before), and a decoder layer whose fused kernels fail to "
    "compile or launch runs its eager forward, which gives the same values; it was checked with 0 answer changes and "
    "0.0 drift on every scored prompt and on mlx-diag (10,653 prompts) against the previous runtime, both with the "
    "fused kernels and with every fused kernel failing."
)


EVICT_SENTENCE = (
    " From this revision the runtime never destroys a captured HIP graph: once its graph cache is full (512 graphs "
    "or 4 GiB of graph outputs), new input shapes run the eager forward, which gives the same values (on ROCm, "
    "evicting large graphs under long, varied traffic could crash the GPU process); it was checked with 0 answer "
    "changes and 0.0 drift on every scored prompt and on mlx-diag (10,653 prompts) against the previous runtime, "
    f"both with the default cache and with a cache of {EVICT_CACHE} graphs."
)


def phase_a_sentence_added(src: dict) -> bool:
    text = src["spec"]["runtime_equivalence"]
    # Lux's main (9B M10 release) already describes the phase A runtime in its own words.
    return (
        src["kind"] == "switch"
        and runtime_sentence(src["key"] == "27b") not in text
        and "the phase A runtime, which replays" not in text
    )


def spec_for(src: dict) -> dict:
    if src["kind"] == "switch":
        return switch_spec_for(src)
    if src["kind"] == "hotfix":
        return hotfix_spec_for(src)
    if src["kind"] == "evict":
        return evict_spec_for(src)
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


def switch_spec_for(src: dict) -> dict:
    old = src["spec"]
    spec = layout.current_ids({k: v for k, v in old.items() if k != "_release"})
    spec["runtime_source"] = mirror(SWITCH_COMMIT)
    spec["card"]["speed"] = {
        "evidence": (src["bench_path"].parent / "bench-new.json").as_posix(),
        "sha256": src["bench"]["new"]["sha256"],
    }
    sentence = SWITCH_SENTENCE
    if phase_a_sentence_added(src):
        sentence = runtime_sentence(src["key"] == "27b") + sentence
    text = spec["runtime_equivalence"]
    if text.count(MARKER) == 1:
        spec["runtime_equivalence"] = text.replace(MARKER, sentence + MARKER)
    else:
        spec["runtime_equivalence"] = text + sentence
    spec["gate_receipt"] = f"{DECISIONS}/{src['name']}.decision.ras.json"
    lat = src["bench"]["latency_ms"]
    spec["_release"] = {
        "runtime_switch": (
            "Runtime-only revision (COORDINATION 2026-10-03 02:38 / 03:18 UTC+8"
            + ("; 08:30" if src["key"] == "9b" else "")
            + f"; {LUX_SWITCH_WORKER if src['key'] == '9b' else 'worker 2d541b40'}): the package "
            f"runtime comes from commit {SWITCH_COMMIT[:9]} (runtime_source): the phase A fast path and the opt-in "
            "shared-context switch (runtime/shared_ctx.py, off by default). Weights, tokenizer, configs, the "
            "vendored training/model sources, card index, assets and remote code are those of the spec that built "
            f"{spec['repo_id']}@{src['revision'][:8]}. card.speed is the new runtime's bench (p50 "
            f"{lat['p50']['old']:.1f} -> {lat['p50']['new']:.1f} ms) and runtime_equivalence gains "
            f"{'the phase A and the switch sentences' if phase_a_sentence_added(src) else 'the switch sentence'}."
        ),
        "replaces_spec": {
            "spec": str(src["spec_path"]),
            "sha256": sha(src["spec_path"]),
        },
        "previous": old.get("_release"),
    }
    return spec


def hotfix_spec_for(src: dict) -> dict:
    old = src["spec"]
    spec = layout.current_ids({k: v for k, v in old.items() if k != "_release"})
    spec["runtime_source"] = mirror(HOTFIX_COMMIT)
    text = spec["runtime_equivalence"]
    if text.count(MARKER) == 1:
        spec["runtime_equivalence"] = text.replace(MARKER, HOTFIX_SENTENCE + MARKER)
    else:
        spec["runtime_equivalence"] = text + HOTFIX_SENTENCE
    spec["gate_receipt"] = f"{DECISIONS}/{src['name']}.decision.rah.json"
    lat = src["bench"]["latency_ms"]
    spec["_release"] = {
        "runtime_hotfix": (
            f"Runtime-only revision (COORDINATION 2026-10-03 12:03 UTC+8; {HOTFIX_WORKER}): the package runtime "
            f"comes from commit {HOTFIX_COMMIT[:9]} (runtime_source): the switch runtime with attn_prep launching its "
            "kernel once per tensor (Triton's ROCm compiler failed on its query / key pointer branch when only the "
            "query projection exceeded 2 GiB) and fused decoder layers falling back to their eager forward when a "
            "kernel fails. Weights, tokenizer, configs, the vendored training/model sources, card index, assets and "
            f"remote code are those of the spec that built {spec['repo_id']}@{src['revision'][:8]}. card.speed stays "
            f"that spec's (the hotfix bench, old then new on one GPU: p50 {lat['p50']['old']:.1f} -> "
            f"{lat['p50']['new']:.1f} ms) and runtime_equivalence gains the hotfix sentence."
        ),
        "replaces_spec": {
            "spec": str(src["spec_path"]),
            "sha256": sha(src["spec_path"]),
        },
        "previous": old.get("_release"),
    }
    return spec


def evict_spec_for(src: dict) -> dict:
    old = src["spec"]
    spec = layout.current_ids({k: v for k, v in old.items() if k != "_release"})
    spec["runtime_source"] = mirror(EVICT_COMMIT)
    text = spec["runtime_equivalence"]
    if text.count(MARKER) == 1:
        spec["runtime_equivalence"] = text.replace(MARKER, EVICT_SENTENCE + MARKER)
    else:
        spec["runtime_equivalence"] = text + EVICT_SENTENCE
    spec["gate_receipt"] = f"{DECISIONS}/{src['name']}.decision.rae.json"
    lat = src["bench"]["latency_ms"]
    spec["_release"] = {
        "runtime_evict": (
            f"Runtime-only revision (COORDINATION 2026-10-03 15:30 UTC+8; {EVICT_WORKER}): the package runtime "
            f"comes from commit {EVICT_COMMIT[:9]} (runtime_source): the hotfix runtime whose HIP graph cache "
            "never destroys a captured graph and runs new input shapes eagerly once it is full (on ROCm, evicting "
            "large graphs from the shared memory pool between large eager batches crashed the GPU process). "
            "Weights, tokenizer, configs, the vendored training/model sources, card index, assets and remote code "
            f"are those of the spec that built {spec['repo_id']}@{src['revision'][:8]}. card.speed stays that "
            f"spec's (the eviction-fix bench, old then new on one GPU: p50 {lat['p50']['old']:.1f} -> "
            f"{lat['p50']['new']:.1f} ms) and runtime_equivalence gains the eviction-fix sentence."
        ),
        "replaces_spec": {
            "spec": str(src["spec_path"]),
            "sha256": sha(src["spec_path"]),
        },
        "previous": old.get("_release"),
    }
    return spec


def runtime_evidence(src: dict) -> dict:
    bench, parity = src["bench"], src["parity"]
    lat, mem = bench["latency_ms"], bench["memory_gib"]["request_peak"]
    return {
        "parity": {
            "compare": src["parity_path"].as_posix(),
            "sha256": sha(src["parity_path"]),
            "panels": {
                panel: [p["prompts"], p["identical_prompts"], p["category_changes"]]
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
    }


def switch_decision_for(src: dict, spec_sha: str) -> dict:
    old, gate, bench = src["decision"], src["gate"], src["bench"]
    old_sha = sha(src["decision_path"])
    repo = layout.current_repo(old["repo_id"])
    lat = bench["latency_ms"]
    fast = phase_a_sentence_added(src)
    new = copy.deepcopy(old)
    new.pop("card_revision", None)
    new.update(
        {
            "repo_id": repo,
            "decided_by": (
                "coordinator (parent agent), Decision 2.0 program: COORDINATION 2026-10-03 02:38 and 03:18 UTC+8 "
                "(the shared-context switch ships opt-in, default off, in runtime-only revisions with the phase A "
                "parity rollout; the user's 01:27 instruction 'same strategy as when private' covers the public "
                "repositories)"
                + (
                    "; COORDINATION 2026-10-03 08:30 UTC+8 (the Lux switch revision is the 9B owner's when no 9B "
                    "release is under way by 10:00)"
                    if src["key"] == "9b"
                    else ""
                )
            ),
            "prepared_by": (
                LUX_SWITCH_PREPARED_BY if src["key"] == "9b" else PREPARED_BY
            ),
            "decided_utc": (
                "2026-10-03T00:30:00Z" if src["key"] == "9b" else "2026-10-02T19:18:00Z"
            ),
            "action": (
                f"Runtime-only revision of the public repository {repo}: the package runtime decision2/*.py comes "
                f"from commit {SWITCH_COMMIT[:9]}"
                + (
                    ", which on a ROCm GPU under Transformers 5.17 replays the backbone of each exact padded shape "
                    "as a HIP graph, casts shared Linear inputs to BF16 once and (on gfx942) runs fused Triton "
                    "element-wise kernels that round exactly as the ops they replace, and"
                    if fast
                    else " (the phase A fast path), which"
                )
                + " adds the opt-in shared-context switch decision2/shared_ctx.py (share_context, off by default). "
                "Weights, tokenizer, configs, the vendored training/model sources, calibration, any Score offsets, "
                f"the banner and the charts are byte-identical to the released revision {gate['revision']}; README.md "
                "states the new median latency and MODEL_MANIFEST.json the new runtime hashes and the runtime "
                "sentences. The collection is not changed."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision. Only the package runtime changes, and with the switch off (the default) it "
                "computes the same values bit for bit: every prompt of the four scored panels (typed-final 1,600, "
                "css15 6,547, public231 231, mlx-diag 2,275) answered through the released package and through this "
                "package on the released weights gave 0 answer changes and 0.0 drift; 400 single requests: p50 "
                f"{lat['p50']['old']:.2f} -> {lat['p50']['new']:.2f} ms, p95 {lat['p95']['old']:.2f} -> "
                f"{lat['p95']['new']:.2f} ms, all bit-identical. release.sh checks the native examples, the card's "
                "Transformers example before upload, after the real download and from the Hub in fresh environments "
                "under Transformers 5.17 and 5.18, the card structure and every card link; then the downloaded "
                "package must pass the Index harness's 86-request parity gate (the kit runner against the "
                "package's own entry point; the phase A revisions of Kai, Eos and Sol pass it with 0.0 drift)."
            ),
            "previous_rationale": old["rationale"],
            "runtime_revision": {
                "kind": (
                    "fast-path-shared-context-runtime"
                    if fast
                    else "shared-context-switch-runtime"
                ),
                "runtime_commit": SWITCH_COMMIT,
                "spec": f"v2/release/specs/dev2-{src['key']}-ras.json",
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


def hotfix_decision_for(src: dict, spec_sha: str) -> dict:
    old, gate, bench, fallback = (
        src["decision"],
        src["gate"],
        src["bench"],
        src["fallback"],
    )
    old_sha = sha(src["decision_path"])
    repo = layout.current_repo(old["repo_id"])
    lat = bench["latency_ms"]
    new = copy.deepcopy(old)
    new.pop("card_revision", None)
    new.update(
        {
            "repo_id": repo,
            "decided_by": (
                "coordinator (parent agent), Decision 2.0 program: COORDINATION 2026-10-03 12:03 UTC+8 (the ROCm "
                "hotfix owner fixes the _attn_prep_kernel compile failure the Index submission's release spot check "
                "found, adds a byte-identical eager fallback for fused kernels and ships runtime-only revisions "
                "through the phase A parity rollout; the user's 11:34 instruction makes the Jev Decision Index "
                "submission the top priority)"
            ),
            "prepared_by": HOTFIX_PREPARED_BY,
            "decided_utc": "2026-10-03T04:03:00Z",
            "action": (
                f"Runtime-only revision of the public repository {repo}: decision2/fast.py and "
                f"decision2/fast_kernels.py come from commit {HOTFIX_COMMIT[:9]}, in which the fused attention-prep "
                "kernel runs once per tensor (on ROCm it failed to compile, and the request returned an error, when "
                "a long multi-question input made the query projection exceed 2 GiB) and a decoder layer whose fused "
                "kernels fail to compile or launch runs its eager forward. Weights, tokenizer, configs, the vendored "
                "training/model sources, calibration, any Score offsets, the banner and the charts are "
                f"byte-identical to the released revision {gate['revision']}, and so is README.md; "
                "MODEL_MANIFEST.json states the new runtime hashes and the runtime sentences. The collection is not "
                "changed."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision. Only the package runtime changes, and it computes the same values bit for bit: "
                "every prompt of the four scored panels (typed-final 1,600, css15 6,547, public231 231, mlx-diag "
                "2,275) answered through the released package and through this package on the released weights gave "
                "0 answer changes and 0.0 drift, and so did this package with every fused kernel made to fail (each "
                f"decoder layer on its eager fallback); 400 single requests: p50 {lat['p50']['old']:.2f} -> "
                f"{lat['p50']['new']:.2f} ms, p95 {lat['p95']['old']:.2f} -> {lat['p95']['new']:.2f} ms, all "
                "bit-identical. release.sh checks the native examples, the card's Transformers example before "
                "upload, after the real download and from the Hub in fresh environments under Transformers 5.17 "
                "and 5.18, the card structure and every card link; then the downloaded package must pass the Index "
                "harness's 86-request parity gate."
            ),
            "previous_rationale": old["rationale"],
            "runtime_revision": {
                "kind": "attn-prep-fix-eager-fallback-runtime",
                "runtime_commit": HOTFIX_COMMIT,
                "spec": f"v2/release/specs/dev2-{src['key']}-rah.json",
                "spec_sha256": spec_sha,
                **runtime_evidence(src),
                "fallback_parity": {
                    "compare": src["fallback_path"].as_posix(),
                    "sha256": sha(src["fallback_path"]),
                    "panels": {
                        panel: [
                            p["prompts"],
                            p["identical_prompts"],
                            p["category_changes"],
                        ]
                        for panel, p in fallback["panels"].items()
                    },
                    "max_abs_drift": fallback["max_abs_drift"],
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


def evict_decision_for(src: dict, spec_sha: str) -> dict:
    old, gate, bench, full = (
        src["decision"],
        src["gate"],
        src["bench"],
        src["fallback"],
    )
    old_sha = sha(src["decision_path"])
    repo = layout.current_repo(old["repo_id"])
    lat = bench["latency_ms"]
    new = copy.deepcopy(old)
    new.pop("card_revision", None)
    new.update(
        {
            "repo_id": repo,
            "decided_by": (
                "coordinator (parent agent), Decision 2.0 program: COORDINATION 2026-10-03 15:30 UTC+8 (the "
                "inference owner 885d85cc found that evicting captured HIP graphs crashes the GPU process on ROCm "
                "and fixed its native engine by never evicting; the released runtime carries the same eviction, so "
                "the eviction-fix owner ports that policy and ships runtime-only revisions of all six repositories "
                "through the phase A parity rollout; the user's full-autonomy mandate)"
            ),
            "prepared_by": EVICT_PREPARED_BY,
            "decided_utc": "2026-10-03T07:30:00Z",
            "action": (
                f"Runtime-only revision of the public repository {repo}: decision2/fast.py comes from commit "
                f"{EVICT_COMMIT[:9]}, whose HIP graph cache never destroys a captured graph: once 512 graphs or "
                "4 GiB of graph outputs are captured, new input shapes run the eager forward (on ROCm, evicting "
                "large graphs from the shared memory pool between large eager batches crashed the GPU process "
                "under Index traffic). Weights, tokenizer, configs, the vendored training/model sources, "
                "calibration, any Score offsets, the banner and the charts are byte-identical to the released "
                f"revision {gate['revision']}, and so is README.md; MODEL_MANIFEST.json states the new runtime "
                "hashes and the runtime sentences. The collection is not changed."
            ),
            "rationale": (
                f"The release judgement of the superseded final decision {old_sha[:8]}… stands unchanged: the same "
                f"identity {old['identity']['model_sha256'][:8]}, scored report, paired comparison, calibration and "
                "licence decision. Only the package runtime changes, and it computes the same values bit for bit: "
                "every prompt of the four scored panels (typed-final 1,600, css15 6,547, public231 231, mlx-diag "
                "2,275) answered through the released package and through this package on the released weights gave "
                f"0 answer changes and 0.0 drift, and so did this package with its graph cache cut to {EVICT_CACHE} "
                "graphs (every other input shape on the eager forward); 400 single requests: p50 "
                f"{lat['p50']['old']:.2f} -> {lat['p50']['new']:.2f} ms, p95 {lat['p95']['old']:.2f} -> "
                f"{lat['p95']['new']:.2f} ms, all bit-identical. release.sh checks the native examples, the card's "
                "Transformers example before upload, after the real download and from the Hub in fresh environments "
                "under Transformers 5.17 and 5.18, the card structure and every card link; then the downloaded "
                "package must pass the Index harness's 86-request parity gate."
            ),
            "previous_rationale": old["rationale"],
            "runtime_revision": {
                "kind": "no-graph-eviction-runtime",
                "runtime_commit": EVICT_COMMIT,
                "spec": f"v2/release/specs/dev2-{src['key']}-rae.json",
                "spec_sha256": spec_sha,
                **runtime_evidence(src),
                "full_cache_parity": {
                    "compare": src["fallback_path"].as_posix(),
                    "sha256": sha(src["fallback_path"]),
                    "max_graphs": EVICT_CACHE,
                    "cache": src["full_cache_path"].as_posix(),
                    "cache_sha256": sha(src["full_cache_path"]),
                    "panels": {
                        panel: [
                            p["prompts"],
                            p["identical_prompts"],
                            p["category_changes"],
                        ]
                        for panel, p in full["panels"].items()
                    },
                    "max_abs_drift": full["max_abs_drift"],
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


def decision_for(src: dict, spec_sha: str) -> dict:
    if src["kind"] == "switch":
        return switch_decision_for(src, spec_sha)
    if src["kind"] == "hotfix":
        return hotfix_decision_for(src, spec_sha)
    if src["kind"] == "evict":
        return evict_decision_for(src, spec_sha)
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


def texts(tier: str, kind: str = "ra") -> list[tuple[Path, str]]:
    src = sources(tier, kind)
    suffix = KINDS[kind][3]
    spec_text = json.dumps(spec_for(src), ensure_ascii=False, indent=2) + "\n"
    spec_sha = hashlib.sha256(spec_text.encode()).hexdigest()
    decision_text = (
        json.dumps(decision_for(src, spec_sha), ensure_ascii=False, indent=2) + "\n"
    )
    return [
        (SPECS / f"dev2-{src['key']}-{suffix}.json", spec_text),
        (OUT / f"{src['name']}.decision.{suffix}.json", decision_text),
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
    r.add_argument("--kind", choices=sorted(KINDS), default="ra")
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
        if tier not in KINDS[args.kind][0]:
            raise SystemExit(f"{tier} has no {args.kind} revision")
        for path, text in texts(tier, args.kind):
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
