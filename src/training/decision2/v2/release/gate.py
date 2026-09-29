"""Release gate: bind the coordinator's decision to one verified package revision (stdlib).

The per-size gate (COORDINATION "Full-autonomy mandate") has a judgement part and
a mechanical part. The coordinator records the judgement in a decision file
(``dev2-release-decision/1``) naming the candidate identity, its same-panel report
and the paired comparison with the tier's own Decision 1.0 model. ``check`` runs
at build time; ``seal`` runs after the verification chain and writes the
``dev2-release-gate/1`` receipt that ``hub collect`` requires.

A decision is ``status: final`` (named ``decided_by``; the default when absent)
or ``status: draft`` (``prepared_by`` release engineering, no decider yet, e.g.
while an independent confirmation is pending). A draft binds the same identity,
report and paired comparison, so it can drive a private build, upload and
verification, but ``seal`` refuses it: nothing enters the collection on a draft.

A size without a Decision 1.0 model (~27B) names ``gate_profile: {"name":
"no-1.0", "reference": <card report key>, "v3_share": 0.9, "types": <path>}``
in its spec (coordinator decision 2026-09-29 11:55). Item 1 then requires
post-key v3 >= v3_share x the reference peer's v3, a human-transfer paired
interval (``card.paired`` = candidate minus the reference) whose upper bound is
not below 0, and every typed-FINAL type ``OK`` in the eval track's
``dev2-gate-types/1`` check of the scored run; its decision must name the
profile and the type check's SHA-256. Specs without the key keep the own-1.0 gate.

A new revision of a released repository (progressive update, coordinator
2026-09-29 16:05 / 17:15) names ``gate_profile: {"name": "successor", ...}``:
item 1 becomes the seven successor-rule items against the CURRENT revision's
scored run, reached through that revision's sealed gate receipt and decision:
R1 paired v3 interval low > 0; R2 human-transfer upper bound >= 0; R3 no type
collapsed; R4 mlx-diag card-eligible macro upper bound >= 0; R5 the tier gates
(own 1.0 low > 0 from ``card.paired``, v3 >= v3_share x the first-tier peer,
human transfer vs that peer upper bound >= 0, no collapse); R6 no overlap
exposure; R7 ``v2.eval.gates public231`` vs the current run is not REGRESSION.
Its decision names the profile, ``current_revision`` and ``evidence_sha256``.
At a size without a Decision 1.0 model the profile's ``tier`` sets ``no_1_0:
true``: R5 then drops the own-1.0 check, ``card.paired`` is the candidate minus
the reference peer and the card keeps the no-1.0 layout. ``c1_postkey`` (the
``SUMMARY.json`` of ``v2/eval/sealed/c1-postkey.sh collect`` on these weights)
adds R8: the JevArena-C1 v1.2 post-key guard vs the tier's registered baseline
must not be REGRESSION (coordinator 2026-09-29 23:40).

  evaluate  print the six gate items with evidence from a work directory
  profile   print item 1 (or the successor items) from a spec, before any upload
  seal      write <work>/receipts/gate.json if every item passes
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from v2.release.layout import sha_file, write_json

DECISION_SCHEMA = "dev2-release-decision/1"
GATE_SCHEMA = "dev2-release-gate/1"
TYPES_SCHEMA = "dev2-gate-types/1"
PAIRED_SCHEMA = "jevarena-v3-paired-aggregate/1"
PUBLIC231_SCHEMA = "dev2-gate-public231/1"
EXPOSURE_SCHEMA = "dev2-overlap-exposure/1"
C1_SUMMARY_SCHEMA = "dev2-c1-postkey/1/summary"
PUBLIC_ALPHA = 0.05
NO_OWN_1_0 = "no-1.0"
SUCCESSOR = "successor"
SUCCESSOR_ITEMS = (
    "1_successor_R1_v3",
    "1_successor_R2_human_transfer",
    "1_successor_R3_no_type_collapsed",
    "1_successor_R4_mlx_diag",
    "1_successor_R5_tier_gates",
    "1_successor_R6_no_overlap_exposure",
    "1_successor_R7_public231",
)
C1_ITEM = "1_successor_R8_c1_postkey"
DECISION_TYPES = ("choice", "noul", "score")
VERIFY_STEPS = (
    "repeat-pre",
    "card-pre",
    "upload",
    "download",
    "tree",
    "post",
    "repeat-post",
    "card-post",
    "readback",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def successor_profile(spec: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    current, tier = value.get("current") or {}, value.get("tier") or {}
    keys = {e["key"] for e in spec["card"]["reports"]}
    share = tier.get("v3_share")
    if (
        not all(value.get(k) for k in ("run", *SUCCESSOR_EVIDENCE))
        or not all(current.get(k) for k in ("gate", "decision", "run"))
        or not re.fullmatch(r"[0-9a-f]{40}", str(current.get("revision", "")))
        or tier.get("reference") not in keys
        or not tier.get("paired")
        or type(share) is not float
        or not 0 < share <= 1
        or not spec["card"].get("paired")
        or tier.get("no_1_0", False) not in (True, False)
        or ("c1_postkey" in value and not value["c1_postkey"])
    ):
        raise ValueError(
            "successor gate_profile needs run, current {revision (40 hex), gate, "
            "decision, run}, paired, types, mlx_paired, exposure, public231, tier "
            "{reference (a card report key), v3_share in (0, 1], paired, optional "
            "no_1_0 true|false}, card.paired and, if given, a c1_postkey summary path"
        )
    return value


SUCCESSOR_EVIDENCE = ("paired", "types", "mlx_paired", "exposure", "public231")


def successor_no_1_0(profile: dict[str, Any] | None) -> bool:
    """A successor profile at a size without a Decision 1.0 model."""
    return bool(
        profile
        and profile.get("name") == SUCCESSOR
        and (profile.get("tier") or {}).get("no_1_0") is True
    )


def evidence_sha256(profile: dict[str, Any]) -> dict[str, str]:
    """The files a successor decision binds by SHA-256."""
    files = {name: profile[name] for name in SUCCESSOR_EVIDENCE}
    files["tier_paired"] = profile["tier"]["paired"]
    files["current_gate"] = profile["current"]["gate"]
    files["current_decision"] = profile["current"]["decision"]
    if profile.get("c1_postkey"):
        files["c1_postkey"] = profile["c1_postkey"]
    return {name: sha_file(Path(path)) for name, path in sorted(files.items())}


def _low_high(ci: Any) -> tuple[float, float]:
    return (ci["low"], ci["high"]) if isinstance(ci, dict) else (ci[0], ci[1])


def _same_path(left: Any, right: Any) -> bool:
    return os.path.normpath(str(left)) == os.path.normpath(str(right))


def _verdicts(types: dict[str, Any]) -> dict[str, str]:
    return {
        kind: (types.get("types", {}).get(kind) or {}).get("verdict", "missing")
        for kind in DECISION_TYPES
    }


def _paired_problems(
    paired: dict[str, Any], left: float, right: float, left_run: Path, right_run: Path
) -> list[str]:
    problems = []
    if paired.get("schema_version") != PAIRED_SCHEMA:
        problems.append("not a v3 paired aggregate")
    point = paired["point"]
    if (
        abs(point["left"]["score"] - left) > 1e-9
        or abs(point["right"]["score"] - right) > 1e-9
    ):
        problems.append("paired scores are not the named reports")
    runs = paired.get("runs") or {}
    if not _same_path(runs.get("left", ""), left_run) or not _same_path(
        runs.get("right", ""), right_run
    ):
        problems.append("paired runs are not the named runs")
    return problems


def _item(check: Any, chain: list[str]) -> dict[str, Any]:
    try:
        passed, evidence, problems = check()
    except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
        return {"passed": False, "evidence": f"unreadable evidence: {error!r}"}
    problems = chain + problems
    return {
        "passed": bool(passed) and not problems,
        "evidence": evidence
        + (f"; problems: {'; '.join(problems)}" if problems else ""),
    }


def successor_items(spec: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Successor-rule items R1-R7 against the current revision's scored run."""
    scored = spec.get("scored") or {}
    predictions = scored.get("predictions_sha256") or {}
    current = profile["current"]
    run, current_run = Path(profile["run"]), Path(current["run"])
    names = SUCCESSOR_ITEMS + ((C1_ITEM,) if profile.get("c1_postkey") else ())
    try:
        chain = []
        if sha_file(run / "REPORT.json") != scored.get("report_sha256"):
            chain.append("candidate run is not the scored run")
        if (
            scored.get("seal_sha256")
            and sha_file(run / "SEAL.json") != scored["seal_sha256"]
        ):
            chain.append("candidate seal is not the scored seal")
        prior = _json(Path(current["gate"]))
        if (
            prior.get("schema") != GATE_SCHEMA
            or prior.get("repo_id") != spec["repo_id"]
            or prior.get("revision") != current["revision"]
        ):
            chain.append(
                "current gate does not seal this repository's current revision"
            )
        if prior.get("decision_sha256") != sha_file(Path(current["decision"])):
            chain.append("current decision is not the one its gate sealed")
        if _json(Path(current["decision"])).get("report_sha256") != sha_file(
            current_run / "REPORT.json"
        ):
            chain.append("current run is not the current revision's scored run")
        v3 = _json(run / "REPORT.json")["v3"]["score"]
        current_v3 = _json(current_run / "REPORT.json")["v3"]["score"]
    except (OSError, KeyError, TypeError, ValueError) as error:
        failed = {
            "passed": False,
            "evidence": f"current-revision chain unreadable: {error!r}",
        }
        return {name: dict(failed) for name in names}

    def r1_r2(axis: str) -> Any:
        def check() -> Any:
            paired = _json(Path(profile["paired"]))
            problems = _paired_problems(paired, v3, current_v3, run, current_run)
            for panel, key in (("typed-final", "typed"), ("css15", "css")):
                left = (paired.get("predictions_sha256") or {}).get("left") or {}
                if panel in predictions and left.get(key) != predictions[panel]:
                    problems.append(
                        f"paired {panel} predictions are not the scored ones"
                    )
            if axis == "v3":
                low, high = _low_high(paired["ci95"])
                delta = paired["point"]["delta"]["score"]
                return (
                    low > 0,
                    f"v3 {v3:.3f} vs current {current_v3:.3f}: {delta:+.2f} [{low:+.2f}, {high:+.2f}]",
                    problems,
                )
            h = paired["axis_ci95"]["H"]["delta"]
            return (
                h["high"] >= 0,
                f"human transfer delta 95% [{h['low']:+.3f}, {h['high']:+.3f}]",
                problems,
            )

        return check

    def r3() -> Any:
        types = _json(Path(profile["types"]))
        verdicts = _verdicts(types)
        problems = (
            []
            if types.get("schema") == TYPES_SCHEMA
            and _same_path(types.get("run", ""), run)
            else ["type check is not on the scored run"]
        )
        ok = all(v == "OK" for v in verdicts.values())
        return (
            ok,
            "types " + ", ".join(f"{k} {v}" for k, v in verdicts.items()),
            problems,
        )

    def r4() -> Any:
        mlx = _json(Path(profile["mlx_paired"]))
        low, high = mlx["bootstrap"]["card_macro_ci95"]
        runs, problems = mlx["runs"], []
        if (
            "mlx-diag" in predictions
            and runs["candidate"]["predictions_sha256"] != predictions["mlx-diag"]
        ):
            problems.append("mlx-diag candidate predictions are not the scored ones")
        if (
            current.get("mlx_predictions")
            and sha_file(Path(current["mlx_predictions"]))
            != runs["released"]["predictions_sha256"]
        ):
            problems.append("mlx-diag comparison is not against the current revision")
        delta = mlx["delta"]["card_macro"]
        return (
            high >= 0,
            f"mlx-diag card-eligible macro {delta:+.4f} [{low:+.4f}, {high:+.4f}]",
            problems,
        )

    def r5() -> Any:
        tier = profile["tier"]
        no_own = successor_no_1_0(profile)
        entry = next(
            e for e in spec["card"]["reports"] if e["key"] == tier["reference"]
        )
        reference_v3 = _json(Path(entry["report"]))["v3"]["score"]
        label = entry.get("label") or tier["reference"]
        own = _json(Path(spec["card"]["paired"]))
        peer = _json(Path(tier["paired"]))
        problems = _paired_problems(
            peer, v3, reference_v3, run, Path(peer.get("runs", {}).get("right", ""))
        )
        if abs(own["point"]["left"]["score"] - v3) > 1e-9:
            problems.append("own 1.0 comparison is not the candidate")
        if no_own and abs(own["point"]["right"]["score"] - reference_v3) > 1e-9:
            problems.append(f"card comparison is not the candidate minus {label}")
        own_low = _low_high(own["ci95"])[0]
        h = peer["axis_ci95"]["H"]["delta"]
        floor = tier["v3_share"] * reference_v3
        types_ok = all(
            v == "OK" for v in _verdicts(_json(Path(profile["types"]))).values()
        )
        checks = {} if no_own else {"own 1.0 low > 0": own_low > 0}
        checks.update(
            {
                f"v3 >= {floor:.3f}": v3 >= floor,
                f"H vs {label} upper >= 0": h["high"] >= 0,
                "no type collapsed": types_ok,
            }
        )
        evidence = (
            "no Decision 1.0 at this size" if no_own else f"own 1.0 low {own_low:+.2f}"
        ) + f"; v3 {v3:.3f} vs {tier['v3_share']:.0%} of {label} " f"{reference_v3:.3f}; H vs {label} [{h['low']:+.3f}, {h['high']:+.3f}]; " + ", ".join(
            f"{k} {'ok' if v else 'FAIL'}" for k, v in checks.items()
        )
        return all(checks.values()), evidence, problems

    def r6() -> Any:
        exposure = _json(Path(profile["exposure"]))
        problems = (
            []
            if exposure.get("schema") == EXPOSURE_SCHEMA
            else ["not an overlap exposure receipt"]
        )
        clean = (
            exposure.get("groups") == []
            and not exposure.get("matched_rows")
            and exposure.get("methods_agree") is True
        )
        files = exposure.get("files") or []
        return (
            clean,
            f"{len(files)} training files, {len(exposure.get('groups') or [])} exposed groups",
            problems,
        )

    def r7() -> Any:
        public = _json(Path(profile["public231"]))
        runs, problems = public.get("runs") or {}, []
        if public.get("schema") != PUBLIC231_SCHEMA:
            problems.append("not a public231 gate receipt")
        if not _same_path(runs.get("left", ""), run) or not _same_path(
            runs.get("right", ""), current_run
        ):
            problems.append("public231 runs are not the candidate and current runs")
        delta = public["left_correct"] - public["right_correct"]
        p = public["mcnemar_exact_p"]
        regression = delta < 0 and p < PUBLIC_ALPHA
        if public.get("verdict") != ("REGRESSION" if regression else "OK"):
            problems.append("stored verdict contradicts the counts")
        return (
            not regression,
            f"public 231 {public['left_correct']} vs {public['right_correct']} ({delta:+d}; McNemar p {p:.3f})",
            problems,
        )

    def r8() -> Any:
        summary = _json(Path(profile["c1_postkey"]))
        rule = summary["item8"]
        problems = []
        if summary.get("schema") != C1_SUMMARY_SCHEMA or summary.get("role") != (
            "successor"
        ):
            problems.append("not the C1 post-key summary of a successor run")
        if (summary.get("model") or {}).get("identity") != (
            spec.get("expected_identity") or {}
        ).get("model_sha256"):
            problems.append("the C1 run scored other weights")
        if not any(c.get("kind") == "item8" for c in summary.get("comparisons", [])):
            problems.append("no item-8 comparison against the tier's baseline")
        low, high = rule["ci95"]
        return (
            rule["verdict"] == "PASS",
            f"JevArena-C1 v1.2 post-key {summary['c1']:.2f} vs {rule['name']}: "
            f"{rule['delta']:+.2f} [{low:+.2f}, {high:+.2f}], p {rule['p']:.3f}, {rule['verdict']}",
            problems,
        )

    checks = (r1_r2("v3"), r1_r2("H"), r3, r4, r5, r6, r7, r8)
    return {name: _item(check, chain) for name, check in zip(names, checks)}


def gate_profile(spec: dict[str, Any]) -> dict[str, Any] | None:
    """The spec's no-1.0 or successor profile, or None for the own Decision 1.0 gate."""
    value = spec.get("gate_profile")
    if value is None:
        return None
    if value.get("name") == SUCCESSOR:
        return successor_profile(spec, value)
    references = {
        e["key"] for e in spec["card"]["reports"] if e.get("role") == "reference"
    }
    share = value.get("v3_share")
    if (
        value.get("name") != NO_OWN_1_0
        or value.get("reference") not in references
        or type(share) is not float
        or not 0 < share <= 1
        or not value.get("types")
    ):
        raise ValueError(
            "gate_profile needs name no-1.0, reference (a card report with role "
            "reference), v3_share in (0, 1] and the types check path"
        )
    return value


def near_first_tier(
    spec: dict[str, Any], profile: dict[str, Any], paired: dict[str, Any]
) -> dict[str, Any]:
    """Item 1 at a size without a Decision 1.0 model (coordinator 2026-09-29 11:55)."""
    entries = spec["card"]["reports"]
    mine = next(e for e in entries if e["role"] == "candidate")
    peer = next(e for e in entries if e["key"] == profile["reference"])
    candidate, reference = _json(Path(mine["report"])), _json(Path(peer["report"]))
    label = peer.get("label") or reference["model"]["label"]
    v3, peer_v3 = candidate["v3"]["score"], reference["v3"]["score"]
    floor = profile["v3_share"] * peer_v3
    h = paired["axis_ci95"]["H"]["delta"]
    types = _json(Path(profile["types"]))
    verdicts = {
        kind: (types.get("types", {}).get(kind) or {}).get("verdict", "missing")
        for kind in DECISION_TYPES
    }
    problems = []
    if sha_file(Path(mine["report"])) != (spec.get("scored") or {}).get(
        "report_sha256"
    ):
        problems.append("candidate report is not the scored report")
    if (
        abs(paired["point"]["left"]["score"] - v3) > 1e-9
        or abs(paired["point"]["right"]["score"] - peer_v3) > 1e-9
    ):
        problems.append(f"paired comparison is not candidate minus {label}")
    run_report = Path(types.get("run", "")) / "REPORT.json"
    if (
        types.get("schema") != TYPES_SCHEMA
        or not run_report.is_file()
        or sha_file(run_report) != (spec.get("scored") or {}).get("report_sha256")
    ):
        problems.append("type check is not on the scored run")
    passed = (
        not problems
        and v3 >= floor
        and h["high"] >= 0
        and all(v == "OK" for v in verdicts.values())
    )
    return {
        "passed": passed,
        "evidence": (
            f"no Decision 1.0 at this size; v3 {v3:.3f} vs {profile['v3_share']:.0%} of "
            f"{label} {peer_v3:.3f} = {floor:.3f}; human transfer minus {label} 95% interval "
            f"[{h['low']:+.3f}, {h['high']:+.3f}]; types "
            + ", ".join(f"{k} {v}" for k, v in verdicts.items())
            + (f"; problems: {'; '.join(problems)}" if problems else "")
        ),
    }


def check(
    spec: dict[str, Any], decision_path: Path, *, final: bool = False
) -> dict[str, Any]:
    """The decision must approve exactly this candidate, report and comparison."""
    decision = _json(decision_path)
    scored = spec.get("scored") or {}
    problems = []
    status = decision.get("status", "final")
    if status not in ("draft", "final"):
        problems.append("status is draft or final")
    elif status == "draft" and (
        not decision.get("prepared_by") or decision.get("decided_by")
    ):
        problems.append("a draft names prepared_by and no decided_by")
    elif final and status != "final":
        problems.append("only a final decision can be sealed")
    if (
        decision.get("schema") != DECISION_SCHEMA
        or decision.get("decision") != "release"
    ):
        problems.append("not a release decision")
    if (
        decision.get("model_name") != spec["model_name"]
        or decision.get("repo_id") != spec["repo_id"]
    ):
        problems.append("decision names a different model or repository")
    if decision.get("identity") != spec["expected_identity"]:
        problems.append("decision names a different checkpoint identity")
    if decision.get("report_sha256") != scored.get("report_sha256"):
        problems.append("decision names a different same-panel report")
    paired = spec["card"].get("paired")
    if not paired or decision.get("paired_sha256") != sha_file(Path(paired)):
        problems.append("decision names a different paired comparison")
    if not decision.get("rationale") or (
        status == "final" and not decision.get("decided_by")
    ):
        problems.append("decision needs a rationale and, when final, decided_by")
    profile = gate_profile(spec)
    if profile and profile["name"] == SUCCESSOR:
        if (
            decision.get("gate_profile") != SUCCESSOR
            or decision.get("current_revision") != profile["current"]["revision"]
            or decision.get("evidence_sha256") != evidence_sha256(profile)
        ):
            problems.append(
                "decision names a different gate profile, current revision or evidence"
            )
    elif profile and (
        decision.get("gate_profile") != NO_OWN_1_0
        or decision.get("types_sha256") != sha_file(Path(profile["types"]))
    ):
        problems.append("decision names a different gate profile or type check")
    if problems:
        raise ValueError(
            "Release decision does not approve this spec: " + "; ".join(problems)
        )
    return decision


def first_items(spec: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Item 1 of the spec's profile, and whom the card tradeoffs compare with."""
    paired = _json(Path(spec["card"]["paired"]))
    profile = gate_profile(spec)
    if successor_no_1_0(profile):
        reference = next(
            e.get("label") or e["key"]
            for e in spec["card"]["reports"]
            if e["key"] == profile["tier"]["reference"]
        )
        return (
            successor_items(spec, profile),
            f"{reference} (no Decision 1.0 at this size)",
        )
    if profile and profile["name"] == SUCCESSOR:
        return successor_items(spec, profile), "own 1.0"
    if profile:
        below = (
            next(
                e.get("label") or e["key"]
                for e in spec["card"]["reports"]
                if e["key"] == profile["reference"]
            )
            + " (no Decision 1.0 at this size)"
        )
        return {
            "1_near_first_tier_no_1_0": near_first_tier(spec, profile, paired)
        }, below
    low = _low_high(paired["ci95"])[0]
    return {
        "1_beats_own_1_0": {
            "passed": low > 0,
            "evidence": f"paired v3 95% interval low {low:+.3f} (sha {sha_file(Path(spec['card']['paired']))[:12]})",
        }
    }, "own 1.0"


def evaluate(work: Path) -> dict[str, Any]:
    receipts = work / "receipts"
    spec = _json(receipts / "spec.json")
    steps = {
        name: _json(receipts / f"{name}.json")
        for name in VERIFY_STEPS
        if (receipts / f"{name}.json").is_file()
    }
    build = _json(receipts / "build.json")
    readback = steps.get("readback", {})
    download = steps.get("download", {})
    # Download receipts written before they carried "passed" count only if they name the upload.
    downloaded = download.get(
        "passed",
        bool(download)
        and download.get("revision") == steps.get("upload", {}).get("revision"),
    )
    first, below = first_items(spec)
    items = {
        **first,
        "2_regressions_disclosed": {
            "passed": readback.get("passed", False)
            and not readback.get("card_problems"),
            "evidence": f"{len(build['card']['tradeoffs'])} results below {below} listed in the card tradeoffs table",
        },
        "3_download_hash_parameters": {
            "passed": bool(downloaded)
            and all(steps.get(s, {}).get("passed") for s in ("tree", "post"))
            and steps.get("post", {}).get("loaded_parameters")
            == build["parameters"]["loaded"],
            "evidence": f"tree {steps.get('tree', {}).get('files')} files re-hashed; loaded {steps.get('post', {}).get('loaded_parameters')}",
        },
        "4_native_examples": {
            "passed": steps.get("post", {}).get("passed", False)
            and steps.get("card-post", {}).get("passed", False),
            "evidence": "examples and the card's Python block reproduced from the download",
        },
        "5_output_consistency": {
            "passed": all(
                steps.get(s, {}).get("passed") for s in ("repeat-pre", "repeat-post")
            )
            and all(
                _json(receipts / f"{s}.json")["passed"]
                for s in ("parity-pre", "parity-post")
                if (receipts / f"{s}.json").is_file()
            ),
            "evidence": "cross-process and pre/post-download answers identical; scored-panel parity where run",
        },
        "6_card_design": {
            "passed": readback.get("passed", False),
            "evidence": f"Hub card {readback.get('card_data', {}).get('license')} / problems {readback.get('card_problems')}",
        },
    }
    decision_path = Path(spec["gate_receipt"]) if spec.get("gate_receipt") else None
    return {
        "items": items,
        "passed": all(item["passed"] for item in items.values()),
        "decision": decision_path
        and {
            "sha256": sha_file(decision_path),
            "status": _json(decision_path).get("status", "final"),
        },
        "spec": spec,
        "steps": steps,
        "build": build,
    }


def seal(work: Path) -> dict[str, Any]:
    result = evaluate(work)
    spec = result["spec"]
    if spec["kind"] != "release" or not spec.get("gate_receipt"):
        raise ValueError("Only release specs with a coordinator decision can be sealed")
    decision = check(spec, Path(spec["gate_receipt"]), final=True)
    if not result["passed"]:
        failing = [name for name, item in result["items"].items() if not item["passed"]]
        raise ValueError(f"Gate items failed: {failing}")
    upload = result["steps"]["upload"]
    gate = {
        "schema": GATE_SCHEMA,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": "release",
        "repo_id": spec["repo_id"],
        "model_name": spec["model_name"],
        "revision": upload["revision"],
        "manifest_sha256": upload["manifest_sha256"],
        "decision_sha256": sha_file(Path(spec["gate_receipt"])),
        "decided_by": decision["decided_by"],
        "items": result["items"],
        "receipts_sha256": {
            name: sha_file(work / "receipts" / f"{name}.json")
            for name in result["steps"]
        },
    }
    profile = gate_profile(spec)
    if profile and profile["name"] == SUCCESSOR:
        gate["gate_profile"] = SUCCESSOR
        gate["supersedes"] = {
            "revision": profile["current"]["revision"],
            "gate_sha256": sha_file(Path(profile["current"]["gate"])),
        }
    write_json(work / "receipts" / "gate.json", gate)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("evaluate", "profile", "seal"))
    parser.add_argument("--work", type=Path)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.command == "profile":
        if not args.spec:
            parser.error("profile needs --spec")
        items, _ = first_items(_json(args.spec))
        passed = all(item["passed"] for item in items.values())
        print(json.dumps({"passed": passed, "items": items}, indent=2))
        sys.exit(0 if passed else 1)
    if not args.work:
        parser.error(f"{args.command} needs --work")
    if args.command == "evaluate":
        result = evaluate(args.work)
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "decision": result["decision"],
                    "items": result["items"],
                },
                indent=2,
            )
        )
        sys.exit(0 if result["passed"] else 1)
    gate = seal(args.work)
    print(json.dumps({"gate": "sealed", "revision": gate["revision"]}))


if __name__ == "__main__":
    main()
