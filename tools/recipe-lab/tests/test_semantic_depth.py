"""The Speed depth cutoff: standalone analysis and the planner bridge.

Standalone (sub-issue 7): the recipes-free cutoff surface - threshold site
inventory and the protected-score proposal. Planner bridge (sub-issue 9): a
graded Speed parent leads to a distinct candidate, a public probe and a v2
schedule card.
"""

from __future__ import annotations

import copy
import io
from contextlib import redirect_stderr
from unittest.mock import patch

import pytest
import yaml
from recipe_lab.loop_evidence import digest
from recipe_lab.semantic_depth import propose_threshold, score_value, threshold_sites
from tests.fixtures import DEPTH_FAST, DEPTH_PUBLIC, DEPTH_STRONG, depth_doc

import test_loop as fixtures
import test_planner
from test_planning_probe import sample
from test_speed_draft import compiled


@pytest.fixture
def doc():
    document = depth_doc()
    document["recipes"].append({"name": "unrelated", "strategy": "priority"})
    return document


@pytest.fixture
def site(doc):
    (found,) = threshold_sites(doc["recipes"][0], DEPTH_STRONG, DEPTH_FAST)
    return found


@pytest.mark.parametrize(
    "value,expected",
    [(0, True), (1, True), (0.5, True), (-0.1, False), (1.0001, False), (True, False)],
)
def test_score_value_bounds(value, expected):
    assert bool(score_value(value)) is expected


def test_inventory_reports_the_supported_edit(site):
    assert site["feature"] == "semantic_depth"
    assert site["signal"] == "depth" and site["before"] == 0.6
    assert site["primary"] == "reasoning"
    assert site["overlap"] == "tools" and site["fallback"] == "fast"
    assert site["bindings"] == {
        "reasoning": [DEPTH_STRONG],
        "tools": [DEPTH_STRONG],
        "fast": [DEPTH_FAST],
    }


def test_unsupported_shapes_return_no_sites():
    recipe = depth_doc()["recipes"][0]
    assert threshold_sites(recipe, DEPTH_STRONG, DEPTH_STRONG) == []
    changed = copy.deepcopy(recipe)
    changed["strategy"] = "RoundRobin"
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["candidate_requirements"]["capabilities"] = "any"
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["decisions"][0]["priority"] = 2
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["decisions"][1]["algorithm"] = {"type": "loop"}
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["decisions"][2]["rules"] = changed["decisions"][1]["rules"]
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["decisions"][1]["modelRefs"] = [{"model": "C"}]
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["signals"]["systemone"][0]["predicate"] = {"gte": 1.5}
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []
    changed = copy.deepcopy(recipe)
    changed["projections"] = [{"name": "x"}]
    assert threshold_sites(changed, DEPTH_STRONG, DEPTH_FAST) == []


def test_graded_scores_protect_every_retained_reasoning_observation(doc, site):
    draft, predicate = propose_threshold(
        doc, DEPTH_PUBLIC, site, [0.62, 0.71], [0.83, 0.94]
    )
    assert predicate["after"] == (0.71 + 0.83) / 2
    # the unrelated recipe is preserved verbatim; only the signal moves
    assert draft["recipes"][1] == doc["recipes"][1]
    assert draft["recipes"][0] is not doc["recipes"][0]
    rendered = yaml.safe_load(
        yaml.safe_dump(draft["recipes"][0], sort_keys=False, allow_unicode=True)
    )
    moved = rendered["signals"]["systemone"][0]["predicate"]["gte"]
    assert moved == (0.71 + 0.83) / 2
    assert rendered["decisions"] == doc["recipes"][0]["decisions"]
    assert propose_threshold(doc, DEPTH_PUBLIC, site, [0.83], [0.83]) is None
    assert propose_threshold(doc, DEPTH_PUBLIC, site, [0.7], []) is None
    assert propose_threshold(doc, DEPTH_PUBLIC, site, [0.65], [0.5]) is None


def test_site_mismatch_is_refused(doc, site):
    other_site = {**site, "before": 0.2}
    with pytest.raises(ValueError, match="site differs"):
        propose_threshold(doc, DEPTH_PUBLIC, other_site, [0.4], [0.8])


# ---- The planner bridge: a graded Speed parent, its probe and the v2 card ----


@pytest.fixture
def lab():
    env = test_planner.PlannerTest()
    env.setUp()
    recipe = compiled()["recipes"][1]
    recipe["name"] = "main"
    for lane in recipe["decisions"]:
        lane["algorithm"] = {"type": "static"}
        lane["modelRefs"] = [{"model": "B" if lane["name"] == "fast" else "A"}]
    env.doc["recipes"][1] = recipe
    fixtures.put(env.config, env.doc)
    try:
        yield env
    finally:
        env.doCleanups()


def measurement(env, observations, arm, task, scores):
    if arm != "fixed":
        env.run_fixture.arm(arm, "A", 12.5)
    make_rows = test_planner.receipts

    def with_decisions(counts):
        rows = make_rows(counts)
        for reply, score in zip(rows[1::2], scores):
            strong = score is None or score >= 0.60
            model = "A" if strong else "B"
            reply.update(
                selected_decision="reasoning" if strong else "fast",
                selected_model=model,
                returned_model=model,
                observed_score=score,
            )
        return rows

    with patch.object(test_planner, "receipts", with_decisions):
        ident = env.measured(arm, counts=[0] * len(scores), task=task)
    trace = env.root / arm / "dsh/runs" / task / arm / "main/requests.jsonl"
    for call, score in enumerate(scores, 1):
        observations.append(
            dict(
                measurement=ident,
                task=task,
                trace_sha256=digest(trace.read_bytes()),
                call=call,
                score=score,
                matched_other_decisions=[],
                fast_eligible=True,
            )
        )
    return ident


def inputs(lab):
    obs = []
    base = measurement(lab, obs, "fixed", "task", [0.63, 0.85, 0.44])
    guards = [
        measurement(lab, obs, "repeat" + str(i), "guard", [score])
        for i, score in enumerate([0.91, 0.94])
    ]
    evidence = lab.root / "scores.json"
    fixtures.put(
        evidence,
        dict(
            schema="recipe-lab-depth-evidence-v1",
            signal="depth",
            public_model="public",
            base_config_sha256=digest(lab.config.read_bytes()),
            observations=obs,
        ),
    )
    return (
        base,
        [
            "--depth-evidence",
            evidence,
            "--prefixes",
            guards[0],
            "--prefixes",
            guards[1],
        ],
        evidence,
    )


def gateway(parent, candidate):
    probe = sample("real")
    probe.update(
        public_model="public",
        source_sha=candidate,
        projected_sha=candidate,
        parent_source_sha=parent,
    )
    probe["probes"] = [
        r
        for r in probe["probes"]
        if r["case"] in {"unsupported-image", "output-exceeds-both"}
    ]
    for case, score, other, before, after in [
        ("depth-changed", 0.63, [], "reasoning", "fast"),
        ("depth-protected", 0.85, [], "reasoning", "reasoning"),
        ("depth-overlap", 0.67, ["tools"], "reasoning", "tools"),
        ("depth-missing", None, [], "reasoning", "reasoning"),
        ("tool-call", 0.15, ["tools"], "tools", "tools"),
        ("tool-result", 0.18, ["tools"], "tools", "tools"),
    ]:
        meta = dict(
            request_sha256=digest(case.encode()),
            score=score,
            matched_other_decisions=other,
            fast_eligible=True,
            http=200,
            done=True,
        )
        row = dict(
            **meta,
            case=case,
            roles=["user"],
            decision=after,
            selected="B" if after == "fast" else "A",
            returned="B" if after == "fast" else "A",
            parent=dict(**meta, decision=before, selected="A", returned="A"),
        )
        if case.startswith("tool-"):
            row["tool_ok"] = True
        if case == "tool-result":
            row["roles"].extend(["assistant", "tool"])
        probe["probes"].append(row)
    return probe


def test_observed_speed_threshold_enrolls_in_native_schedule(lab):
    base, args, evidence = inputs(lab)
    (site,) = lab.inventory()["inventory"]["semantic_thresholds"]
    assert site["signal"] == "depth"
    assert lab.plan(base)["next"]["action"] == "collect_depth_observations"
    candidate = lab.root / "candidate.yaml"
    plan = lab.plan(base, *args, "--out-candidate", candidate)
    assert plan["next"]["action"] == "probe_public_route"
    assert plan["next"]["affected_parent_calls"] == [1]
    assert plan["next"]["observed_seconds"] == 2
    expected = copy.deepcopy(lab.doc)
    expected["recipes"][1]["signals"]["systemone"][1]["predicate"]["gte"] = (
        0.63 + 0.85
    ) / 2
    assert yaml.safe_load(candidate.read_bytes()) == expected
    receipt = lab.root / "probe.json"
    probe = gateway(digest(lab.config.read_bytes()), digest(candidate.read_bytes()))
    fixtures.put(receipt, {**probe, "kind": "fixture", "backend_mode": "fixture"})
    assert (
        lab.plan(base, *args, "--probe-receipt", receipt)["next"]["action"]
        == "probe_real_backends"
    )
    fixtures.put(receipt, probe)
    plan = lab.plan(base, *args, "--probe-receipt", receipt)
    assert plan["next"]["action"] == "plan_paired_development"
    saved = lab.root / "plan.json"
    fixtures.put(saved, plan)
    card = lab.root / "card.json"
    fixtures.put(
        card,
        dict(
            schema="recipe-lab-schedule-v2",
            study="threshold",
            based_on=base,
            signature=saved.name,
            signature_sha256=digest(saved.read_bytes()),
            candidate=candidate.name,
            target_task="task",
            guard_task="guard",
            budget_episodes=4,
            repeats=1,
            hypotheses=["midrange fast saves time", "fast adds extra turns"],
            prediction="retain guard",
            decisions={"advance": "confirm", "revise": "measure", "drop": "retain"},
        ),
    )
    assert lab.cli("schedule", "--card", card)["next"]["task"] == "guard"
    packet = fixtures.get(evidence)
    packet["observations"][0]["score"] = 0.64
    fixtures.put(evidence, packet)
    with redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        lab.cli("schedule", "--card", card)


def test_parent_identity_and_independent_guards(lab):
    base, args, evidence = inputs(lab)
    packet = fixtures.get(evidence)
    for field, bad in (
        ("trace_sha256", "1" * 64),
        ("call", 100),
        ("score", 0.25),
        ("matched_other_decisions", ["not-a-lane"]),
    ):
        broken = copy.deepcopy(packet)
        broken["observations"][0][field] = bad
        fixtures.put(evidence, broken)
        with redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            lab.plan(base, *args)
    packet["observations"][-1]["score"] = 0.63
    fixtures.put(evidence, packet)
    candidate = lab.root / "candidate.yaml"
    lab.plan(base, *args, "--out-candidate", candidate)
    receipt = lab.root / "probe.json"
    probe = gateway(digest(lab.config.read_bytes()), digest(candidate.read_bytes()))
    fixtures.put(receipt, probe)
    assert (
        lab.plan(base, *args, "--probe-receipt", receipt)["next"]["action"]
        == "verify_guard_repeatability"
    )
    changed = next(r for r in probe["probes"] if r["case"] == "depth-changed")
    changed["parent"]["request_sha256"] = "0" * 64
    fixtures.put(receipt, probe)
    assert (
        lab.plan(base, *args, "--probe-receipt", receipt)["next"]["action"]
        == "diagnose_gateway_probe"
    )


def test_unscored_parent_and_unsupported_shape_never_gain_a_candidate(lab):
    base, args, evidence = inputs(lab)
    packet = fixtures.get(evidence)
    for row in packet["observations"]:
        row["fast_eligible"] = False
    fixtures.put(evidence, packet)
    assert not lab.plan(base, *args)["ranked_probes"]
    lab.doc["recipes"][1]["decisions"][1]["rules"]["on_unknown"] = "no_match"
    fixtures.put(lab.config, lab.doc)
    assert not lab.inventory()["inventory"].get("semantic_thresholds")


def test_graded_scores_protect_retained_reasoning_on_the_speed_parent(lab):
    (site,) = lab.inventory()["inventory"]["semantic_thresholds"]
    draft, predicate = propose_threshold(
        lab.doc, "public", site, [0.62, 0.71], [0.83, 0.94]
    )
    assert predicate["after"] == (0.71 + 0.83) / 2
    assert draft["recipes"][0] == lab.doc["recipes"][0]
    assert propose_threshold(lab.doc, "public", site, [0.83], [0.83]) is None
    assert propose_threshold(lab.doc, "public", site, [0.7], []) is None
    other_site = {**site, "before": 0.2}
    with pytest.raises(ValueError, match="site differs"):
        propose_threshold(lab.doc, "public", other_site, [0.4], [0.8])
