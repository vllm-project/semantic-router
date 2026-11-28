"""AG6: the session-policy proposer - cause-keyed one-variable candidates
(Z6), and the golden flow (an out-of-lock flap diagnosis produces the hold
candidates, and the raised candidate switches less on replay)."""

from __future__ import annotations

import copy
import sys

import pytest
import yaml
from recipe_lab.proposers.session_policy import (
    propose_session_policy,
    session_policy_handoffs,
)

BASE_CONFIG = {
    "version": "v0.2",
    "global": {
        "router": {
            "learning": {
                "adaptation": {"enabled": False},
                "protection": {
                    "enabled": True,
                    "scope": "conversation",
                    "identity": {
                        "headers": {
                            "session": "x-session-id",
                            "conversation": "x-conversation-id",
                        }
                    },
                    "tuning": {
                        "stability_weight": 1.0,
                        "switch_margin": 0.05,
                        "min_turns_before_switch": 1,
                    },
                },
            }
        }
    },
}

CONFIG_WITHOUT_IDENTITY = copy.deepcopy(BASE_CONFIG)
del CONFIG_WITHOUT_IDENTITY["global"]["router"]["learning"]["protection"]["identity"]

# identity missing: every in-loop switch is one the protection never saw
IDENTITY_FLAP_SUMMARY = {
    "records": 3,
    "sessions": 1,
    "model_switches": 2,
    "tool_loop_switches": 2,
    "decision_drift_switches": 0,
    "selection_rejected": 0,
    "tool_loop_interrupted": 0,
    "mean_switches_per_session": 2.0,
}

# identity present: the loop hard-locked, the post-loop request flapped
OUT_OF_LOCK_SUMMARY = {
    "records": 4,
    "sessions": 1,
    "model_switches": 1,
    "tool_loop_switches": 0,
    "decision_drift_switches": 0,
    "selection_rejected": 0,
    "tool_loop_interrupted": 0,
    "mean_switches_per_session": 1.0,
}

# every in-loop switch was a drift release (in-design, #5567)
DRIFT_SUMMARY = {
    "records": 6,
    "sessions": 2,
    "model_switches": 2,
    "tool_loop_switches": 2,
    "decision_drift_switches": 2,
    "selection_rejected": 0,
    "tool_loop_interrupted": 0,
    "mean_switches_per_session": 1.0,
}

NOT_SEPARATE_DIAGNOSIS = {
    "agentic": {
        "agent_separation": {
            "status": "not_separate",
            "agent_requests": 0,
            "requests_missing_identity": 4,
            "reason": "protection identity missing on 4 request(s)",
        }
    }
}


def _variables_of(candidates):
    return [c.variable for c in candidates]


def test_no_evidence_proposes_nothing():
    assert propose_session_policy(BASE_CONFIG) == []
    assert propose_session_policy(BASE_CONFIG, replay_summary={}) == []
    assert session_policy_handoffs(BASE_CONFIG, replay_summary={}) == []


def test_unprotected_switches_propose_identity_headers():
    # Z6: switches the protection never saw are an identity problem (W1/W2
    # class), not a tuning-knob problem
    candidates = propose_session_policy(
        CONFIG_WITHOUT_IDENTITY, replay_summary=IDENTITY_FLAP_SUMMARY
    )
    variables = _variables_of(candidates)
    assert variables == ["identity.headers"]
    candidate = candidates[0]
    assert candidate.trigger["reason"] == "unprotected_tool_loop_switches"
    assert candidate.new == {
        "session": "x-session-id",
        "conversation": "x-conversation-id",
    }
    # exactly one variable changed in the candidate config: the whole
    # identity block is new (the base had none), per the leaf-level diff
    changed = _diff(CONFIG_WITHOUT_IDENTITY, candidate.config)
    assert changed == [("global.router.learning.protection", "identity")]
    protection = candidate.config["global"]["router"]["learning"]["protection"]
    assert protection["identity"]["headers"]["session"] == "x-session-id"
    # the hold knobs stay untouched - the hard lock, not stability_weight,
    # is what governs the tool loop
    assert (
        protection["tuning"]
        == BASE_CONFIG["global"]["router"]["learning"]["protection"]["tuning"]
    )
    # a partially configured identity only adds the missing header
    partial = copy.deepcopy(CONFIG_WITHOUT_IDENTITY)
    partial["global"]["router"]["learning"]["protection"]["identity"] = {
        "headers": {"session": "x-session-id"}
    }
    candidates = propose_session_policy(partial, replay_summary=IDENTITY_FLAP_SUMMARY)
    assert candidates[0].new == {"conversation": "x-conversation-id"}


def test_configured_identity_hands_off_to_the_client():
    # the headers are mapped but the switches still happened unprotected:
    # the client is not sending them (W1/W2) - a handoff, never a knob
    candidates = propose_session_policy(
        BASE_CONFIG, replay_summary=IDENTITY_FLAP_SUMMARY
    )
    assert candidates == []
    handoffs = session_policy_handoffs(
        BASE_CONFIG, replay_summary=IDENTITY_FLAP_SUMMARY
    )
    assert [h["reason"] for h in handoffs] == ["unprotected_tool_loop_switches"]
    assert "W1/W2" in handoffs[0]["note"]
    # with the headers still unmapped the candidate path applies instead -
    # no handoff yet
    assert (
        session_policy_handoffs(
            CONFIG_WITHOUT_IDENTITY, replay_summary=IDENTITY_FLAP_SUMMARY
        )
        == []
    )


def test_out_of_lock_flaps_raise_margin_and_stability():
    # Z6: the hold knobs govern ordinary (out-of-lock) scoring only - both
    # candidates trigger together and each changes exactly one variable
    candidates = propose_session_policy(BASE_CONFIG, replay_summary=OUT_OF_LOCK_SUMMARY)
    assert _variables_of(candidates) == ["stability_weight", "switch_margin"]
    stability, margin = candidates
    assert stability.trigger["reason"] == "model_switches"
    assert stability.old == 1.0
    assert stability.new == 2.0
    assert "hard lock" in stability.trigger["note"]
    assert margin.old == 0.05
    assert margin.new == 0.1
    assert _diff(BASE_CONFIG, stability.config) == [
        ("global.router.learning.protection.tuning", "stability_weight")
    ]
    assert _diff(BASE_CONFIG, margin.config) == [
        ("global.router.learning.protection.tuning", "switch_margin")
    ]


def test_not_separate_separation_adds_the_agent_header():
    candidates = propose_session_policy(BASE_CONFIG, diagnosis=NOT_SEPARATE_DIAGNOSIS)
    assert "identity.headers.agent" in _variables_of(candidates)
    candidate = next(c for c in candidates if c.variable == "identity.headers.agent")
    protection = candidate.config["global"]["router"]["learning"]["protection"]
    assert protection["identity"]["headers"]["agent"] == "x-claude-code-agent-id"
    assert (
        "agent"
        not in BASE_CONFIG["global"]["router"]["learning"]["protection"]["identity"][
            "headers"
        ]
    )


def test_drift_releases_are_a_calibration_handoff_not_a_knob():
    # Z6/#5567: the drift transfer is in-design - over the per-task ceiling
    # it points at threshold calibration, never at a protection knob
    candidates = propose_session_policy(BASE_CONFIG, replay_summary=DRIFT_SUMMARY)
    assert candidates == []
    handoffs = session_policy_handoffs(BASE_CONFIG, replay_summary=DRIFT_SUMMARY)
    assert [h["reason"] for h in handoffs] == ["decision_drift_rate"]
    assert handoffs[0]["decision_drift_switches"] == 2
    assert handoffs[0]["sessions"] == 2
    assert "propose-calibrated" in handoffs[0]["note"]


def test_rejected_selections_hand_off_to_the_candidate_set_check():
    # Z6: rejected selections are a candidate-set/lock question, not
    # min_turns_before_switch
    summary = dict(
        OUT_OF_LOCK_SUMMARY,
        model_switches=0,
        tool_loop_switches=0,
        mean_switches_per_session=0.0,
        selection_rejected=3,
    )
    candidates = propose_session_policy(BASE_CONFIG, replay_summary=summary)
    assert _variables_of(candidates) == []
    handoffs = session_policy_handoffs(BASE_CONFIG, replay_summary=summary)
    assert [h["reason"] for h in handoffs] == ["selection_rejected"]
    assert "candidate set" in handoffs[0]["note"]
    assert "min_turns_before_switch" in handoffs[0]["note"]


def test_exhausted_knobs_produce_no_candidate():
    raised = yaml.safe_load(yaml.safe_dump(BASE_CONFIG))  # deep copy
    raised["global"]["router"]["learning"]["protection"]["tuning"][
        "stability_weight"
    ] = 2.0
    raised["global"]["router"]["learning"]["protection"]["tuning"][
        "switch_margin"
    ] = 0.2
    candidates = propose_session_policy(raised, replay_summary=OUT_OF_LOCK_SUMMARY)
    # every hold knob is at its raised value: nothing left to change
    assert candidates == []


def _diff(base, candidate):
    """The (path, key) pairs where the candidate differs from the base."""
    diffs = []

    def walk(b, c, path):
        if isinstance(b, dict) and isinstance(c, dict):
            for key in set(b) | set(c):
                walk(b.get(key), c.get(key), [*path, str(key)])
        elif b != c:
            diffs.append((".".join(path[:-1]), path[-1]))

    walk(base, candidate, [])
    return diffs


# ---- golden flow: the raised candidate switches less on replay ----


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the runner's process-group teardown needs os.killpg (POSIX)",
)
def test_out_of_lock_flap_candidate_switches_less_on_replay(tmp_path):
    from pathlib import Path

    from tests.test_route_replay import _build_route_stack

    stack = _build_route_stack(tmp_path)

    # identity present: the loop hard-locked onto the baseline, and the
    # post-loop request flapped under the low hold
    base_config, base_env = stack("protected")
    base_result = evaluate(
        base_config, base_env, tmp_path, "base", identity="conv-ag6-base"
    )
    summary = base_result["summary"]
    assert summary["tool_loop_switches"] == 0
    assert summary["model_switches"] == 1

    # the out-of-lock flap evidence triggers the hold candidates
    base_spec = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    candidates = propose_session_policy(base_spec, replay_summary=summary)
    assert _variables_of(candidates) == ["stability_weight", "switch_margin"]

    # the raised candidate holds the post-loop lane on replay
    raised_config, raised_env = stack("protected", stability_weight=2.0)
    raised = evaluate(
        raised_config, raised_env, tmp_path, "raised", identity="conv-ag6-raised"
    )
    assert raised["summary"]["tool_loop_switches"] == 0
    assert raised["summary"]["model_switches"] == 0


def evaluate(config, env, tmp_path, name, *, identity):
    from recipe_lab.runner import evaluate_route_replay
    from tests.test_route_replay import _session_case_post_loop

    env = RunnerEnvLike(env, tmp_path, name)
    return evaluate_route_replay(
        candidate_config=config,
        cases=[_session_case_post_loop("s1", identity)],
        env=env.env,
        router_model="vllm-sr/auto",
        request_timeout=30.0,
        trial_timeout_s=60,
        request_id_prefix=f"rl-rt-{name}-",
        gates={"max_unprotected_tool_loop_switches": 0},
    )


class RunnerEnvLike:
    """A per-evaluation RunnerEnv (fresh work dir and backend port)."""

    def __init__(self, env, tmp_path, name):
        import dataclasses

        from tests.test_route_replay import _free_port

        env_dict = dataclasses.asdict(env)
        env_dict["work_dir"] = tmp_path / f"work-{name}"
        env_dict["backend_port"] = _free_port()
        self.env = dataclasses.replace(env, **env_dict)
