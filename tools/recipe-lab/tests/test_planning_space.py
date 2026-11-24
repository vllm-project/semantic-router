"""Exercise P2 on guarded two-model parents, historical references, and complex recipes."""

import copy
import json
import unittest
from unittest.mock import patch

from recipe_lab.planning_space import (
    build_space,
    extract_candidate,
    extract_reference,
    native_edits,
    render_handoff,
)
from recipe_lab.window_action import apply_window

ARGS = {"public_model": "public", "strong": "a", "fast": "b"}


def parent():
    providers = [
        {
            "name": alias,
            "provider_model_id": ident,
            "capabilities": ["chat", "tools"],
            "context_window_size": 131072,
            "max_output_tokens": 32768,
            "backend_refs": [{"endpoint": "DO-NOT-PRINT", "auth": "SECRET"}],
        }
        for alias, ident in (("a", "GLM-5.3"), ("b", "Qwen/Qwen3.5-35B-A3B"))
    ]
    return {
        "version": "v0.5",
        "providers": {"models": providers},
        "entrypoints": [{"model_names": ["public"], "recipe": "work"}],
        "recipes": [
            {
                "name": "work",
                "strategy": "priority",
                "candidate_requirements": {
                    "context": "known_limits",
                    "capabilities": "declared",
                },
                "signals": {},
                "decisions": [
                    {
                        "name": "default",
                        "priority": 10,
                        "algorithm": {"type": "static"},
                        "modelRefs": [{"model": "a"}],
                        "reliability": {"total_timeout": "0s"},
                        "adaptations": {"adaptation": {"candidate_set": "decision"}},
                    }
                ],
            },
            {"name": "sibling", "decisions": []},
        ],
    }


class PlanningSpaceTest(unittest.TestCase):
    def test_planner_uses_routing_reference_metadata_without_a_specific_case(self):
        doc = parent()
        with patch(
            "recipe_lab.planning_space.load_cases",
            return_value=[{"id": "observation"}],
        ):
            space = build_space(doc, **ARGS)
        self.assertEqual([c["id"] for c in space["candidates"]], ["parent"])
        self.assertEqual(space["historical_references"], [])
        ref = {
            "id": "office-followup",
            "strong_ids": ["writer"],
            "fast_ids": ["editor"],
            "feature": "tool_results",
            "lower": 2,
            "required_overrides": [],
        }
        for card, model_id in zip(
            doc["providers"]["models"], ["writer", "editor"], strict=True
        ):
            card["provider_model_id"] = model_id
        with patch(
            "recipe_lab.planning_space.load_cases",
            return_value=[{"id": "documents", "routing_references": [ref]}],
        ):
            space = build_space(doc, **ARGS)
        self.assertEqual(space["candidates"][1]["id"], "office-followup")
        self.assertEqual(space["candidates"][1]["source"], "documents")

    def test_casebook_is_pair_specific_and_renders_only_target_branch(self):
        doc = parent()
        original = copy.deepcopy(doc)
        plan = build_space(doc, **ARGS)
        self.assertEqual([c["id"] for c in plan["candidates"]], ["parent", "e7b-35b"])
        self.assertEqual(
            plan["candidates"][1]["predicate"],
            {"feature": "tool_results", "lower": 30, "upper": None},
        )
        self.assertNotIn("SECRET", json.dumps(plan))
        self.assertNotIn("DO-NOT-PRINT", json.dumps(plan))
        changed = render_handoff(doc, **ARGS, threshold=30)
        self.assertEqual(doc, original)
        self.assertEqual(changed["recipes"][1], doc["recipes"][1])
        branch, default = changed["recipes"][0]["decisions"]
        self.assertEqual(default, doc["recipes"][0]["decisions"][0])
        self.assertEqual(branch["reliability"], default["reliability"])
        self.assertEqual(branch["adaptations"], default["adaptations"])
        self.assertEqual(branch["modelRefs"], [{"model": "b"}, {"model": "a"}])
        self.assertEqual(extract_candidate(changed, **ARGS)["kind"], "count_handoff")

    def test_unknown_identity_requires_explicit_reference_and_9b_needs_its_guard(self):
        doc = parent()
        cards = doc["providers"]["models"]
        cards[1]["provider_model_id"] = "qwen35-small"
        report = build_space(doc, **ARGS)
        self.assertEqual(len(report["candidates"]), 1)
        self.assertFalse(
            any(r["model_pair_match"] for r in report["historical_references"])
        )
        cards[0]["provider_model_id"] = "deepseek-v4-flash"
        cards[1]["provider_model_id"] = "qwen3.5-9b"
        report = build_space(doc, **ARGS)
        refs = [r for r in report["historical_references"] if r["model_pair_match"]]
        self.assertEqual([(r["lower"], r["renderable"]) for r in refs], [(35, False)])
        self.assertIn("long_context", refs[0]["reason"])

    def test_roles_and_bounds_from_the_declared_rule(self):
        doc = render_handoff(parent(), **ARGS, threshold=25)
        recipe = doc["recipes"][0]
        signal = recipe["signals"]["conversation"][0]
        signal["feature"]["source"] = {"type": "message"}
        signal["predicate"] = {"gte": 25, "lte": 32}
        self.assertEqual(
            extract_candidate(doc, **ARGS)["predicate"],
            {"feature": "all_messages", "lower": 25, "upper": 32},
        )
        signal["feature"]["source"]["role"] = "assistant"
        self.assertEqual(extract_candidate(doc, **ARGS)["kind"], "requires_probe")

    def test_migration_extracts_only_a_dominant_reference_predicate(self):
        doc = render_handoff(parent(), **ARGS, threshold=30)
        r = doc["recipes"][0]
        del r["candidate_requirements"]
        protected = dict(
            r["decisions"][1],
            name="analysis",
            priority=1,
            rules={
                "operator": "OR",
                "conditions": [{"type": "systemone", "name": "analysis"}],
            },
        )
        r["decisions"].append(protected)
        self.assertEqual(extract_candidate(doc, **ARGS)["kind"], "requires_probe")
        self.assertEqual(extract_reference(doc, **ARGS)["predicate"]["lower"], 30)
        protected["priority"] = 999
        self.assertEqual(extract_reference(doc, **ARGS)["kind"], "requires_probe")

    def test_render_rejects_protected_partial_and_shared_parents(self):
        for change in (
            lambda r: r["decisions"][0].update(action={"type": "route"}),
            lambda r: r["decisions"][0].update(plugins=[{"type": "allowlist"}]),
            lambda r: r["candidate_requirements"].pop("context"),
            lambda r: r["decisions"][0].update(adaptations={"unexpected": {}}),
        ):
            doc = parent()
            change(doc["recipes"][0])
            with self.assertRaises(ValueError):
                render_handoff(doc, **ARGS, threshold=30)
        doc = parent()
        doc["entrypoints"][0]["model_names"].append("shared")
        with self.assertRaises(ValueError):
            render_handoff(doc, **ARGS, threshold=30)

    def test_complex_decisions_remain_visible_without_reporting_a_fake_replay(self):
        doc = parent()
        recipe = doc["recipes"][0]
        recipe["decisions"].insert(
            0,
            {
                "name": "depth",
                "priority": 300,
                "algorithm": {"type": "static"},
                "rules": {
                    "operator": "OR",
                    "conditions": [{"type": "systemone", "name": "depth"}],
                },
                "modelRefs": [{"model": "a"}],
                "plugins": [
                    {
                        "type": "request_params",
                        "configuration": {
                            "default_max_tokens": 8192,
                            "secret": "SECRET",
                        },
                    }
                ],
            },
        )
        report = build_space(doc, **ARGS)
        self.assertEqual(report["candidates"][0]["kind"], "requires_probe")
        depth = report["inventory"]["decisions"][0]
        self.assertEqual(
            depth["dependencies"], [{"type": "systemone", "name": "depth"}]
        )
        self.assertEqual(depth["generation"], [{"default_max_tokens": 8192}])
        self.assertNotIn("SECRET", json.dumps(report))

    def test_native_edit_shapes_are_found_on_this_parent(self):
        doc = parent()
        recipe = doc["recipes"][0]
        default = recipe["decisions"][0]

        def leaf(name):
            return {"type": "conversation", "name": name}

        def no(name):
            return {"operator": "NOT", "conditions": [leaf(name)]}

        def counted(name, source, threshold):
            return {
                "name": name,
                "feature": {"type": "count", "source": source},
                "predicate": {"gte": threshold},
            }

        recipe["signals"] = {
            "conversation": [
                counted("lo", {"type": "assistant_tool_cycle"}, 1),
                counted("hi", {"type": "assistant_tool_cycle"}, 4),
                counted("offered", {"type": "tool_definition"}, 1),
                counted("past", {"type": "message", "role": "assistant"}, 1),
                {
                    "name": "off",
                    "feature": {
                        "type": "exists",
                        "source": {"type": "tool_choice_none"},
                    },
                },
            ]
        }
        recipe["decisions"] = [
            {
                **default,
                "name": "window",
                "rules": {"operator": "AND", "conditions": [leaf("lo"), no("hi")]},
            },
            {
                **default,
                "name": "open",
                "rules": {
                    "operator": "AND",
                    "on_unknown": "no_match",
                    "conditions": [leaf("offered"), no("off"), no("past")],
                },
            },
            default,
        ]
        families = build_space(doc, **ARGS)["inventory"]["families"]
        [window] = families["trigger"]["tool_result_window_upper"]["targets"]
        self.assertEqual((window["current_lower"], window["current_upper"]), (1, 3))
        [opening] = families["binding"]["first_tool_model"]["targets"]
        self.assertEqual(opening["current_model"], "a")
        self.assertFalse(families["trigger"]["late_tool_results"]["parent_compatible"])
        spec = {
            **{
                k: opening[k]
                for k in (
                    "public_model",
                    "recipe",
                    "decision",
                    "tools_signal",
                    "disabled_signal",
                    "assistant_signal",
                    "allowed_models",
                )
            },
            "action": "first_tool_model",
            "schema": "recipe-lab-first-tool-space-v1",
            "base_config_sha256": "a" * 64,
        }
        [(descriptor, changed)] = native_edits(
            doc, json.dumps(spec).encode(), base_sha="a" * 64, **ARGS
        )
        self.assertEqual(descriptor["kind"], "requires_probe")
        self.assertEqual(descriptor["edit"]["after_model"], "b")
        self.assertEqual(
            changed["recipes"][0]["decisions"][1]["modelRefs"], [{"model": "b"}]
        )
        self.assertEqual(recipe["decisions"][1]["modelRefs"], [{"model": "a"}])
        recipe["decisions"][1]["rules"]["conditions"].pop()
        self.assertEqual(
            build_space(doc, **ARGS)["inventory"]["families"]["binding"][
                "first_tool_model"
            ]["targets"],
            [],
        )
        with self.assertRaises(ValueError):
            native_edits(doc, json.dumps(spec).encode(), base_sha="a" * 64, **ARGS)

    def test_declared_window_domain_scored_only_on_newly_affected_calls(self):
        doc = render_handoff(parent(), **ARGS, threshold=1)
        r = doc["recipes"][0]
        sig = r["signals"]["conversation"][0]
        r["signals"]["conversation"].append(
            {**sig, "name": "end", "predicate": {"gte": 4}}
        )
        rule = {
            "operator": "NOT",
            "conditions": [{"type": "conversation", "name": "end"}],
        }
        r["decisions"][0]["rules"]["conditions"].append(rule)
        space = {
            "schema": "recipe-lab-window-space-v1",
            "action": "tool_result_window_upper",
            "base_config_sha256": "b" * 64,
            "public_model": "public",
            "recipe": "work",
            "decision": "lab_late_fast",
            "lower_signal": sig["name"],
            "upper_signal": "end",
            "allowed_upper": [2, 3, 8],
        }
        raw = json.dumps(space).encode()
        pair = native_edits(doc, raw, base_sha="b" * 64, **ARGS)
        self.assertEqual(len(pair), 2)
        (down, _), (up, child) = pair
        self.assertEqual(
            up["predicate"], {"feature": "tool_results", "lower": 4, "upper": 8}
        )
        self.assertEqual(up["predicate_scope"], "changed_window_only")
        self.assertEqual(down["change_direction"], "narrow")
        self.assertEqual(down["kind"], "requires_probe")
        self.assertEqual(
            child,
            apply_window(
                doc, space, public_model="public", config_sha256="b" * 64, upper=8
            )[0],
        )
        with self.assertRaises(ValueError):
            native_edits(doc, raw, base_sha="a" * 64, **ARGS)
        other = {**r["decisions"][1], "priority": 999, "name": "higher"}
        r["decisions"].append(other)
        (_, _), (overridden, _) = native_edits(doc, raw, base_sha="b" * 64, **ARGS)
        self.assertEqual(overridden["kind"], "requires_probe")
