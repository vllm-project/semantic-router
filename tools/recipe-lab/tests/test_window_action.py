"""Exercise a scoped edit through the same ledger and CLI as an agent."""

import copy
import io
import unittest
from contextlib import redirect_stderr

import test_loop
import yaml
from recipe_lab.loop_evidence import digest
from recipe_lab.window_action import apply_window, load_space, window_bounds


class WindowTest(unittest.TestCase):
    def setUp(self):
        self.run_fixture = test_loop.LoopTest()
        self.run_fixture.setUp()
        self.addCleanup(self.run_fixture.doCleanups)
        self.root = self.run_fixture.root
        self.config = self.root / "fixed/config.yaml"
        self.doc = test_loop.get(self.config)
        recipe = self.doc["recipes"][1]
        self.feature = {"type": "count", "source": {"type": "assistant_tool_cycle"}}
        recipe["signals"] = {
            "conversation": [
                {"name": "started", "feature": self.feature, "predicate": {"gte": 1}},
                {"name": "late", "feature": self.feature, "predicate": {"gte": 4}},
            ]
        }
        self.lower_ref = {"type": "conversation", "name": "started"}
        self.upper_ref = {"type": "conversation", "name": "late"}
        self.conditions = [
            self.lower_ref,
            {"operator": "NOT", "conditions": [self.upper_ref]},
        ]
        recipe["decisions"][0]["rules"] = {
            "operator": "AND",
            "conditions": self.conditions,
        }
        test_loop.put(self.config, self.doc)
        self.base_sha = digest(self.config.read_bytes())
        binding_path = self.root / "fixed/dsh/bindings/task/fixed.json"
        binding = test_loop.get(binding_path)
        test_loop.put(binding_path, dict(binding, source_sha256=self.base_sha))
        self.base_id = self.run_fixture.record("fixed")
        self.space = {
            "schema": "recipe-lab-window-space-v1",
            "action": "tool_result_window_upper",
            "base_config_sha256": self.base_sha,
            "public_model": "public",
            "recipe": "main",
            "decision": "fast",
            "lower_signal": "started",
            "upper_signal": "late",
            "allowed_upper": [3, 8],
        }

    def propose(self, space, upper):
        file = self.root / "space.yaml"
        file.write_text(yaml.safe_dump(space), encoding="utf-8")
        return self.run_fixture.cli(
            "propose",
            "--base",
            self.base_id,
            "--space",
            file,
            "--upper",
            upper,
            "--out",
            self.root / "candidate.yaml",
        )

    def test_proposal_changes_only_gte_and_pins_evidence(self):
        receipt = self.propose(self.space, 8)
        candidate = self.root / "candidate.yaml"
        expected = copy.deepcopy(self.doc)
        expected["recipes"][1]["signals"]["conversation"][1]["predicate"]["gte"] = 9
        self.assertEqual(yaml.safe_load(candidate.read_bytes()), expected)
        self.assertEqual(receipt["based_on"], self.base_id)
        self.assertEqual(receipt["base_config_sha256"], self.base_sha)
        self.assertEqual(receipt["config_sha256"], digest(candidate.read_bytes()))
        self.assertEqual(
            receipt["space_sha256"], digest((self.root / "space.yaml").read_bytes())
        )
        self.assertEqual(receipt["edit"]["before_upper"], 3)
        self.assertEqual(receipt["edit"]["after_upper"], 8)

    def test_invalid_action_never_publishes_candidate(self):
        for changed, upper in [
            ({}, 9),
            ({}, 3),
            ({"recipe": "sibling"}, 8),
            ({"public_model": "other"}, 8),
            ({"upper_signal": "absent"}, 8),
            ({"decision": "absent"}, 8),
            ({"base_config_sha256": "e" * 64}, 8),
            ({"allowed_upper": [0, 1]}, 0),
            ({"allowed_upper": [True, 8]}, 8),
            ({"allowed_upper": [3, 8, 8]}, 8),
            ({"allowed_upper": [2**53]}, 2**53),
            ({"extra": {"max_completion_tokens": 12}}, 8),
        ]:
            with (
                self.subTest(changed=changed, upper=upper),
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    self.propose({**self.space, **changed}, upper)
                self.assertFalse((self.root / "candidate.yaml").exists())

    def test_rejects_different_window_semantics_or_shared_upper_signal(self):
        for replacement in [
            {"operator": "OR", "conditions": self.conditions},
            {"operator": "AND", "conditions": [self.lower_ref, self.upper_ref]},
            {"operator": "AND", "conditions": [*self.conditions, self.upper_ref]},
        ]:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                doc = copy.deepcopy(self.doc)
                doc["recipes"][1]["decisions"][0]["rules"] = replacement
                window_bounds(doc, self.space)
        rules = self.doc["recipes"][1]["signals"]["conversation"]
        for change in [
            {
                "feature": {
                    "type": "count",
                    "source": {"type": "message", "role": "user"},
                }
            },
            {"predicate": {"gte": 4, "lt": 9}},
            {"predicate": {"gte": True}},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                doc = copy.deepcopy(self.doc)
                doc["recipes"][1]["signals"]["conversation"][1].update(change)
                window_bounds(doc, self.space)
        rules.append(copy.deepcopy(rules[1]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            window_bounds(self.doc, self.space)

    def test_yaml_aliases_do_not_change_frozen_fields(self):
        original_predicate = self.doc["recipes"][1]["signals"]["conversation"][1][
            "predicate"
        ]
        self.doc["recipes"][0]["signals"] = {
            "conversation": [
                {
                    "name": "sibling",
                    "feature": self.feature,
                    "predicate": original_predicate,
                }
            ]
        }
        raw = yaml.safe_dump(self.doc).encode()
        space = {**self.space, "base_config_sha256": digest(raw)}
        doc = yaml.safe_load(raw)
        candidate, _ = apply_window(
            doc, space, public_model="public", config_sha256=digest(raw), upper=8
        )
        self.assertEqual(candidate["recipes"][0], self.doc["recipes"][0])
        self.assertEqual(window_bounds(doc, space), (1, 3))
        self.assertEqual(window_bounds(candidate, space), (1, 8))

    def test_ambiguous_space_file_is_rejected(self):
        raw = yaml.safe_dump(self.space) + "recipe: another\n"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_space(raw.encode())
