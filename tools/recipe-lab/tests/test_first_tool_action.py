"""Keep first-turn model edits within the measured config's capability and history guards."""

import contextlib
import io
import unittest

import test_loop
import yaml
from recipe_lab.first_tool_action import first_tool_edit
from recipe_lab.loop_evidence import digest


class FirstToolTest(unittest.TestCase):
    def setUp(self):
        self.fixture = test_loop.LoopTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.config = self.root / "fixed/config.yaml"
        self.doc = test_loop.get(self.config)
        for backend in self.doc["providers"]["models"]:
            backend["capabilities"] = ["text", "tools"]
            backend["context_window_size"] = 131072
            backend["max_output_tokens"] = 8192
        recipe = self.doc["recipes"][1]
        recipe["candidate_requirements"] = {
            "capabilities": "declared",
            "context": "known_limits",
        }
        recipe["signals"] = {
            "conversation": [
                {
                    "name": "offered",
                    "feature": {"type": "count", "source": {"type": "tool_definition"}},
                    "predicate": {"gte": 1},
                },
                {
                    "name": "disabled",
                    "feature": {
                        "type": "exists",
                        "source": {"type": "tool_choice_none"},
                    },
                },
                {
                    "name": "history",
                    "feature": {
                        "type": "count",
                        "source": {"type": "message", "role": "assistant"},
                    },
                    "predicate": {"gte": 1},
                },
            ]
        }

        def condition(name):
            return {"type": "conversation", "name": name}

        decision = recipe["decisions"][0]
        decision.update(
            algorithm={"type": "static"},
            rules={
                "operator": "AND",
                "on_unknown": "no_match",
                "conditions": [
                    condition("offered"),
                    {"operator": "NOT", "conditions": [condition("disabled")]},
                    {"operator": "NOT", "conditions": [condition("history")]},
                ],
            },
        )
        self.doc["recipes"][0]["decisions"] = [decision]
        test_loop.put(self.config, self.doc)
        self.sha = digest(self.config.read_bytes())
        binding = self.root / "fixed/dsh/bindings/task/fixed.json"
        test_loop.put(binding, dict(test_loop.get(binding), source_sha256=self.sha))
        self.space = {
            "schema": "recipe-lab-first-tool-space-v1",
            "action": "first_tool_model",
            "public_model": "public",
            "recipe": "main",
            "decision": "fast",
            "tools_signal": "offered",
            "disabled_signal": "disabled",
            "assistant_signal": "history",
            "allowed_models": ["A", "B"],
            "base_config_sha256": self.sha,
        }

    def edit(self, doc, space, model="B"):
        return first_tool_edit(
            doc,
            yaml.safe_dump(space).encode(),
            public_model="public",
            config_sha256=self.sha,
            model=model,
        )

    def test_descriptions_do_not_change_guard_semantics(self):
        signals = self.doc["recipes"][1]["signals"]["conversation"]
        for signal in signals:
            signal["description"] = "Human-readable intent"
        result, _ = self.edit(self.doc, self.space)
        self.assertEqual(result["recipes"][1]["signals"]["conversation"], signals)

    def test_normal_cli_changes_only_one_destination(self):
        expected = test_loop.get(self.config)
        expected["recipes"][1]["decisions"][0]["modelRefs"] = [{"model": "B"}]
        output = self.root / "candidate.yaml"
        spacefile = self.root / "space.yaml"
        spacefile.write_bytes(yaml.safe_dump(self.space).encode())
        parent = self.fixture.record("fixed")
        receipt = self.fixture.cli(
            "propose",
            "--base",
            parent,
            "--space",
            spacefile,
            "--first-tool-model",
            "B",
            "--out",
            output,
        )
        self.assertEqual(yaml.safe_load(output.read_bytes()), expected)
        self.assertEqual(receipt["space_sha256"], digest(spacefile.read_bytes()))
        self.assertEqual(receipt["edit"]["before_model"], "A")
        self.assertEqual(receipt["edit"]["after_model"], "B")
        self.assertIn("visible to the Router", receipt["edit"]["request_scope"])
        # A shared YAML decision must also stay unchanged in a sibling recipe.
        changed, _ = self.edit(self.doc, self.space)
        self.assertEqual(changed["recipes"][0], self.doc["recipes"][0])
        self.assertEqual(
            self.doc["recipes"][1]["decisions"][0]["modelRefs"], [{"model": "A"}]
        )

    def test_refuses_unqualified_models_or_misleading_guards(self):
        recipe = ("recipes", 1)
        decision = (*recipe, "decisions", 0)
        rules = (*decision, "rules")
        for path, wrong in [
            (("providers", "models", 1, "capabilities"), ["text"]),
            (("providers", "models", 1, "capabilities"), ["tools"]),
            (("providers", "models", 1, "context_window_size"), None),
            (("providers", "models", 0, "max_output_tokens"), True),
            (("providers", "models", 1, "max_output_tokens"), 0),
            ((*recipe, "candidate_requirements"), {"capabilities": "declared"}),
            ((*recipe, "candidate_requirements"), None),
            ((*recipe, "candidate_requirements"), {}),
            ((*decision, "rules"), None),
            ((*recipe, "signals"), []),
            ((*rules, "conditions"), []),
            (
                (*recipe, "signals", "conversation", 2, "feature", "source", "role"),
                "user",
            ),
            ((*rules, "conditions", 1), {"type": "conversation", "name": "disabled"}),
            ((*rules, "on_unknown"), "match"),
            ((*decision, "modelRefs", 0), {"model": "A", "reasoning_effort": "high"}),
            ((*decision, "algorithm"), {"type": "random"}),
        ]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                doc = test_loop.get(self.config)
                node = doc
                for field in path[:-1]:
                    node = node[field]
                node[path[-1]] = wrong
                self.edit(doc, self.space)
        for change, model in [
            ({"recipe": "sibling"}, "B"),
            ({"base_config_sha256": "f" * 64}, "B"),
            ({}, "A"),
            ({}, "C"),
        ]:
            with (
                self.subTest(change=change, model=model),
                self.assertRaises(ValueError),
            ):
                self.edit(self.doc, {**self.space, **change}, model)

    def test_router_aliases_and_smaller_declared_window_are_supported(self):
        doc = test_loop.get(self.config)
        backend = doc["providers"]["models"][1]
        backend.update(
            capabilities=[" Chat ", "TOOL_USE"],
            context_window_size=32768,
            max_output_tokens=1024,
        )
        doc["recipes"][1]["signals"]["conversation"][2]["predicate"]["gte"] = 1.0
        changed, _ = self.edit(doc, self.space)
        self.assertEqual(changed["providers"], doc["providers"])

    def test_canonical_scope_and_malformed_inputs(self):
        doc = test_loop.get(self.config)
        rules = doc["recipes"][1]["decisions"][0]["rules"]
        rules["conditions"].append({"type": "conversation", "name": "history"})
        with self.assertRaisesRegex(ValueError, "three first-tool guards"):
            self.edit(doc, self.space)
        for raw in (b"", b"[]", b"schema: wrong\n", b"schema: x\nschema: y\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                first_tool_edit(
                    self.doc,
                    raw,
                    public_model="public",
                    config_sha256=self.sha,
                    model="B",
                )
        with self.assertRaisesRegex(ValueError, "configuration mapping"):
            self.edit(None, self.space)

    def test_yaml_anchor_is_preserved_in_other_recipes(self):
        # The source document really aliases both recipes' decisions.
        doc = yaml.safe_load(yaml.safe_dump(self.doc))
        old = doc["recipes"][1]["decisions"][0]
        self.assertIs(doc["recipes"][0]["decisions"][0], old)
        changed, _ = self.edit(doc, self.space)
        reloaded = yaml.safe_load(yaml.safe_dump(changed))
        self.assertEqual(
            reloaded["recipes"][0]["decisions"][0]["modelRefs"], [{"model": "A"}]
        )
        self.assertEqual(
            reloaded["recipes"][1]["decisions"][0]["modelRefs"], [{"model": "B"}]
        )
        self.assertEqual(old["modelRefs"], [{"model": "A"}])

    def test_cli_rejects_missing_dev_parent_and_incompatible_action(self):
        parent = self.fixture.record("fixed")
        space = self.root / "opening.yaml"
        out = self.root / "candidate.yaml"
        space.write_bytes(yaml.safe_dump(self.space).encode())
        for base, flags in (
            ("missing", ["--space", space, "--first-tool-model", "B"]),
            (parent, ["--space", space, "--upper", "8"]),
            (parent, ["--space", space, "--upper", "8", "--first-tool-model", "B"]),
            (parent, ["--recipe", self.config, "--first-tool-model", "B"]),
        ):
            with (
                self.subTest(flags=flags),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as err,
            ):
                self.fixture.cli("propose", "--base", base, *flags, "--out", out)
            self.assertEqual(err.exception.code, 2)
            self.assertFalse(out.exists())
