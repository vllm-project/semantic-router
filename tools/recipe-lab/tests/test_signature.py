"""Exercise a bounded proposal against the prefixes of exactly one dev measurement."""

import argparse
import json
import unittest

import yaml
import test_loop as fixtures

import recipe_lab.loop as loop
import recipe_lab.signature as signature
from recipe_lab.loop_evidence import digest
from recipe_lab.window_action import apply_window


def receipts(counts):
    rows = []
    for call, count in enumerate(counts, 1):
        roles = ["user", *(["tool"] * count)]
        rows.extend(
            [
                {
                    "event": "dispatch",
                    "call": call,
                    "model": "public",
                    "roles": roles,
                    "last_role": roles[-1],
                    "messages_sha256": digest(str(call).encode()),
                    "wire_fields": ["model", "messages", "max_completion_tokens"],
                },
                {
                    "event": "response",
                    "call": call,
                    "http": 200,
                    "done": True,
                    "elapsed_s": 1,
                    "selected_decision": "fallback",
                    "selected_model": "A",
                    "returned_model": "A",
                },
            ]
        )
    return rows


def encode_rows(rows):
    return "\n".join(map(json.dumps, rows)).encode("utf-8")


class SignatureTest(unittest.TestCase):
    def test_fixed_requests_include_narrowing_and_missing_fields(self):
        rows = receipts([0, 1, 3, 8, 9, 1])
        del rows[-2]["roles"]
        summary = signature.task_signature(
            encode_rows(rows),
            "public",
            {"lower": 1, "before_upper": 8, "after_upper": 3},
        )
        self.assertEqual(
            (summary["dispatches"], summary["observed"], summary["unknown"]), (6, 5, 1)
        )
        self.assertEqual(summary["before_hit_calls"], [2, 3, 4])
        self.assertEqual(summary["after_hit_calls"], [2, 3])
        self.assertEqual((summary["removed"], summary["added"]), (1, 0))
        self.assertEqual(summary["changed_calls"][0]["observed_source_model"], "A")
        for row in rows[::2]:
            row["wire_fields"].remove("max_completion_tokens")
        unsupported = signature.task_signature(
            encode_rows(rows),
            "public",
            {"lower": 1, "before_upper": 3, "after_upper": 8},
        )
        self.assertEqual((unsupported["unknown"], unsupported["observed"]), (2, 4))
        self.assertEqual(
            unsupported["missing_calls"][1],
            "unsupported_or_unidentifiable_wire_protocol",
        )
        self.assertEqual(
            unsupported["missing_calls"][6], "missing_role_sequence_or_identity"
        )

    def test_inconsistent_or_duplicate_receipts_rejected(self):
        action = {"lower": 1, "before_upper": 3, "after_upper": 8}
        for damage in (
            "wrong_last",
            "duplicate",
            "missing_response",
            "bool_call",
            "bad_digest",
        ):
            with self.subTest(damage=damage):
                rows = receipts([1])
                if damage == "wrong_last":
                    rows[0]["last_role"] = "user"
                elif damage == "duplicate":
                    rows.append(rows[0])
                elif damage == "missing_response":
                    rows.pop()
                elif damage == "bool_call":
                    rows[0]["call"] = True
                else:
                    rows[0]["messages_sha256"] = "truncated"
                with self.assertRaises(ValueError):
                    signature.task_signature(encode_rows(rows), "public", action)

    def test_dev_arm_and_candidate_identity(self):
        run = fixtures.LoopTest()
        run.setUp()
        self.addCleanup(run.doCleanups)
        cfg = run.root / "draft/config.yaml"
        doc = fixtures.get(cfg)
        recipe = doc["recipes"][1]
        recipe["signals"] = {
            "conversation": [
                {
                    "name": name,
                    "feature": {
                        "type": "count",
                        "source": {"type": "assistant_tool_cycle"},
                    },
                    "predicate": {"gte": value},
                }
                for name, value in (("seen", 1), ("late", 4))
            ]
        }
        recipe["decisions"][0]["rules"] = {
            "operator": "AND",
            "conditions": [
                {"type": "conversation", "name": "seen"},
                {
                    "operator": "NOT",
                    "conditions": [{"type": "conversation", "name": "late"}],
                },
            ],
        }
        fixtures.put(cfg, doc)
        binding = run.root / "draft/dsh/bindings/task/draft.json"
        fixtures.put(
            binding,
            {**fixtures.get(binding), "source_sha256": digest(cfg.read_bytes())},
        )
        # The proposal parent and the prefix arm intentionally have different histories.
        folder = run.root / "fixed/dsh/runs/task/fixed/main"
        plugins = fixtures.get(folder / "provider.json")
        plugins[1]["config"]["providers"]["source"]["api"] = "openai-completions"
        fixtures.put(folder / "provider.json", plugins)
        (folder / "requests.jsonl").write_bytes(encode_rows(receipts([3, 4, 8, 9])))
        result = folder / "result.json"
        fixtures.put(result, {**fixtures.get(result), "model_requests": 4})
        prefixes = run.record("fixed")
        base = run.record("draft")
        hidden = run.record("draft", "--split", "holdout")
        ledger = loop.Ledger(run.workspace)
        space = {
            "schema": "recipe-lab-window-space-v1",
            "action": "tool_result_window_upper",
            "base_config_sha256": digest(cfg.read_bytes()),
            "public_model": "public",
            "recipe": "main",
            "decision": "fast",
            "lower_signal": "seen",
            "upper_signal": "late",
            "allowed_upper": [3, 8],
        }
        decl = run.root / "space.yaml"
        fixtures.put(decl, space)
        candidate = run.root / "candidate.yaml"
        expected, _ = apply_window(
            doc,
            space,
            public_model="public",
            config_sha256=digest(cfg.read_bytes()),
            upper=8,
        )
        candidate.write_bytes(
            yaml.safe_dump(expected, sort_keys=False, allow_unicode=True).encode()
        )
        parser = argparse.ArgumentParser()
        signature.add_arguments(parser)
        args = parser.parse_args(
            [
                "--prefixes",
                prefixes,
                "--base",
                base,
                "--space",
                str(decl),
                "--upper",
                "8",
                "--candidate",
                str(candidate),
            ]
        )
        observed = signature.signature(ledger, args)
        self.assertEqual(observed["candidate_sha256"], digest(candidate.read_bytes()))
        self.assertEqual(observed["prefixes"], prefixes)
        self.assertEqual(observed["based_on"], base)
        self.assertEqual(observed["total"]["dispatches"], 4)
        self.assertEqual(observed["by_task"]["task"]["changed"], 2)
        self.assertEqual(observed["candidate_route_evidence"]["status"], "unobserved")
        for key in ("prefixes", "base"):
            with self.subTest(holdout=key):
                params = argparse.Namespace(**{**vars(args), key: hidden})
                with self.assertRaisesRegex(ValueError, "development"):
                    signature.signature(ledger, params)
        candidate.write_bytes(
            candidate.read_bytes().replace(b"model: A", b"model: B") + b"# changed"
        )
        with self.assertRaisesRegex(ValueError, "candidate bytes"):
            signature.signature(ledger, args)
        (folder / "requests.jsonl").write_bytes(encode_rows(receipts([3])))
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            signature.signature(ledger, args)
        assigned = fixtures.get(binding)
        fixtures.put(binding, {**assigned, "assignment_id": "7" * 64})
        args.base = run.record("draft")
        with self.assertRaisesRegex(ValueError, "assigned episodes"):
            signature.signature(ledger, args)
