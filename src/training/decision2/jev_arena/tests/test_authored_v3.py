"""Contract tests for distinct authored v3 operations and rendered oracle."""

import json
from pathlib import Path
import random
from collections import Counter
import tempfile
import unittest

from jev_arena.authored_v3 import CHALLENGES, build, make_spec, render, solve_visible
from jev_arena.authored_v3_ops import (
    OPERATIONS,
    evaluate,
    generate_facts,
    inject_missing,
)
from jev_arena.authored_v3_reference import evaluate_rendered


class AuthoredV3Test(unittest.TestCase):
    def test_sixty_formulas_are_behaviorally_distinct_on_common_probes(self):
        self.assertEqual(len(OPERATIONS), 60)
        for kind in ("choice", "noul", "score"):
            ops = [op for op in OPERATIONS if op.kind == kind]
            signatures = [
                tuple(
                    str(evaluate(op, generate_facts(op, random.Random(seed))))
                    for seed in range(128)
                )
                for op in ops
            ]
            self.assertEqual(len(set(signatures)), 20, kind)

    def test_two_oracles_agree_on_independent_facts_and_missing_evidence(self):
        for op in OPERATIONS:
            for seed in range(24):
                facts = generate_facts(op, random.Random(seed))
                self.assertEqual(
                    evaluate(op, facts), evaluate_rendered(op, facts), op.id
                )
                if seed < 2:
                    missing = inject_missing(facts)
                    self.assertEqual(
                        evaluate(op, missing), evaluate_rendered(op, missing), op.id
                    )

    def test_visible_parser_recovers_each_operation_and_challenge(self):
        seed = bytes(range(32))
        ids = set()
        for op in OPERATIONS:
            for challenge in CHALLENGES:
                spec = make_spec(seed, "dev", op, challenge, 0)
                prompt, target = render(spec, seed)
                self.assertEqual(
                    solve_visible(prompt), target["gold"]["decision"]["value"]
                )
                self.assertEqual(set(prompt), {"id", "state", "questions"})
                self.assertNotIn("gold", prompt)
                ids.add(prompt["id"])
        self.assertEqual(len(ids), 240)

    def test_missing_field_is_a_decision_field_and_dev_long_positions_vary(self):
        seed = bytes(range(32))
        positions = Counter()
        for op in OPERATIONS:
            facts = generate_facts(op, random.Random(9))
            if (
                "checks" in facts
                and "indices" in facts
                and next(iter(facts)) == "checks"
            ):
                missing = inject_missing(facts)
                self.assertIsNone(missing["checks"][facts["indices"][0]])
            if "signers" in facts and "attested" in facts:
                missing = inject_missing(facts)
                self.assertIsNone(missing["attested"][0])
            spec = make_spec(seed, "dev", op, "long_context", 0)
            prompt, _ = render(spec, seed)
            ratio = prompt["state"].find(f"BEGIN EVIDENCE [{spec['target_id']}]") / len(
                prompt["state"]
            )
            positions[
                "front" if ratio < 0.2 else "middle" if ratio < 0.8 else "end"
            ] += 1
        self.assertEqual(sum(positions.values()), 60)
        self.assertEqual(set(positions), {"front", "middle", "end"})
        self.assertGreater(min(positions.values()), 0)

    def test_ambiguous_or_changed_current_evidence_is_rejected(self):
        seed = bytes(range(32))
        spec = make_spec(seed, "dev", OPERATIONS[0], "near_distractor", 0)
        prompt, _ = render(spec, seed)
        prompt["state"] = prompt["state"].replace(
            "CURRENT POLICY", "ARCHIVED POLICY", 1
        )
        with self.assertRaisesRegex(ValueError, "CURRENT policy missing"):
            solve_visible(prompt)

    def test_duplicate_target_evidence_is_rejected(self):
        seed = bytes(range(32))
        spec = make_spec(seed, "dev", OPERATIONS[0], "near_distractor", 0)
        prompt, _ = render(spec, seed)
        target_marker = f"BEGIN EVIDENCE [{spec['target_id']}]"
        prompt["state"] += "\n" + prompt["state"].split(target_marker, 1)[1].split(
            f"END EVIDENCE [{spec['target_id']}]", 1
        )[0].join((target_marker, f"END EVIDENCE [{spec['target_id']}]"))
        with self.assertRaisesRegex(ValueError, "must be unique"):
            solve_visible(prompt)

    def test_normalized_protected_overlap_blocks_build(self):
        seed = bytes(range(32))
        spec = make_spec(seed, "dev", OPERATIONS[0], CHALLENGES[0], 0)
        prompt, _ = render(spec, seed)
        prompt["state"] = prompt["state"].replace(
            "Target case ID:", "Target   case ID:", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            seed_path = base / "seed.bin"
            protected_path = base / "protected.jsonl"
            list_path = base / "protected-list.json"
            seed_path.write_bytes(seed)
            protected_path.write_text(json.dumps(prompt) + "\n")
            list_path.write_text(
                json.dumps(
                    [
                        {
                            "kind": "prompts",
                            "name": "heldout",
                            "path": str(protected_path),
                        }
                    ]
                )
            )
            output = base / "candidate"
            with self.assertRaisesRegex(
                ValueError, "automated overlap/coverage audit blocked"
            ):
                build("dev", seed_path, list_path, output)
            self.assertFalse(output.exists())

    def test_blind_review_packet_and_key_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            seed_path = base / "seed.bin"
            protected_path = base / "protected.jsonl"
            list_path = base / "protected-list.json"
            seed_path.write_bytes(bytes(range(32)))
            protected_path.write_text(
                json.dumps(
                    {
                        "id": "unrelated",
                        "state": "An unrelated text.",
                        "questions": {"x": {"type": "noul"}},
                    }
                )
                + "\n"
            )
            list_path.write_text(
                json.dumps(
                    [
                        {
                            "kind": "prompts",
                            "name": "heldout",
                            "path": str(protected_path),
                        }
                    ]
                )
            )
            output = base / "candidate"
            manifest = build("dev", seed_path, list_path, output)
            self.assertEqual(manifest["quality_gate"]["status"], "blocked")
            reviews = [
                json.loads(line)
                for line in (output / "review_packet.private.jsonl")
                .read_text()
                .splitlines()
            ]
            keys = [
                json.loads(line)
                for line in (output / "review_key.private.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(reviews), len(keys), 240)
            self.assertEqual({r["id"] for r in reviews}, {r["id"] for r in keys})
            self.assertTrue(
                all("gold" not in r and "gold" not in r["prompt"] for r in reviews)
            )
            self.assertTrue(all("gold" in r for r in keys))
            for name in (
                "targets.jsonl",
                "source_specs.jsonl",
                "review_packet.private.jsonl",
                "review_key.private.jsonl",
            ):
                self.assertEqual((output / name).stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
