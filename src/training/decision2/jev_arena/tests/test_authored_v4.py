"""Regression and quality gates for the blocked authored v4 candidate."""

from collections import Counter
import json
from pathlib import Path
import random
import tempfile
import unittest

from jev_arena.authored_v4 import (
    CHALLENGES,
    OPERATIONS,
    build,
    make_spec,
    render,
    solve_visible,
)
from jev_arena.authored_v4_ops import (
    BY_ID,
    WORKFLOW_ORDERS,
    evaluate,
    generate_facts,
    possible_worlds,
)
from jev_arena.authored_v4_reference import evaluate_rendered


class AuthoredV4Test(unittest.TestCase):
    def test_sixty_operations_have_independent_visible_oracle(self):
        self.assertEqual(len(OPERATIONS), 60)
        seed = bytes(range(32))
        for op in OPERATIONS:
            for challenge in CHALLENGES:
                spec = next(
                    make_spec(seed, "dev", op, challenge, 0, attempt)
                    for attempt in range(128)
                    if make_spec(seed, "dev", op, challenge, 0, attempt) is not None
                )
                prompt, target = render(spec, seed)
                self.assertEqual(
                    solve_visible(prompt), target["gold"]["decision"]["value"]
                )
                self.assertEqual(
                    evaluate(op, spec["facts"]), evaluate_rendered(op, spec["facts"])
                )

    def test_partial_evidence_can_be_resolved_or_unresolved(self):
        seed = bytes(range(32))
        labels = Counter()
        for op in OPERATIONS:
            for attempt in range(80):
                spec = make_spec(
                    seed, "release", op, "insufficient_evidence", 0, attempt
                )
                if spec is None:
                    continue
                self.assertEqual(
                    evaluate(op, spec["facts"]), evaluate_rendered(op, spec["facts"])
                )
                worlds = possible_worlds(spec["facts"])
                answers = [evaluate(op, world) for world in worlds]
                if spec["partial_resolution"] == "resolved":
                    self.assertEqual(answers[0], answers[1])
                else:
                    self.assertNotEqual(answers[0], answers[1])
                labels[(op.kind, spec["partial_resolution"])] += 1
        self.assertEqual(len(labels), 6)

    def test_editorial_semantic_regressions(self):
        op = BY_ID["score14-forecast-error-band"]
        for error, expected in ((0, 4), (1, 3), (2, 3), (3, 3), (4, 2)):
            facts = {"forecast": 16, "actual": 16 + error}
            self.assertEqual(evaluate(op, facts), expected)
            self.assertEqual(evaluate_rendered(op, facts), expected)
        op = BY_ID["noul05-conditional-obligation"]
        facts = {"checks": [False, False, True, True], "indices": [0, 1, 2, 3]}
        self.assertTrue(evaluate(op, facts))
        self.assertTrue(evaluate_rendered(op, facts))
        self.assertTrue(
            all(
                (
                    order[0] == "intake"
                    and order[-1] == "close"
                    and order.index("approve") > order.index("review")
                    if "review" in order
                    else order[0] == "intake" and order[-1] == "close"
                )
                for order in WORKFLOW_ORDERS
            )
        )
        reach = BY_ID["noul11-path-reachability"]
        self.assertTrue(evaluate(reach, {"edges": [[0, 4]]}))
        self.assertFalse(evaluate(reach, {"edges": [[4, 0]]}))
        self.assertTrue(evaluate_rendered(reach, {"edges": [[0, 4]]}))

    def test_release_cell_balance_and_blind_key_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            seed = base / "seed"
            seed.write_bytes(bytes(range(32)))
            protected = base / "protected.jsonl"
            protected.write_text(
                json.dumps(
                    {
                        "id": "unrelated",
                        "state": "An unrelated text.",
                        "questions": {"x": {"type": "noul"}},
                    }
                )
                + "\n"
            )
            inventory = base / "protected-list.json"
            inventory.write_text(
                json.dumps(
                    [{"kind": "prompts", "name": "dummy", "path": str(protected)}]
                )
            )
            output = base / "candidate"
            manifest = build("release", seed, inventory, output)
            audit = json.loads((output / "audit.json").read_text())
            self.assertEqual(manifest["items"], 1440)
            self.assertEqual(manifest["quality_gate"]["status"], "blocked")
            self.assertEqual(audit["balance_violations"], [])
            self.assertEqual(audit["near_internal_pairs"], [])
            self.assertGreater(audit["long_context_min_words"], 2000)
            self.assertEqual(len(audit["cell_label_counts"]), 240)
            packet = (output / "review_packet.private.jsonl").read_text()
            self.assertNotIn('"gold"', packet)
            self.assertEqual(len(packet.splitlines()), 240)
            self.assertEqual(
                (output / "review_key.private.jsonl").stat().st_mode & 0o777, 0o600
            )


if __name__ == "__main__":
    unittest.main()
