"""Golden contract checks for the sealed authored candidate builder."""

from pathlib import Path
import json
import tempfile
import unittest

from jev_arena.sealed_authored import (
    CHALLENGES,
    DOMAINS,
    TYPES,
    _bootstrap_macro,
    _check_release_lock,
    _near_overlap,
    _prediction_rows,
    _review_gate,
    input_digest,
    make_spec,
    render,
    sha_file,
    solve_rendered,
    solve_spec,
)


class SealedAuthoredTest(unittest.TestCase):
    def test_direct_and_rendered_oracles_agree_across_grid(self):
        seed = bytes(range(32))
        seen = set()
        for kind in TYPES:
            for challenge in CHALLENGES:
                for domain in DOMAINS:
                    for ordinal in range(2):
                        spec = make_spec(seed, "dev", kind, challenge, domain, ordinal)
                        prompt, target = render(spec)
                        self.assertEqual(
                            solve_spec(spec), solve_rendered(prompt, target)
                        )
                        self.assertEqual(
                            target["source_input_sha256"], input_digest(prompt)
                        )
                        self.assertNotIn("gold", prompt)
                        seen.add(target["group_id"])
        self.assertEqual(len(seen), 144)

    def test_dev_gold_diversity_and_release_context_floor(self):
        seed = bytes(range(32))
        observed = {kind: set() for kind in TYPES}
        for kind in TYPES:
            for challenge in CHALLENGES:
                for domain in DOMAINS:
                    for ordinal in range(2):
                        spec = make_spec(seed, "dev", kind, challenge, domain, ordinal)
                        observed[kind].add(solve_spec(spec))
        self.assertGreaterEqual(len(observed["choice"]), 4)
        self.assertEqual(observed["noul"], {False, True})
        self.assertEqual(observed["score"], set(range(5)))
        spec = make_spec(seed, "release", "score", "long_context", "build", 0)
        prompt, _ = render(spec)
        self.assertGreaterEqual(len(prompt["state"].split()), 2800)

    def test_choice_policy_maps_complete_to_release_and_missing_to_review(self):
        seed = bytes(range(32))
        complete = make_spec(seed, "release", "choice", "near_distractor", "build", 2)
        self.assertEqual(solve_spec(complete), "release")
        missing = make_spec(
            seed, "release", "choice", "insufficient_evidence", "build", 2
        )
        self.assertEqual(solve_spec(missing), "request_rollout_review")

    def test_operative_final_row_cannot_be_ambiguous(self):
        spec = make_spec(
            bytes(range(32)), "dev", "choice", "rule_precedence", "build", 0
        )
        prompt, target = render(spec)
        line = next(
            line
            for line in prompt["state"].splitlines()
            if line.startswith("FINAL | id=" + spec["target_id"])
        )
        prompt["state"] += "\n" + line
        with self.assertRaisesRegex(ValueError, "exactly one target FINAL"):
            solve_rendered(prompt, target)

    def test_high_similarity_is_audited(self):
        text = "target record alpha beta gamma delta epsilon evidence one two three four five"
        panel = [{"id": "candidate", "state": text}]
        protected = [{"id": "train", "state": text}]
        self.assertEqual(_near_overlap(panel, protected)[0]["other_id"], "train")

    def test_missing_human_review_blocks_quality_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            panel = Path(directory)
            manifest = {"automated_audit_sha256": "a" * 64}
            audit = {"automated_status": "passed"}
            gate = _review_gate(panel, manifest, audit, None)
            self.assertEqual(gate["status"], "blocked")
            self.assertIn("review_pending", gate["reason"])

    def test_native_receipt_cannot_be_bypassed_by_inline_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt, _ = render(
                make_spec(
                    bytes(range(32)), "dev", "choice", "near_distractor", "build", 0
                )
            )
            prompts = root / "prompts.jsonl"
            prompts.write_text(json.dumps(prompt) + "\n")
            prediction = root / "pred.jsonl"
            row = {
                "id": prompt["id"],
                "answers": {},
                "model_id": "model",
                "model_revision": "revision",
                "source_input_sha256": input_digest(prompt),
                "input_sha256": input_digest(prompt),
                "model_sha256": "b" * 64,
                "adapter_sha256": "a" * 64,
            }
            prediction.write_text(json.dumps(row) + "\n")
            manifest = root / "pred.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model_id": "model",
                        "model_revision": "revision",
                        "predictions_sha256": sha_file(prediction),
                        "input_sha256": sha_file(prompts),
                        "input_items": 1,
                        "counts": {"items": 1, "questions": 1},
                        "adapter_version": "native-v1",
                        "model_sha256": "c" * 64,
                        "adapter_sha256": "a" * 64,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "wrong-model"):
                _prediction_rows(
                    prediction, prompts, [prompt], "model", "revision", manifest
                )

    def test_release_roster_binds_exact_model_and_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}\n")
            model = {
                "model_id": "m",
                "model_revision": "r",
                "model_sha256": "a" * 64,
                "adapter_sha256": "b" * 64,
            }
            lock = root / "lock.json"
            lock.write_text(
                json.dumps(
                    {
                        "lock_version": "jevarena-authored-selection-lock/1",
                        "selection_basis": "development_only",
                        "panel_manifest_sha256": sha_file(root / "manifest.json"),
                        "models": [model],
                    }
                )
            )
            self.assertEqual(
                _check_release_lock(lock, root, "m", "r", model), sha_file(lock)
            )
            with self.assertRaisesRegex(ValueError, "mismatched"):
                _check_release_lock(lock, root, "m", "different", model)

    def test_template_cell_interval_does_not_claim_item_independence(self):
        rows = []
        for kind in TYPES:
            for challenge in CHALLENGES:
                for domain in DOMAINS:
                    for _ in range(2):
                        rows.append(
                            {
                                "family": f"{kind}/{challenge}",
                                "domain": domain,
                                "status": "ok",
                                "correct": domain != "archive",
                            }
                        )
        result = _bootstrap_macro(rows, "a" * 64, replicates=200)
        self.assertEqual(result["scenario_bootstrap_ci95"]["replicates"], 200)
        self.assertLess(
            result["domain_cell_bootstrap_ci95"]["lower"],
            result["domain_cell_bootstrap_ci95"]["upper"],
        )


if __name__ == "__main__":
    unittest.main()
