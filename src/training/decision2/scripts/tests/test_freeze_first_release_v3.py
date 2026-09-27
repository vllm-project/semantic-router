"""Synthetic, gold-free tests for exclusive JevArena v3 pre-key sealing."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from jev_arena.arena_v3 import REQUIRED_PROTOCOL_SOURCES
from scripts.freeze_first_release_v3 import create_freeze
from scripts.plan_first_release_v3 import SOURCE_FILES, pair_digest

SOURCE_ROOT = Path(__file__).resolve().parents[2]
WHEN = datetime(2026, 9, 2, 1, 30, tzinfo=timezone.utc)


def write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root: Path) -> dict:
    pairs = [
        {
            "candidate": "d2-4b",
            "comparator": "nox",
            "size_relation": "same",
            "rationale": "",
        }
    ]
    roster = [
        {
            "key": key,
            "group": group,
            "model_id": f"example/{key}",
            "revision": f"revision-{key}",
            "native_model_sha256": "1" * 64,
            "adapter_sha256": "2" * 64,
            "calibration_sha256": "3" * 64,
        }
        for key, group in (
            ("nox", "decision1"),
            ("eikos4b", "open"),
            ("d2-4b", "decision2"),
        )
    ]
    plan = {
        "plan_version": "decision2-first-release-v3-plan/1",
        "source_root": str(SOURCE_ROOT),
        "source_sha256": {
            name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
            for name in REQUIRED_PROTOCOL_SOURCES
        },
        "candidate_freeze_sha256": "4" * 64,
        "gate_document_sha256": "5" * 64,
        "formula": "100*sqrt(T*H)",
        "comparison_pairs": pairs,
        "comparison_pairs_sha256": pair_digest(pairs),
        "paired_ci_commands_after_prekey_freeze": [
            {
                **pairs[0],
                "command": "python -m jev_arena.compare_v3 --replicates 5000 --seed 20260927",
            }
        ],
        "model_roster": roster,
        "css_prompts": {"sha256": "6" * 64},
        "public_panel": {"prompts_sha256": "7" * 64},
    }
    audit = {
        "status": "gold_free_prekey_predictions_verified",
        "comparison_pairs_sha256": pair_digest(pairs),
        "prompt_sha256": {
            "typed": "8" * 64,
            "css": "6" * 64,
            "public": "7" * 64,
        },
        "models": {
            row["key"]: {
                "typed": "9" * 64,
                "css": "a" * 64,
                "public": "b" * 64,
            }
            for row in roster
        },
        "raw_hashes_sha256": "c" * 64,
    }
    plan_path, audit_path = root / "plan.json", root / "audit.json"
    plan_sha, audit_sha = write(plan_path, plan), write(audit_path, audit)
    chronology_path = root / "chronology.json"
    chronology_sha = write(
        chronology_path,
        {
            "schema_version": "jevarena-v3-prekey-chronology/1",
            "candidate_lock": {
                "at_utc": "2026-09-02T01:00:00+00:00",
                "sha256": "4" * 64,
            },
            "prediction_seal": {
                "at_utc": "2026-09-02T01:10:00+00:00",
                "sha256": "c" * 64,
            },
            "audit_seal": {
                "at_utc": "2026-09-02T01:20:00+00:00",
                "sha256": audit_sha,
            },
        },
    )
    return {
        "plan": plan,
        "audit": audit,
        "args": {
            "plan_path": plan_path,
            "plan_sha256": plan_sha,
            "audit_path": audit_path,
            "audit_sha256": audit_sha,
            "chronology_path": chronology_path,
            "chronology_sha256": chronology_sha,
            "typed_gold_sha256": "d" * 64,
            "css_gold_sha256": "e" * 64,
            "output": root / "freeze.json",
            "now": WHEN,
        },
    }


class FirstReleaseV3FreezeTest(unittest.TestCase):
    def test_full_gold_free_audit_is_required_and_receipt_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = fixture(Path(temporary))
            with patch(
                "scripts.freeze_first_release_v3.audit_prekey_predictions",
                return_value=copy.deepcopy(state["audit"]),
            ) as full_audit:
                digest, when = create_freeze(**state["args"])
            full_audit.assert_called_once()
            self.assertEqual(len(digest), 64)
            self.assertEqual(when, WHEN.isoformat(timespec="microseconds"))
            self.assertEqual(
                stat.S_IMODE(state["args"]["output"].stat().st_mode), 0o600
            )
            frozen = json.loads(state["args"]["output"].read_text())
            self.assertEqual(frozen["panels"]["typed_gold_sha256"], "d" * 64)
            self.assertEqual(
                frozen["models"]["d2-4b"]["predictions_sha256"]["public"],
                "b" * 64,
            )
            with self.assertRaises(FileExistsError):
                create_freeze(**state["args"])

    def test_changed_saved_audit_fails_before_receipt_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = fixture(Path(temporary))
            changed = copy.deepcopy(state["audit"])
            changed["models"]["nox"]["public"] = "0" * 64
            with patch(
                "scripts.freeze_first_release_v3.audit_prekey_predictions",
                return_value=changed,
            ):
                with self.assertRaisesRegex(ValueError, "Saved audit differs"):
                    create_freeze(**state["args"])
            self.assertFalse(state["args"]["output"].exists())

    def test_chronology_tamper_fails_before_receipt_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = fixture(Path(temporary))
            path = state["args"]["chronology_path"]
            chain = json.loads(path.read_text())
            chain["prediction_seal"]["at_utc"] = "2026-09-02T00:59:00+00:00"
            state["args"]["chronology_sha256"] = write(path, chain)
            with patch(
                "scripts.freeze_first_release_v3.audit_prekey_predictions",
                return_value=copy.deepcopy(state["audit"]),
            ):
                with self.assertRaisesRegex(ValueError, "not strictly ordered"):
                    create_freeze(**state["args"])
            self.assertFalse(state["args"]["output"].exists())

    def test_pair_source_or_formula_tamper_fails_closed(self) -> None:
        for mutation, expected in (
            ("pair", "comparison mapping changed"),
            ("source", "protocol source changed"),
            ("formula", "formula or paired bootstrap"),
        ):
            with self.subTest(
                mutation=mutation
            ), tempfile.TemporaryDirectory() as temporary:
                state = fixture(Path(temporary))
                plan = state["plan"]
                if mutation == "pair":
                    plan["comparison_pairs_sha256"] = "0" * 64
                elif mutation == "source":
                    plan["source_sha256"]["benchmark/score.py"] = "0" * 64
                else:
                    plan["formula"] = "100*T"
                state["args"]["plan_sha256"] = write(state["args"]["plan_path"], plan)
                with patch(
                    "scripts.freeze_first_release_v3.audit_prekey_predictions",
                    return_value=copy.deepcopy(state["audit"]),
                ):
                    with self.assertRaisesRegex(ValueError, expected):
                        create_freeze(**state["args"])
                self.assertFalse(state["args"]["output"].exists())

    def test_invalid_gold_digest_and_public_output_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = fixture(Path(temporary))
            state["args"]["css_gold_sha256"] = "not-a-digest"
            with self.assertRaisesRegex(ValueError, "css_gold_sha256"):
                create_freeze(**state["args"])
            state["args"]["css_gold_sha256"] = "e" * 64
            state["args"]["output"] = SOURCE_ROOT / "accidental-public-freeze.json"
            with self.assertRaisesRegex(ValueError, "source tree"):
                create_freeze(**state["args"])

    def test_generator_source_is_part_of_frozen_protocol_closure(self) -> None:
        name = "scripts/freeze_first_release_v3.py"
        self.assertIn(name, SOURCE_FILES)
        self.assertIn(name, REQUIRED_PROTOCOL_SOURCES)


if __name__ == "__main__":
    unittest.main()
