from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from training.model.data import validate_row
from v2.data.a7 import admit, build_a7, inventory, lengths


def _raw(
    index: int,
    kind: str = "choice",
    *,
    source: object = "stage4-general-composition-v2",
    family: str = "stage4_boolean",
    group: str | None = None,
    state: str | None = None,
    label: int = 0,
    keys: list[str] | None = None,
) -> dict:
    if keys is None:
        keys = {"noul": ["false", "true"], "score": ["0", "1", "2"]}.get(
            kind, ["a", "b", "c"]
        )
    if isinstance(source, str) and source == "stage4-general-composition-v2":
        source = {"type": "objective_generator", "generator": source, "seed": 1}
    return {
        "id": f"row-{index}",
        "state": state or f"state number {index} with enough characters to link",
        "instructions": "Decide.",
        "options": [{"key": key, "description": f"option {key}"} for key in keys],
        "label": label,
        "task_type": kind,
        "family": family,
        "group_id": group or f"g{index}",
        "language": "en",
        "source": source,
        "split": "train",
    }


def _write(path: Path, rows: list[dict]) -> dict:
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(data)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(rows),
    }


class LineageTest(unittest.TestCase):
    def test_replay_uses_original_source(self) -> None:
        replay = {
            "type": "stage3_replay",
            "original_source": "CLINC150 upstream train; CC-BY-3.0",
        }
        self.assertEqual(build_a7.lineage(replay), "CLINC150 upstream train; CC-BY-3.0")
        nested = {
            "type": "stage3_replay",
            "original_source": "{'generator': 'stage3-src/transitions.py', 'seed': 1}",
        }
        self.assertEqual(build_a7.lineage(nested), "stage3-src/transitions.py")

    def test_sub_arm_assignment(self) -> None:
        self.assertEqual(build_a7.sub_arm_of("natural8k", "dec10:snli_train"), "A7h")
        self.assertEqual(
            build_a7.sub_arm_of("stage4v2", "dec10:multinli_nonfiction_train"), "A7m"
        )
        self.assertEqual(build_a7.sub_arm_of("stage1", "dec10:banking77_train"), "A7i")
        self.assertEqual(
            build_a7.sub_arm_of("stage4v2", "dec10:generated_stage1_3"), "A7p"
        )
        self.assertEqual(
            build_a7.sub_arm_of("stage3", "dec10:generated_stage1_3"), "A7o"
        )
        with self.assertRaises(build_a7.Excluded):
            build_a7.sub_arm_of("natural8k", "dec10:banking77_train")


class NormalizeTest(unittest.TestCase):
    def test_row_validates_and_keeps_provenance(self) -> None:
        raw = _raw(1, "noul", keys=["true", "false"])
        raw["upstream_label"] = 3
        row = build_a7.normalize_row(raw, "stage4v2", "f" * 64)
        validate_row(row, "train")
        self.assertEqual(row["id"], "a7:stage4v2:row-1")
        self.assertEqual(row["source"], "dec10:generated_stage4_v2")
        origin = row["audit_metadata"]["a7"]
        self.assertEqual(origin["sub_arm"], "A7g")
        self.assertEqual(
            origin["original_fields"], {"upstream_label": 3, "split": "train"}
        )
        self.assertEqual(row["render_template"], "dec10_unspecified")

    def test_rules_exclude_opaque_keys_and_legacy_numeric_choice(self) -> None:
        opaque = _raw(
            2, "noul", source="objective_generator", keys=["result_0", "result_1"]
        )
        with self.assertRaises(build_a7.Excluded) as caught:
            build_a7.normalize_row(opaque, "stage1", "f" * 64)
        self.assertEqual(caught.exception.reason, "opaque_noul_keys")
        numeric = _raw(3, source="objective_generator", family="stage3_arithmetic_sum")
        with self.assertRaises(build_a7.Excluded) as caught:
            build_a7.normalize_row(numeric, "stage3", "f" * 64)
        self.assertEqual(caught.exception.reason, "legacy_numeric_choice")
        noul = _raw(
            4, "noul", source="objective_generator", family="stage3_arithmetic_sum"
        )
        self.assertEqual(
            build_a7.normalize_row(noul, "stage3", "f" * 64)["family"], noul["family"]
        )
        replayed = _raw(
            5,
            family="stage4_replay_arithmetic",
            source={"type": "stage3_replay", "original_source": "objective_generator"},
        )
        with self.assertRaises(build_a7.Excluded):
            build_a7.normalize_row(replayed, "stage4v2", "f" * 64)


class BuildTest(unittest.TestCase):
    def test_dedup_conflict_components_isolation_and_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = _raw(
                10,
                source={
                    "type": "stage3_replay",
                    "original_source": "objective_generator",
                },
                family="stage4_replay_policy",
                group="s4g",
            )
            older = dict(shared, id="old-10", group_id="s3g")
            sibling = _raw(
                11, source="objective_generator", family="policy", group="s3g"
            )
            conflict_a = _raw(
                12, group="c1", state="identical conflicting state text here"
            )
            conflict_b = dict(conflict_a, id="row-12b", label=1, group_id="c2")
            a0_row = _raw(13, group="a0g")
            touching = _raw(
                14, group="a0g", state="another state long enough to be linked"
            )
            held = _raw(15, group="selectgroup")
            natural = _raw(
                16, source={"dataset": "snli"}, family="natural_snli", group="snli:x"
            )
            rows4 = [shared, conflict_a, conflict_b, touching, held]
            spec = {
                "version": build_a7.VERSION,
                "sources": [
                    {"name": "stage4v2", **_write(root / "s4.jsonl", rows4)},
                    {"name": "natural8k", **_write(root / "nat.jsonl", [natural])},
                    {"name": "stage3", **_write(root / "s3.jsonl", [older, sibling])},
                ],
                "isolation_train": [
                    {
                        "name": "A0",
                        **_write(root / "a0.jsonl", [dict(a0_row, source="x")]),
                    }
                ],
                "isolation_holdout": [
                    {
                        "name": "SELECT",
                        **_write(
                            root / "sel.jsonl", [dict(_raw(99), group_id="selectgroup")]
                        ),
                    }
                ],
                "views": [
                    {
                        "name": "v",
                        **_write(
                            root / "view.jsonl", [shared, natural, conflict_a, _raw(77)]
                        ),
                    }
                ],
            }
            result = build_a7.build(spec, commit="abc")
            manifest = result["manifest"]
            self.assertEqual(
                manifest["duplicates_dropped"],
                {"stage4v2<-stage3": 1, "stage4v2<-stage4v2": 1},
            )
            self.assertEqual(sum(manifest["label_conflicts_dropped"].values()), 1)
            self.assertEqual(sum(manifest["isolation_dropped"].values()), 1)
            rows = {
                row["id"]: (name, part)
                for name, parts in result["sub_arms"].items()
                for part, items in parts.items()
                for row in items
            }
            self.assertEqual(rows["a7:stage4v2:row-10"][0], "A7p")
            self.assertEqual(rows["a7:stage3:row-11"][0], "A7o")
            self.assertEqual(rows["a7:stage4v2:row-10"][1], rows["a7:stage3:row-11"][1])
            self.assertEqual(rows["a7:stage4v2:row-14"][1], "train")
            self.assertNotIn("a7:stage4v2:row-12", rows)
            self.assertNotIn("a7:stage4v2:row-15", rows)
            view = result["views"]["v"]
            self.assertEqual(view["covered"], 2)
            self.assertEqual(
                view["uncovered"], {"label_conflict": 1, "not_in_a7_sources": 1}
            )
            out = root / "out"
            hashes = build_a7.write_outputs(result, out)
            self.assertTrue((out / "build-manifest.json").exists())
            self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_aho_rule_is_deterministic(self) -> None:
        keys = [f"group-{index}" for index in range(2000)]
        share = sum(build_a7.is_aho(key) for key in keys) / len(keys)
        self.assertTrue(0.07 < share < 0.13)
        self.assertEqual(
            [build_a7.is_aho(key) for key in keys[:50]],
            [build_a7.is_aho(key) for key in keys[:50]],
        )


class AdmitTest(unittest.TestCase):
    def _row(self, index: int, family: str, group: str) -> dict:
        return build_a7.normalize_row(
            _raw(index, family=family, group=group), "stage4v2", "f" * 64
        )

    def test_quarantine_budget_and_shortcut_cells(self) -> None:
        rows = [
            self._row(1, "stage4_boolean", "q"),
            self._row(2, "stage4_boolean", "long"),
            self._row(3, "stage4_scope", "cellgroup"),
            self._row(4, "stage4_relations", "ok"),
        ]
        overlap = {
            "groups": {
                "q": {"roles": ["css15_goldfree"]},
                "ok": {"roles": ["rights_clean_train"]},
            }
        }
        shortcut = {
            "views": {
                "option_only": {
                    "by_task_family": {
                        "choice": {
                            "stage4_scope": {"n": 40, "exceeds_margin": True},
                            "stage4_relations": {"n": 10, "exceeds_margin": True},
                        }
                    }
                }
            }
        }
        lengths_by_id = {
            row["id"]: {
                "qwen3.5-0.8b-base@dc7cdfe2": (
                    9000 if row["group_id"] == "long" else 100
                ),
                "kai-0.6b@7185f514": 2000,
            }
            for row in rows
        }
        kept, report = admit.admit_sub_arm(
            "A7g",
            {"train": rows, "aho": []},
            overlap,
            shortcut,
            lengths_by_id,
            {"rights_clean_train"},
        )
        self.assertEqual([row["id"] for row in kept["train"]], ["a7:stage4v2:row-4"])
        self.assertEqual(
            report["removed_totals"],
            {"overlap_quarantine": 1, "native_budget": 1, "shortcut_cell": 1},
        )
        self.assertEqual(
            report["report_only_groups_by_role"], {"rights_clean_train": 1}
        )
        self.assertEqual(report["kai_over_1024_rows_flagged"], 1)
        human, _ = admit.admit_sub_arm(
            "A7h",
            {"train": rows[2:], "aho": []},
            {"groups": {}},
            shortcut,
            lengths_by_id,
            set(),
        )
        self.assertEqual(len(human["train"]), 2)

    def test_resolve_view_drops_removed_members(self) -> None:
        view = {
            "view": "v",
            "source_file_sha256": "x",
            "rows": 3,
            "used_by": [],
            "covered": 2,
            "uncovered": {"a": 1},
            "members": [
                {"id": "k", "part": "train", "sub_arm": "A7g"},
                {"id": "gone", "part": "aho", "sub_arm": "A7g"},
            ],
        }
        resolved = admit.resolve_view(view, {"k": "aho"})
        self.assertEqual(
            resolved["members"], [{"id": "k", "part": "aho", "sub_arm": "A7g"}]
        )
        self.assertEqual(resolved["removed_by_admission"], 1)


class LengthsAndInventoryTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "needs the runtime image")
    def test_native_length_matches_segment_sum(self) -> None:
        class CharTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return list(range(len(text)))

        raw = _raw(1, "score", keys=["result_0", "result_1"])
        from training.model.decision_model import segments

        prefix, options, suffix = segments(raw)
        expected = len(prefix) + sum(map(len, options)) + len(suffix)
        self.assertEqual(lengths.native_length(raw, CharTokenizer()), expected)

    def test_census_counts_levels_lineage_and_duplicates(self) -> None:
        rows = [_raw(1, "score"), _raw(2, "score", keys=["0", "1", "2", "3"]), _raw(3)]
        rows.append(dict(rows[2], id="dup"))
        report, hashes = inventory.census(rows, {"row-1": {"id": "row-1", "tok": 5}})
        self.assertEqual(report["score_levels"], {"3": 1, "4": 1})
        self.assertEqual(report["duplicate_inputs"], 1)
        self.assertEqual(report["lineages"], {"stage4-general-composition-v2": 4})
        self.assertEqual(report["tokens"]["tok"]["type:score"], 5)
        self.assertEqual(len(hashes), 3)

    def test_census_reads_kai_native_rows(self) -> None:
        row = {
            "id": "n1",
            "state_text": "text",
            "language": "en",
            "source_id": "snli",
            "domain": "caption_inference",
            "component_id": "c1",
            "question": {"type": "Score", "levels": [{}, {}, {}]},
            "target": {"probabilities": [0.0, 0.4, 0.6]},
            "provenance": {"label_origin": "original human label"},
        }
        noul = dict(
            row, id="n2", question={"type": "Noul"}, target={"probability": 1.0}
        )
        report, _ = inventory.census([row, noul], None)
        self.assertEqual(report["types"], {"noul": 1, "score": 1})
        self.assertEqual(report["score_levels"], {"3": 1})
        self.assertEqual(report["soft_label_fields"], {"soft_target": 1})
        self.assertEqual(report["label_origins"], {"original human label": 2})
        self.assertEqual(report["lineages"], {"snli": 2})


if __name__ == "__main__":
    unittest.main()
