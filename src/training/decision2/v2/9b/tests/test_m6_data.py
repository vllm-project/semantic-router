import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

from training.model.data import (
    INPUT_FIELDS,
    canonical,
    digest,
    file_sha256,
)  # noqa: E402
from v2.data.build_a0_variants import native_prompt  # noqa: E402
from v2.data.m3.waves import prompt_line  # noqa: E402
from v2.data.replay_targets import collector_digest  # noqa: E402
from lux9b import m6_data  # noqa: E402

RULE = {
    "pools": ["H1", "A7q", "A7g", "A0s-strict"],
    "human_sources": {
        "H1": ["h1_train"],
        "A7q": ["*"],
        "A0s-strict": ["google_goemotions_official_train", "legacy:stage3_replay"],
    },
    "replay_source": "legacy:stage3_replay",
}


def row(
    rid,
    group,
    source="h1_train",
    family="fam",
    split="train",
    task="choice",
    state=None,
    **extra,
):
    keys = ("false", "true") if task == "noul" else ("a", "b")
    out = {
        "id": rid,
        "state": state or f"state {rid}",
        "instructions": "Pick one.",
        "options": [{"key": k, "description": f"option {k}"} for k in keys],
        "label": 0,
        "task_type": task,
        "family": family,
        "group_id": group,
        "language": "en",
        "split": split,
        "source": source,
        "evaluation_role": split,
        "render_template": "t",
        "audit_metadata": {},
        **extra,
    }
    out["input_sha256"] = digest({f: out[f] for f in INPUT_FIELDS})
    return out


def target(r, p=0.5):
    return {
        "id": r["id"],
        "input_sha256": r["input_sha256"],
        "teacher_probs": {"a": p, "b": 1 - p},
    }


def write(path: Path, records) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(r) + "\n" for r in records), encoding="utf-8")
    return {"sha256": file_sha256(path)}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pools = {
            "h1-covered": "H1",
            "h1-wave": "H1",
            "h1-long": "H1",
            "a7q-wave": "A7q",
            "a7g-typed": "A7g",
            "a0s-goemo": "A0s-strict",
            "a0s-prog": "A0s-strict",
            "a0s-replay-h": "A0s-strict",
            "a0s-replay-t": "A0s-strict",
            "h1-qa": "H1",
        }
        sources = {
            "h1-qa": "csqa_train",
            "a0s-goemo": "google_goemotions_official_train",
            "a0s-prog": "decision2_programmatic_original_v1",
            "a0s-replay-h": "legacy:stage3_replay",
            "a0s-replay-t": "legacy:stage3_replay",
            "a7g-typed": "dec10:generated_stage4_v2",
        }
        self.rows = []
        for i, (rid, pool) in enumerate(self.pools.items()):
            extra = {"upstream_label": "x"} if rid == "a0s-replay-h" else {}
            state = "L" * 400 if rid == "h1-long" else None
            self.rows.append(
                row(rid, f"g{i}", sources.get(rid, "h1_train"), state=state, **extra)
            )
        self.rows.sort(key=lambda r: r["id"])
        self.by_id = {r["id"]: r for r in self.rows}
        t = self.tmp
        self.spec = {
            "name": "m6-test",
            "seed": "s",
            "max_length": 100,
            "soft_target_rule": RULE,
            "x60": {
                "train": {
                    "file": "t:x60/train.jsonl",
                    **write(t / "x60/train.jsonl", self.rows),
                },
                "teacher": {
                    "file": "t:x60/teacher.jsonl",
                    **write(
                        t / "x60/teacher.jsonl", [target(r, 0.9) for r in self.rows]
                    ),
                },
            },
            "ids": {
                "file": "t:ids.jsonl",
                **write(
                    t / "ids.jsonl",
                    [
                        {
                            "id": r["id"],
                            "pool": self.pools[r["id"]],
                            "source": r["source"],
                            "native": 10,
                            "task_type": r["task_type"],
                            "language": "en",
                        }
                        for r in self.rows
                    ],
                ),
            },
            "aj_targets": [
                {
                    "name": "aj-a0s-strict",
                    "file": "t:aj/a0s.jsonl",
                    **write(t / "aj/a0s.jsonl", [target(self.by_id["a0s-goemo"], 0.2)]),
                },
                {
                    "name": "aj-m",
                    "file": "t:aj/m.jsonl",
                    **write(t / "aj/m.jsonl", [target(self.by_id["h1-covered"], 0.3)]),
                },
            ],
            "aj_prompts": [
                {
                    "file": "t:aj/p.jsonl",
                    **write(
                        t / "aj/p.jsonl",
                        [native_prompt(self.by_id["h1-covered"]) | {"state": "x" * 60}],
                    ),
                },
            ],
            "split_dir": "t:split",
        }
        self.roots = {"t": t}

    def split(self):
        spec = self.tmp / "spec.json"
        spec.write_text(json.dumps(self.spec))
        with contextlib.redirect_stdout(io.StringIO()):
            m6_data.main(
                [
                    "split",
                    "--spec",
                    str(spec),
                    "--root",
                    f"t={self.tmp}",
                    "--output-dir",
                    str(self.tmp / "split"),
                ]
            )
        return json.loads((self.tmp / "split/manifest.json").read_text())


class RuleTest(Fixture):
    def test_pool_rule(self):
        got = {
            r["id"]: m6_data.human_rated(r, self.pools[r["id"]], self.spec)
            for r in self.rows
        }
        self.assertEqual(
            sorted(k for k, v in got.items() if v),
            [
                "a0s-goemo",
                "a0s-replay-h",
                "a7q-wave",
                "h1-covered",
                "h1-long",
                "h1-wave",
            ],
        )
        with self.assertRaises(ValueError):
            m6_data.human_rated(self.rows[0], "Z9", self.spec)


class SplitTest(Fixture):
    def test_split_covers_waves_and_cap(self):
        m = self.split()
        s_ids = (self.tmp / "split/S.ids.txt").read_text().split()
        self.assertEqual(
            s_ids, ["a0s-goemo", "a0s-replay-h", "a7q-wave", "h1-covered", "h1-wave"]
        )
        wave = [
            json.loads(line)
            for line in (self.tmp / "split/wave.rows.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [r["id"] for r in wave], ["a0s-replay-h", "a7q-wave", "h1-wave"]
        )
        prompts = (
            (self.tmp / "split/wave.prompts.jsonl")
            .read_text()
            .splitlines(keepends=True)
        )
        self.assertEqual(prompts[1], prompt_line(self.by_id["a7q-wave"]))
        for line, r in zip(prompts, wave):
            self.assertEqual(
                collector_digest(json.loads(line)), collector_digest(native_prompt(r))
            )
        self.assertEqual(m["excluded_over_cap_rows"], 1)
        self.assertEqual(
            m["s_rows_by_production_file"], {"aj-a0s-strict": 1, "aj-m": 1}
        )
        self.assertEqual(
            m["files"]["S.ids.txt"], file_sha256(self.tmp / "split/S.ids.txt")
        )


class KaTest(Fixture):
    def wave(self, ids, prompts_sha):
        records = [target(self.by_id[i], 0.1) for i in ids]
        path = self.tmp / "wave/targets.jsonl"
        write(path, records)
        report = self.tmp / "wave/report.json"
        report.write_text(
            json.dumps(
                {"content_sha256": file_sha256(path), "prompts_sha256": prompts_sha}
            )
        )
        return path, report

    def test_ka_teacher_uses_autojev_on_s_only(self):
        m = self.split()
        path, report = self.wave(
            ["a0s-replay-h", "a7q-wave", "h1-wave"], m["files"]["wave.prompts.jsonl"]
        )
        out = m6_data.ka(self.spec, self.roots, path, report, self.tmp / "ka")
        teacher = {
            json.loads(x)["id"]: json.loads(x)
            for x in (self.tmp / "ka/teacher.jsonl").read_text().splitlines()
        }
        self.assertEqual(teacher["h1-covered"]["teacher_probs"]["a"], 0.3)
        self.assertEqual(teacher["a0s-goemo"]["teacher_probs"]["a"], 0.2)
        self.assertEqual(teacher["h1-wave"]["teacher_probs"]["a"], 0.1)
        for own in ("h1-long", "h1-qa", "a7g-typed", "a0s-prog", "a0s-replay-t"):
            self.assertEqual(teacher[own]["teacher_probs"]["a"], 0.9)
        self.assertEqual(out["train_sha256"], self.spec["x60"]["train"]["sha256"])
        self.assertEqual(out["teacher_origin"]["own-lux"], 5)

    def test_ka_refuses_a_wave_that_misses_rows(self):
        m = self.split()
        path, report = self.wave(
            ["a7q-wave", "h1-wave"], m["files"]["wave.prompts.jsonl"]
        )
        with self.assertRaises(ValueError):
            m6_data.ka(self.spec, self.roots, path, report, self.tmp / "ka")

    def test_ka_refuses_a_report_for_other_prompts(self):
        self.split()
        path, report = self.wave(["a0s-replay-h", "a7q-wave", "h1-wave"], "0" * 64)
        with self.assertRaises(ValueError):
            m6_data.ka(self.spec, self.roots, path, report, self.tmp / "ka")


class KhTest(Fixture):
    def setUp(self):
        super().setUp()
        t = self.tmp
        x60 = [row(f"h1-x{i:02d}", f"xg{i:02d}") for i in range(30)]
        self.spec["x60"] = {
            "train": {"file": "t:k/train.jsonl", **write(t / "k/train.jsonl", x60)},
            "teacher": {
                "file": "t:k/teacher.jsonl",
                **write(t / "k/teacher.jsonl", [target(r, 0.9) for r in x60]),
            },
        }
        self.spec["ids"] = {
            "file": "t:k/ids.jsonl",
            **write(
                t / "k/ids.jsonl",
                [
                    {
                        "id": r["id"],
                        "pool": "H1",
                        "source": r["source"],
                        "native": 10,
                        "task_type": "choice",
                        "language": "en",
                    }
                    for r in x60
                ],
            ),
        }
        hs1 = [
            row(
                "f3-1",
                "hg1",
                "decision2_hardskills_hs1",
                family="hs1_unmet_condition",
                task="noul",
            ),
            row(
                "f3-2",
                "hg1",
                "decision2_hardskills_hs1",
                family="hs1_unmet_condition",
                task="noul",
            ),
            row("f1-1", "hq1", "decision2_hardskills_hs1", family="hs1_quote_check"),
            row("f1-2", "hq2", "decision2_hardskills_hs1", family="hs1_quote_check"),
            row("f2-1", "hp1", "decision2_hardskills_hs1", family="hs1_policy_packet"),
        ]
        self.spec["hs1"] = {
            "train": {"file": "t:hs1.jsonl", **write(self.tmp / "hs1.jsonl", hs1)},
            "f1_share_num": 1,
            "f1_share_den": 2,
        }
        sel = [row("sel-1", "sg1", split="select")]
        self.spec["isolation"] = [
            {
                "role": "select",
                "file": "t:sel.jsonl",
                **write(self.tmp / "sel.jsonl", sel),
            }
        ]
        (self.tmp / "reg.json").write_text(json.dumps({"sources": []}))
        self.spec["denied_sources"] = {"c1_registry": "t:reg.json", "extra": ["xnli"]}
        self.spec["keep_tolerance"] = 0.01

    def lengths(self, rows, tokenizer, workers):
        return [10 for _ in rows]

    def test_kh_substitutes_the_block_at_matched_tokens(self):
        with mock.patch.object(m6_data, "token_lengths", self.lengths):
            m = m6_data.kh(
                self.spec, self.roots, Path("/nonexistent"), 1, self.tmp / "kh"
            )
        self.assertEqual(m["block"]["f3_rows"], 2)
        self.assertEqual(m["block"]["f1_groups_kept"], 1)
        self.assertEqual(m["block"]["rows"], 3)
        self.assertEqual(m["x60_native_tokens"], 300)
        self.assertEqual(m["kept_native_tokens"], 270)
        self.assertEqual(m["train_native_tokens"], 300)
        train = [
            json.loads(x)
            for x in (self.tmp / "kh/train.jsonl").read_text().splitlines()
        ]
        teacher = [
            json.loads(x)
            for x in (self.tmp / "kh/teacher.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            {r["id"] for r in teacher},
            {r["id"] for r in train if not r["id"].startswith("f")},
        )
        self.assertEqual(len(teacher), 27)
        self.assertNotIn("f2-1", {r["id"] for r in train})

    def test_kh_refuses_a_block_row_shared_with_select(self):
        leaked = row("sel-1", "sg1", split="select")
        leaked_train = dict(
            leaked,
            split="train",
            evaluation_role="train",
            id="f3-9",
            family="hs1_unmet_condition",
            source="decision2_hardskills_hs1",
        )
        hs1 = [
            json.loads(x) for x in (self.tmp / "hs1.jsonl").read_text().splitlines()
        ] + [leaked_train]
        self.spec["hs1"]["train"] = {
            "file": "t:hs1.jsonl",
            **write(self.tmp / "hs1.jsonl", hs1),
        }
        with mock.patch.object(m6_data, "token_lengths", self.lengths):
            with self.assertRaises(ValueError):
                m6_data.kh(
                    self.spec, self.roots, Path("/nonexistent"), 1, self.tmp / "kh"
                )


if __name__ == "__main__":
    unittest.main()
