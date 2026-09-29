"""M8 `publish`: path-free Score offsets with the scored offsets (stdlib)."""

from __future__ import annotations

import copy
import importlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from training.model.score_bias import load_score_bias
from v2.release import build, layout

m8 = importlib.import_module("v2.06b.m8_scorebias")

ROOT = Path(__file__).resolve().parents[3]
SCORED = ROOT / "v2/06b/records/m8/dev/s5-b05/score_bias.json"
SCORED_SHA256 = "03f0c9c777de95f8ea56112c05a3c29521687609aa67c9c7ca466c14bf8dadae"
PUBLIC = ROOT / "v2/06b/records/m8/release/s5-b05.score_bias.public.json"


def keys_named(value, name):
    if isinstance(value, dict):
        return (name in value) + sum(keys_named(v, name) for v in value.values())
    if isinstance(value, list):
        return sum(keys_named(v, name) for v in value)
    return 0


def private(text):
    return any(p.search(text) for p in (build.PRIVATE, build.IPV4, build.SECRET))


def publish(source: Path, output: Path) -> dict:
    with redirect_stdout(io.StringIO()) as stream:
        m8.main(["publish", "--score-bias", str(source), "--output", str(output)])
    return json.loads(stream.getvalue())


class PublishTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.dir = Path(self.scratch.name)

    def tearDown(self):
        self.scratch.cleanup()

    def test_scored_record_publishes_with_identical_offsets(self):
        self.assertEqual(layout.sha_file(SCORED), SCORED_SHA256)
        self.assertTrue(private(SCORED.read_text()))
        output = self.dir / "public.json"
        printed = publish(SCORED, output)
        source, public = json.loads(SCORED.read_text()), json.loads(output.read_text())
        for key in ("format", "model_sha256", "offsets"):
            self.assertEqual(public[key], source[key])
        self.assertEqual(
            load_score_bias(output, source["model_sha256"])[0],
            load_score_bias(SCORED, source["model_sha256"])[0],
        )
        fit = public["fit"]
        self.assertEqual(fit["scored_file_sha256"], SCORED_SHA256)
        self.assertEqual(printed["scored_file_sha256"], SCORED_SHA256)
        self.assertEqual(printed["sha256"], layout.sha_file(output))
        self.assertEqual(keys_named(public, "path"), 0)
        self.assertFalse(private(output.read_text()))
        self.assertEqual(
            fit["inputs_sha256"],
            {
                "score5t_check_gold": source["fit"]["inputs"]["score5t_check_gold"][
                    "sha256"
                ],
                "score5t_fit_gold": source["fit"]["inputs"]["score5t_fit_gold"][
                    "sha256"
                ],
                "score5t_panel/gold/score5t-dev.gold.jsonl": "31abc1a5d7820e06961a903327c76dc4503b46987f0db58b645e0e6562e08aed",
                "score5t_panel/goldfree/score5t-dev.prompts.jsonl": "8e35bfffc2c3b8e4252d3c39ec1c65054250a3ddf80159d219a16b41bca1d93c",
                "score5t_predictions": source["fit"]["inputs"]["score5t_predictions"][
                    "sha256"
                ],
            },
        )
        for key in m8.PUBLIC_FIT_KEYS:
            self.assertEqual(fit[key], source["fit"][key])
        self.assertEqual(fit["rows"], source["fit"]["rows"])
        self.assertEqual(fit["fitted_on"], m8.FITTED_ON)
        self.assertIn("benchmark/generate.py", fit["fitted_on"])
        stage = self.dir / "stage"
        stage.mkdir()
        (stage / "score_bias.json").write_bytes(output.read_bytes())
        self.assertEqual(build.screen(stage)["text_files_screened"], 1)
        (stage / "score_bias.json").write_bytes(SCORED.read_bytes())
        with self.assertRaises(ValueError):
            build.screen(stage)

    def test_committed_public_record_is_the_published_file(self):
        output = self.dir / "public.json"
        publish(SCORED, output)
        self.assertEqual(output.read_bytes(), PUBLIC.read_bytes())

    def test_builder_binds_the_public_copy_to_the_scored_offsets(self):
        output = self.dir / "public.json"
        publish(SCORED, output)
        model = json.loads(SCORED.read_text())["model_sha256"]
        spec = {
            "score_bias": {"path": str(output), "sha256": layout.sha_file(output)},
            "expected_identity": {"model_sha256": model},
        }
        verified = build.verify_score_bias(spec, model)
        scored = json.loads(SCORED.read_text())
        # As training.model.infer records the applied file in its native manifest.
        offsets, _ = load_score_bias(SCORED, model)
        native = {
            "model_sha256": model,
            "score_bias_sha256": SCORED_SHA256,
            "score_bias": {
                "file_sha256": SCORED_SHA256,
                "offsets": {str(k): v for k, v in sorted(offsets.items())},
            },
        }
        self.assertEqual(native["score_bias"]["offsets"], scored["offsets"])
        self.assertEqual(
            build.check_scored_score_bias(spec, native, verified), SCORED_SHA256
        )

    def test_refuses_to_overwrite(self):
        output = self.dir / "public.json"
        output.write_text("{}")
        with self.assertRaises(FileExistsError):
            publish(SCORED, output)
        self.assertEqual(output.read_text(), "{}")

    def test_offsets_round_trip_exactly(self):
        source = json.loads(SCORED.read_text())
        source["offsets"] = {
            "5": [0.1 + 0.2, -1e-17, 1 / 3, 123456.789012345, -(0.1 + 0.2 + 1e-17)]
        }
        path = self.dir / "scored.json"
        path.write_text(json.dumps(source))
        output = self.dir / "public.json"
        publish(path, output)
        self.assertEqual(json.loads(output.read_text())["offsets"], source["offsets"])

    def test_human_inputs_keep_hashes_only(self):
        source = json.loads(SCORED.read_text())
        source["fit"].update(candidate="s5h-b05", human_weight=1.0)
        source["fit"]["rows"]["human"] = {
            "rows": 7,
            "rows_by_source": {"H6": 7},
            "rows_by_level": {"0": 7},
            "inputs": {
                "fit_aho_probs_sha256": "a" * 64,
                "cal_parity_max_abs_drift": 0.0,
                "aho_dir": "/data/dev2/runs/06b/m7/a/aho",
                "score5_panel_rows": {"path": "/data/dev2/x.jsonl", "sha256": "b" * 64},
            },
        }
        path = self.dir / "scored.json"
        path.write_text(json.dumps(source))
        output = self.dir / "public.json"
        publish(path, output)
        public = json.loads(output.read_text())
        self.assertFalse(private(output.read_text()))
        self.assertEqual(
            public["fit"]["inputs_sha256"]["human/fit_aho_probs"], "a" * 64
        )
        self.assertEqual(
            public["fit"]["inputs_sha256"]["human/score5_panel_rows"], "b" * 64
        )
        self.assertNotIn("inputs", public["fit"]["rows"]["human"])
        self.assertEqual(public["fit"]["rows"]["human"]["rows_by_source"], {"H6": 7})
        self.assertIn("FIT_AHO", public["fit"]["fitted_on"])

    def test_private_text_in_a_public_field_is_refused(self):
        source = json.loads(SCORED.read_text())
        bad = {
            "prereg": copy.deepcopy(source),
            "offsets": copy.deepcopy(source),
            "candidate": copy.deepcopy(source),
        }
        bad["prereg"]["fit"]["prereg"] = "/data/dev2/src/prereg.md"
        bad["offsets"]["offsets"] = {"5": [0.0] * 4}
        bad["candidate"]["fit"]["candidate"] = "other"
        for name, value in bad.items():
            path = self.dir / f"{name}.json"
            path.write_text(json.dumps(value))
            output = self.dir / f"{name}.public.json"
            with self.subTest(name), self.assertRaises(ValueError):
                publish(path, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
