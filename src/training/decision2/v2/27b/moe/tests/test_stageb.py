"""Stage B helpers: header-based parameter counts, M1's latency roster, the verdict tie-break (stdlib)."""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

params = importlib.import_module("v2.27b.moe.moe_params")
latency = importlib.import_module("v2.27b.moe.latency")
verdicts = importlib.import_module("v2.27b.moe.moe_verdicts")


def write_header(path: Path, tensors: dict[str, list[int]]) -> None:
    header = {
        name: {"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


class ParameterCounts(unittest.TestCase):
    def test_active_keeps_top_k_of_the_routed_experts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, ckpt = Path(tmp) / "base", Path(tmp) / "ckpt"
            write_header(
                base / "model-1.safetensors",
                {
                    "model.language_model.embed_tokens.weight": [10, 4],
                    "model.language_model.layers.0.self_attn.q_proj.weight": [4, 4],
                    "model.language_model.layers.0.experts.gate_up_proj": [8, 4, 6],
                    "model.language_model.layers.0.experts.down_proj": [8, 3, 4],
                    "model.vision_tower.patch.weight": [5, 5],
                },
            )
            (base / "config.json").write_text(
                json.dumps({"text_config": {"num_experts": 8, "top_k_experts": 2}})
            )
            write_header(
                ckpt / "adapter/adapter_model.safetensors", {"a": [2, 4], "b": [4, 2]}
            )
            write_header(ckpt / "decision_head.safetensors", {"h": [3]})
            (ckpt / "decision_config.json").write_text(
                json.dumps(
                    {
                        "checkpoint_format": "peft-lora/1",
                        "lora": {
                            "rank": 2,
                            "source_kind": "posttrained",
                            "source_fingerprint": {
                                "files_sha256": {
                                    "model-1.safetensors": "x",
                                    "config.json": "y",
                                }
                            },
                        },
                    }
                )
            )
            result = params.counts(ckpt, base)
        routed = 8 * 4 * 6 + 8 * 3 * 4
        loaded = 40 + 16 + routed + 16 + 3
        self.assertEqual(result["routed_expert_parameters"], routed)
        self.assertEqual(result["loaded_parameters"], loaded)
        self.assertEqual(result["active_parameters"], loaded - routed + routed * 2 // 8)


class Roster(unittest.TestCase):
    def test_m1_roster_is_the_smallest_hashes(self):
        rows = [{"id": f"r{i}"} for i in range(100)]
        picked = latency.roster(rows)
        key = lambda r: hashlib.sha256(
            ("27b-probe/" + r["id"]).encode()
        ).hexdigest()  # noqa: E731
        self.assertEqual(len(picked), 65)
        self.assertEqual(picked, sorted(rows, key=key)[:65])

    def test_summary_percentiles(self):
        out = latency.summary([float(v) for v in range(1, 65)], [10] * 64)
        self.assertEqual((out["n"], out["p50_ms"], out["p95_ms"]), (64, 33.0, 61.0))


class TieBreak(unittest.TestCase):
    def test_lower_bound_vs_autojev_then_a20r_then_active(self):
        def entry(lb_aj, lb_a20r, active):
            return {
                "paired": {
                    "autojev27": {"ci95": [lb_aj, 1.0]},
                    "A20r": {"ci95": [lb_a20r, 1.0]},
                },
                "package": {"active_parameters": active},
            }

        out = {
            "a": entry(0.1, 0.0, 5),
            "b": entry(0.2, -1.0, 9),
            "c": entry(0.1, 0.0, 4),
        }
        self.assertEqual(verdicts.rank(out, ["a", "b", "c"]), ["b", "c", "a"])


if __name__ == "__main__":
    unittest.main()
