import importlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

verify = importlib.import_module("v2.27b.verify_candidates")
launch = importlib.import_module("v2.27b.launch")


class ComponentTest(unittest.TestCase):
    def test_qwen_and_gemma_names(self):
        cases = {
            "model.language_model.embed_tokens.weight": "text_embeddings",
            "model.language_model.layers.0.mlp.down_proj.weight": "text_other",
            "model.language_model.layers.3.experts.gate_up_proj": "text_moe_experts",
            "model.language_model.layers.3.router.proj.weight": "text_moe_router",
            "model.visual.blocks.0.attn.qkv.weight": "vision",
            "model.vision_tower.encoder.layers.0.weight": "vision",
            "model.embed_vision.embedding_projection.weight": "vision",
            "mtp.layers.0.mlp.gate_proj.weight": "mtp",
            "lm_head.weight": "lm_head",
            "something.else": "other",
        }
        for name, expected in cases.items():
            self.assertEqual(verify.component(name), expected, name)

    def test_counts_and_moe_active(self):
        headers = {
            "a.safetensors": {
                "model.language_model.embed_tokens.weight": {
                    "dtype": "BF16",
                    "shape": [10, 4],
                },
                "model.language_model.layers.0.experts.down_proj": {
                    "dtype": "BF16",
                    "shape": [8, 4, 2],
                },
                "model.language_model.layers.0.router.proj.weight": {
                    "dtype": "BF16",
                    "shape": [8, 4],
                },
                "model.language_model.layers.0.self_attn.q_proj.weight": {
                    "dtype": "F32",
                    "shape": [4, 4],
                },
                "model.vision_tower.w": {"dtype": "BF16", "shape": [3]},
            }
        }
        counts = verify.count_parameters(headers)
        self.assertEqual(counts["stored_parameter_count"], 40 + 64 + 32 + 16 + 3)
        self.assertEqual(counts["text_decoder_parameters"], 40 + 64 + 32 + 16)
        self.assertEqual(counts["stored_tensor_bytes"], (40 + 64 + 32 + 3) * 2 + 16 * 4)
        active = verify.moe_active(
            {"text_config": {"num_experts": 8, "top_k_experts": 2}}, counts
        )
        self.assertEqual(
            active["active_text_parameters_per_token"], 40 + 32 + 16 + 64 * 2 // 8
        )
        self.assertIsNone(verify.moe_active({"text_config": {}}, counts))

    def test_license_facts(self):
        readme = (
            "---\nlicense: apache-2.0\nlibrary_name: transformers\n---\nBody Apache 2.0"
        )
        facts = verify.license_facts(
            readme,
            "  Apache License\n Version 2.0, January 2004",
            ["license:apache-2.0"],
        )
        self.assertEqual(facts["card_front_matter"]["license"], "apache-2.0")
        self.assertTrue(facts["license_file_is_apache_2"])
        self.assertFalse(facts["card_mentions_gemma_terms_of_use"])


class LaunchTest(unittest.TestCase):
    def test_rejects_foreign_gpu(self):
        for gpu in (0, 4, 7):
            with self.assertRaises(ValueError):
                launch.render_node(gpu)

    def test_render_node_pci_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pci = root / "devices" / "0000:ab:00.0"
            pci.mkdir(parents=True)
            node = root / "drm" / "renderD169"
            node.mkdir(parents=True)
            (node / "device").symlink_to(pci)
            self.assertEqual(launch.render_node(5, root / "drm").name, "renderD169")
            wrong = root / "drm" / "renderD177"
            wrong.mkdir()
            (wrong / "device").symlink_to(pci)
            with self.assertRaises(ValueError):
                launch.render_node(6, root / "drm")

    def test_create_argv_rejects_secret_env_and_bad_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = dict(
                name="x",
                workdir="/code",
                command=["python3", "-V"],
                mount=[f"{tmp}:/code"],
            )
            argv = launch.create_argv(
                Namespace(env=["A=1"], **base), Path("/dev/dri/renderD169")
            )
            self.assertIn(
                "type=bind,src=%s,dst=/code,readonly" % Path(tmp).resolve(), argv
            )
            self.assertEqual(argv[-2:], ["python3", "-V"])
            self.assertIn("ROCR_VISIBLE_DEVICES=0", argv)
            with self.assertRaises(ValueError):
                launch.create_argv(
                    Namespace(env=["HF_TOKEN=abc"], **base), Path("/dev/dri/renderD169")
                )
            bad = dict(base, mount=[f"{tmp}:/code:xx"])
            with self.assertRaises(ValueError):
                launch.create_argv(
                    Namespace(env=[], **bad), Path("/dev/dri/renderD169")
                )

    def test_lease_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                launch.write_lease(5, {"status": "running"}, root)
            (root / "gpu5.lock").mkdir()
            (root / "gpu5.lock" / "owner").write_text(json.dumps({"track": "eval"}))
            with self.assertRaises(ValueError):
                launch.write_lease(5, {"status": "running"}, root)
            (root / "gpu5.lock" / "owner").write_text(
                json.dumps({"track": "27b", "status": "reserved-idle"})
            )
            launch.write_lease(5, {"status": "running", "container": "a"}, root)
            with self.assertRaises(ValueError):
                launch.write_lease(5, {"status": "running", "container": "b"}, root)
            launch.write_lease(5, {"status": "reserved-idle", "container": None}, root)
            self.assertEqual(
                json.loads((root / "gpu5.lock" / "owner").read_text())["status"],
                "reserved-idle",
            )


if __name__ == "__main__":
    unittest.main()


summarize = importlib.import_module("v2.27b.summarize")


def _reports(correct, css_f1, invalid=0):
    dev = {
        "macro_family_accuracy": sum(correct.values()) / 1600,
        "overall": {
            "correct_n": sum(correct.values()),
            "invalid_or_missing_n": invalid,
            "brier": 0.2,
        },
        "by_type": {
            k: {"correct_n": v, "n": 800 if k == "choice" else 400}
            for k, v in correct.items()
        },
        "by_family": {"a": {"accuracy_all": 0.5}},
    }
    tasks = {
        f"t{i}": {"role": "pilot", "macro_f1_all": f} for i, f in enumerate(css_f1)
    }
    css = {
        "roles": {
            "pilot": {
                "median_task_macro_f1_all": sorted(css_f1)[1],
                "items": 10,
                "valid_items": 10,
            }
        },
        "tasks": tasks,
    }
    return dev, css


class SummarizeTest(unittest.TestCase):
    def test_proxy_matches_best368_record(self):
        dev, css = _reports(
            {"choice": 791, "noul": 261, "score": 161}, [0.47488, 0.6177652, 0.68757]
        )
        summary = summarize.summarize(dev, css)
        self.assertAlmostEqual(summary["P_dev"], 68.4357, places=3)

    def test_decision_rule(self):
        control = summarize.summarize(
            *_reports({"choice": 791, "noul": 261, "score": 161}, [0.5, 0.6, 0.7])
        )
        better = summarize.summarize(
            *_reports({"choice": 790, "noul": 290, "score": 200}, [0.5, 0.63, 0.7])
        )
        drop = summarize.summarize(
            *_reports({"choice": 700, "noul": 330, "score": 260}, [0.5, 0.63, 0.7])
        )
        self.assertTrue(summarize.decide(control, better, None)["displaces_control"])
        self.assertFalse(
            summarize.decide(control, better, 5.0)["checks"]["p_dev_margin"]
        )
        self.assertFalse(
            summarize.decide(control, drop, None)["checks"]["no_type_drop_over_3"]
        )


class SharedLeaseTest(unittest.TestCase):
    def test_shared_job_does_not_replace_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gpu6.lock").mkdir()
            owner = root / "gpu6.lock" / "owner"
            owner.write_text(json.dumps({"track": "27b", "status": "reserved-idle"}))
            launch.write_lease(6, {"status": "running", "container": "probe"}, root)
            launch.write_lease(
                6, {"status": "running", "container": "heads"}, root, shared="heads"
            )
            state = json.loads(owner.read_text())
            self.assertEqual(
                (state["container"], state["shared_containers"]), ("probe", ["heads"])
            )
            launch.write_lease(
                6, {"status": "reserved-idle", "container": None}, root, shared="heads"
            )
            state = json.loads(owner.read_text())
            self.assertEqual(
                (state["status"], state["container"], state["shared_containers"]),
                ("running", "probe", []),
            )
