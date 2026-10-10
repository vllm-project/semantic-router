from __future__ import annotations

import subprocess
import unittest

from d25.omni.k8s.train import job, memory_plan

NODE = "example-node-04"
COMMON = ["--node", NODE, "--tag", "abc123"]
TRAIN = [
    "train",
    "--run",
    "graft-r50",
    *COMMON,
    "--init",
    "/data/d25/omni/ckpt/init-graft",
    "--arm",
    "O-graft-frozen",
    "--rows",
    "/data/d25/omni/data/mm/v1/rows-*.jsonl.gz",
    "--replay-rows",
    "/data/d25/shared/data/text/m1/rows-0.jsonl.gz",
]
LABELS = {
    "app.kubernetes.io/part-of": "decision-2.5",
    "d25/campaign": "omni",
    "d25/ws": "train",
}


def audit(manifest: dict) -> dict:
    spec = manifest["spec"]
    template = spec["template"]
    pod = template["spec"]
    for key, value in LABELS.items():
        assert manifest["metadata"]["labels"][key] == value
        assert template["metadata"]["labels"][key] == value
    assert manifest["metadata"]["namespace"] == "semantic-router"
    assert manifest["metadata"]["name"].startswith("d25-omni-train-")
    assert spec["backoffLimit"] == 0 and spec["ttlSecondsAfterFinished"] == 86400
    assert pod["restartPolicy"] == "Never"
    assert "nodeName" not in pod and "priorityClassName" not in pod
    assert pod["nodeSelector"] == {"kubernetes.io/hostname": NODE}
    (container,) = pod["containers"]
    assert container["resources"]["requests"] == container["resources"]["limits"]
    assert container["securityContext"]["privileged"] is False
    for env in container["env"]:
        assert env["name"] not in (
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        )
        assert env["name"] != "HF_TOKEN" or "value" not in env
    mounts = {m["name"]: m for m in container["volumeMounts"]}
    for volume in pod["volumes"]:
        if "hostPath" in volume:
            path = volume["hostPath"]["path"]
            assert not path.startswith("/dev")
            if path.startswith("/data/d25/omni/"):
                assert volume["hostPath"]["type"] == "DirectoryOrCreate"
            else:
                assert path.startswith(("/data/d25/shared/", "/data/d25/vega/ckpt"))
                assert mounts[volume["name"]].get("readOnly") is True
        else:
            assert volume["emptyDir"]["medium"] == "Memory"
    return container


class JobTest(unittest.TestCase):
    def test_train_job(self) -> None:
        manifest = job.build(TRAIN)
        container = audit(manifest)
        self.assertEqual(container["resources"]["limits"]["amd.com/gpu"], "8")
        script = container["command"][2]
        self.assertIn(
            "torchrun --standalone --nproc_per_node 8 -m d25.omni.train.train", script
        )
        self.assertIn("--rows /data/d25/omni/data/mm/v1/rows-*.jsonl.gz", script)
        self.assertIn("--arm O-graft-frozen", script)
        self.assertIn("--code-tag abc123", script)
        self.assertIn("/data/d25/omni/ckpt/graft-r50", script)
        writable = [
            m["mountPath"] for m in container["volumeMounts"] if not m.get("readOnly")
        ]
        self.assertEqual(sorted(writable), ["/data/d25/omni/ckpt", "/dev/shm"])
        subprocess.run(["bash", "-n", "-c", script], check=True)

    def test_assemble_job_is_cpu_only(self) -> None:
        manifest = job.build(
            [
                "assemble",
                "--run",
                "init-graft",
                *COMMON,
                "--vega",
                "/data/d25/vega/ckpt/run-a/step-01000",
            ]
        )
        container = audit(manifest)
        self.assertEqual(
            container["resources"]["limits"], {"cpu": "8", "memory": "32Gi"}
        )
        self.assertIn(
            "--vega /data/d25/vega/ckpt/run-a/step-01000", container["command"][2]
        )
        self.assertIn(
            "--stock /data/d25/shared/hf-home/hub/models--Qwen--Qwen3.8-27B/snapshots/",
            container["command"][2],
        )
        subprocess.run(["bash", "-n", "-c", container["command"][2]], check=True)

    def test_eval_job(self) -> None:
        manifest = job.build(
            [
                "eval",
                "--run",
                "graft-r50",
                *COMMON,
                "--gpus",
                "4",
                "--ckpt",
                "/data/d25/omni/ckpt/graft-r50/checkpoints/step-00500",
                "--suite",
                "/data/d25/omni/suite/vision-0.3.1",
            ]
        )
        container = audit(manifest)
        script = container["command"][2]
        self.assertEqual(container["resources"]["limits"]["amd.com/gpu"], "4")
        self.assertEqual(script.count("run_suite run"), 4)
        self.assertIn("--device cuda:3", script)
        self.assertIn("/data/d25/omni/evals/graft-r50/step-00500/public", script)
        subprocess.run(["bash", "-n", "-c", script], check=True)

    def test_refusals(self) -> None:
        bad = [
            [*TRAIN[:3], "--node", "example-node-02", *TRAIN[5:]],
            [*TRAIN, "--out", "/data/d25/vega/ckpt/x"],
            [*TRAIN[:-1], "/data/elsewhere/rows.jsonl"],
            [*TRAIN[:-1], "/data/d25/shared/data/x;rm -rf /"],
            [*TRAIN, "--token-budget", "262144"],
            ["train", "--run", "Bad_Name", *TRAIN[3:]],
            ["assemble", "--run", "x", *COMMON, "--vega", "/data/other/ckpt"],
        ]
        for argv in bad:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                job.build(argv)

    def test_memory_plan(self) -> None:
        default = memory_plan.plan()
        self.assertLess(default.peak, 0.5 * memory_plan.HBM_GB)
        self.assertGreater(default.peak, 100)
        frozen = memory_plan.plan(encoder_trainable=False)
        self.assertLess(frozen.peak, default.peak)
        self.assertEqual(memory_plan.max_token_budget(), 65_536)
        self.assertGreater(memory_plan.math_attention_penalty_gb(16_384), 30)


if __name__ == "__main__":
    unittest.main()
