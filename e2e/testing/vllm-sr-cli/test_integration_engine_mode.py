#!/usr/bin/env python3
"""Native inference remains available across CLI startup mode changes.

The tiny Decision fixture runs offline in the built Router image. The test
checks public auth/discovery and native inference after starting both Engine
and Router modes with the supported serve flags. It owns an isolated stack.
"""

import copy
import math
import os
import shutil
import socket
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

import yaml
from cli.runtime_stack import MAX_PORT_OFFSET
from runtime_http import call, page_requests, write_fixture

QUICKSTART = (
    Path(__file__).resolve().parents[3] / "website/docs/model-runtime/quickstart.md"
)
API_KEY = "engine-e2e-key"
MANAGEMENT_KEY = "engine-e2e-management-key"
PUBLIC_MODEL = "test/decision"


def available_offset():
    # Probe every port the minimal stack publishes before selecting an offset.
    for offset in range(10000, MAX_PORT_OFFSET + 1, 137):
        sockets = []
        try:
            for port in (6379, 8080, 8899, 9190):
                probe = socket.socket()
                sockets.append(probe)
                probe.bind(("127.0.0.1", port + offset))
            return offset
        except OSError:
            pass
        finally:
            for probe in sockets:
                probe.close()
    raise RuntimeError("No isolated test port range is free")


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Set RUN_INTEGRATION_TESTS=true to start an isolated real frontend stack.",
)
class TestEngineMode(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="vllm-sr-engine-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.stack = "engine-e2e-" + uuid.uuid4().hex[:10]
        self.offset = available_offset()
        self.env = {
            **os.environ,
            "VLLM_SR_STACK_NAME": self.stack,
            "VLLM_SR_PORT_OFFSET": str(self.offset),
            "HF_HUB_OFFLINE": "1",
            "ENGINE_E2E_MANAGEMENT_KEY": MANAGEMENT_KEY,
        }
        self.container = f"{self.stack}-vllm-sr-router-container"
        self.base = f"http://127.0.0.1:{8899 + self.offset}"
        self.config = self.root / "config.yaml"
        write_fixture(self.root / "models" / "decision", "decision2", "qwen3")
        self.config.write_text(
            yaml.safe_dump(
                {
                    "version": "v0.3",
                    "listeners": [
                        {
                            "name": "main",
                            "address": "0.0.0.0",
                            "port": 8899,
                            "api_keys": [API_KEY],
                            "models": ["vllm-sr/auto"],
                            "systemone": {"models": [PUBLIC_MODEL]},
                        }
                    ],
                    "providers": {
                        "models": [
                            {
                                "name": "answer",
                                "backend_refs": [
                                    {
                                        "name": "answer",
                                        "endpoint": "127.0.0.1:18000",
                                        "provider": "vllm",
                                    }
                                ],
                            }
                        ]
                    },
                    "routing": {
                        "decisions": [
                            {
                                "name": "saved",
                                "priority": 1,
                                "rules": {"operator": "AND", "conditions": []},
                                "modelRefs": [{"model": "answer"}],
                            }
                        ]
                    },
                    "global": {
                        "router": {"enabled": False},
                        "model_catalog": {
                            "system": {"decision_model": {"deployment": "primary"}},
                            "deployments": {
                                "primary": {
                                    "provider": "model_runtime",
                                    "artifact": "/app/models/decision",
                                    "public_name": PUBLIC_MODEL,
                                    "device": "cpu",
                                }
                            },
                        },
                        "services": {
                            "management_api": {
                                "bind_address": "0.0.0.0",
                                "port": 8080,
                                "auth": {
                                    "mode": "bearer",
                                    "tokens": [
                                        {
                                            "env": "ENGINE_E2E_MANAGEMENT_KEY",
                                            "role": "admin",
                                        }
                                    ],
                                },
                            },
                            "observability": {"tracing": {"enabled": False}},
                        },
                    },
                }
            )
        )
        self.addCleanup(self.stop)
        self.serve(engine=True)

    def serve(self, *, engine):
        self.cli(
            "serve",
            "--config",
            str(self.config),
            *(["--engine"] if engine else []),
            "--gateway",
            "standalone",
            "--minimal",
            "--image-pull-policy",
            "ifnotpresent",
            "--startup-timeout",
            "300",
        )

    def cli(self, *arguments):
        result = subprocess.run(
            ["vllm-sr", *arguments],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=360,
            check=False,
        )
        with (self.root / "cli.log").open("a") as log:
            log.write(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def stop(self):
        subprocess.run(
            ["vllm-sr", "stop"],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            timeout=90,
            check=False,
        )

    def native(self):
        body = copy.deepcopy(page_requests(QUICKSTART)["/v1/systemone"])
        body["model"] = PUBLIC_MODEL
        status, response = call(
            self.base,
            "/v1/systemone",
            body,
            headers={"Authorization": "Bearer " + API_KEY},
        )
        self.assertEqual(status, 200, response)
        kind, reasoning = response["answers"]["kind"], response["answers"]["reasoning"]
        self.assertIn(kind["choice"], body["questions"]["kind"]["criteria"])
        self.assertTrue(
            math.isclose(sum(kind["probabilities"].values()), 1.0, abs_tol=1e-6)
        )
        self.assertGreaterEqual(reasoning["noul"], 0.0)
        self.assertLessEqual(reasoning["noul"], 1.0)
        return body, response

    def test_native_inference_survives_startup_mode_changes(self):
        status, _ = call(self.base, "/v1/systemone/models")
        self.assertEqual(status, 401)
        status, models = call(
            self.base,
            "/v1/systemone/models",
            headers={"Authorization": "Bearer " + API_KEY},
        )
        self.assertEqual(status, 200, models)
        self.assertEqual([model["id"] for model in models["data"]], [PUBLIC_MODEL])
        body, response = self.native()
        status, alias = call(
            self.base,
            "/v1/decisions",
            body,
            headers={"Authorization": "Bearer " + API_KEY},
        )
        self.assertEqual(status, 200, alias)
        self.assertEqual(alias["answers"], response["answers"])
        for mode in ("router", "engine"):
            self.serve(engine=mode == "engine")
            management_base = f"http://127.0.0.1:{8080 + self.offset}"
            status, _ = call(management_base, "/api/v1/instance")
            self.assertEqual(status, 401)
            status, state = call(
                management_base,
                "/api/v1/instance",
                headers={"Authorization": "Bearer " + MANAGEMENT_KEY},
            )
            self.assertEqual(status, 200, state)
            self.assertEqual(state["observed_mode"], mode, state)
            status, _ = call(self.base, "/v1/systemone/models")
            self.assertEqual(status, 401)
            self.native()


if __name__ == "__main__":
    unittest.main()
