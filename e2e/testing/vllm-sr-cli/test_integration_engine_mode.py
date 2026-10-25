#!/usr/bin/env python3
"""Native inference remains available across CLI startup mode changes.

The tiny Decision fixtures run offline in the built Router image. The tests
check public auth/discovery and native inference after starting both Engine
and Router modes with the supported serve flags, that images and videos
reach a Decision 3.0 model through the public System One API, that the API
reads the System One SDKs' forms (compressed and multipart bodies, assets,
long context) and answers a failed question for the whole request on
/v1/systemone, that the Router refuses options that API does not define, and
that Engine mode starts only the Router container by default. Each test owns an isolated stack.
"""

import base64
import copy
import gzip
import json
import math
import os
import random
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import unittest
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import yaml
from cli.runtime_stack import MAX_PORT_OFFSET
from runtime_http import HTTP_OK, call, page_requests, write_fixture

QUICKSTART = (
    Path(__file__).resolve().parents[3] / "website/docs/model-runtime/quickstart.md"
)
API_KEY = "engine-e2e-key"
MANAGEMENT_KEY = "engine-e2e-management-key"
PUBLIC_MODEL = "test/decision"
# The public System One API accepted at most 2 MiB before images.
FORMER_REQUEST_LIMIT = 2 << 20
# The most bytes of a request body the public System One API reads.
REQUEST_LIMIT = 48 << 20
# A 64 x 48 MPEG-4 clip (8 frames, 4 per second) of a blue square moving over a green ramp.
TINY_MP4 = (
    "data:video/mp4;base64,"
    "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"
)


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


def png_data_url(width, height, seed):
    """An RGB PNG of seeded noise, which does not compress, as a data URL."""
    pixels = random.Random(seed).randbytes(width * height * 3)
    rows = b"".join(
        b"\x00" + pixels[row * width * 3 : (row + 1) * width * 3]
        for row in range(height)
    )

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def engine_config():
    return {
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


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS", "").lower() == "true",
    "Set RUN_INTEGRATION_TESTS=true to start an isolated real frontend stack.",
)
class EngineStack(unittest.TestCase):
    """An Engine mode stack publishing one fixture package as PUBLIC_MODEL."""

    fixture = ("decision2", "qwen3")

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
        write_fixture(self.root / "models" / "decision", *self.fixture)
        self.config.write_text(yaml.safe_dump(engine_config()))
        self.addCleanup(self.stop)
        self.serve(engine=True)

    def serve(self, *, engine):
        # Engine mode is minimal by default; Router mode needs the flag.
        self.cli(
            "serve",
            "--config",
            str(self.config),
            "--engine" if engine else "--minimal",
            "--gateway",
            "standalone",
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

    def public(self, path, body=None):
        return call(
            self.base, path, body, headers={"Authorization": "Bearer " + API_KEY}
        )


class TestEngineMode(EngineStack):
    def native(self):
        body = copy.deepcopy(page_requests(QUICKSTART)["/v1/systemone"])
        body["model"] = PUBLIC_MODEL
        status, response = self.public("/v1/systemone", body)
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
        status, models = self.public("/v1/systemone/models")
        self.assertEqual(status, 200, models)
        self.assertEqual([model["id"] for model in models["data"]], [PUBLIC_MODEL])
        body, response = self.native()
        status, alias = self.public("/v1/decisions", body)
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


class TestEngineModeContainers(EngineStack):
    def containers(self):
        runtime = shutil.which(os.environ.get("CONTAINER_RUNTIME") or "docker") or (
            shutil.which("podman")
        )
        listed = subprocess.run(
            [runtime, "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return {name for name in listed.stdout.split() if name.startswith(self.stack)}

    def test_engine_mode_starts_only_the_router_and_stop_removes_it(self):
        # No Dashboard, sr-bench, Redis, Jaeger, Prometheus or Grafana.
        self.assertEqual(self.containers(), {self.container})
        status = self.cli("status")
        self.assertIn("Router", status)
        self.assertNotIn("Dashboard", status)
        status, models = self.public("/v1/systemone/models")
        self.assertEqual(status, 200, models)
        self.stop()
        self.assertEqual(self.containers(), set())


class TestEngineModeRestartPolicy(EngineStack):
    """--restart-policy brings the stack back after its process exits."""

    def serve(self, *, engine):
        self.cli(
            "serve",
            "--config",
            str(self.config),
            "--engine" if engine else "--minimal",
            "--gateway",
            "standalone",
            "--image-pull-policy",
            "ifnotpresent",
            "--startup-timeout",
            "300",
            "--restart-policy",
            "unless-stopped",
        )

    def runtime(self, *arguments):
        binary = shutil.which(os.environ.get("CONTAINER_RUNTIME") or "docker") or (
            shutil.which("podman")
        )
        return subprocess.run(
            [binary, *arguments],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def inspect(self, template):
        result = self.runtime("inspect", "-f", template, self.container)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_the_router_comes_back_after_its_process_exits(self):
        self.assertEqual(
            self.inspect("{{.HostConfig.RestartPolicy.Name}}"), "unless-stopped"
        )
        status, _ = self.public("/v1/systemone/models")
        self.assertEqual(status, 200)
        restarts = int(self.inspect("{{.RestartCount}}"))
        # The Router exits as on a crash; the runtime, not the CLI, restarts it.
        self.runtime("exec", self.container, "sh", "-c", "kill -TERM 1; kill -TERM -1")
        deadline = time.monotonic() + 240
        status = None
        while time.monotonic() < deadline:
            if int(self.inspect("{{.RestartCount}}")) > restarts:
                try:
                    status, _ = self.public("/v1/systemone/models")
                except OSError:  # the listener is not back yet
                    status = None
                if status == HTTP_OK:
                    break
            time.sleep(2)
        self.assertGreater(int(self.inspect("{{.RestartCount}}")), restarts)
        self.assertEqual(status, 200)
        self.stop()
        self.assertNotEqual(
            self.runtime("inspect", self.container).returncode,
            0,
            "vllm-sr stop must still remove a container with a restart policy",
        )


class TestEngineModeEngineArgs(EngineStack):
    """Engine flags set the default deployment's engine arguments in its worker."""

    def serve(self, *, engine):
        self.cli(
            "serve",
            "--config",
            str(self.config),
            "--engine" if engine else "--minimal",
            "--gateway",
            "standalone",
            "--image-pull-policy",
            "ifnotpresent",
            "--startup-timeout",
            "300",
            "--max-num-seqs",
            "64",
            "--batch-window-ms",
            "0.5",
        )

    def test_engine_flags_reach_the_served_model(self):
        status, inventory = call(
            f"http://127.0.0.1:{8080 + self.offset}",
            "/api/v1/inventory/model-runtime",
            headers={"Authorization": "Bearer " + MANAGEMENT_KEY},
        )
        self.assertEqual(status, HTTP_OK, inventory)
        (primary,) = [
            deployment
            for deployment in inventory["deployments"]
            if deployment["name"] == "primary"
        ]
        # The flags reach the worker; what they leave out keeps its default.
        self.assertEqual(
            primary["engine_args"],
            {
                "max_num_seqs": 64,
                "max_num_batched_tokens": 65536,
                "batch_window_ms": 0.5,
                "dtype": "auto",
                "enable_shared_context": True,
            },
        )
        self.assertEqual(primary["numerics"], "approximate")
        body = copy.deepcopy(page_requests(QUICKSTART)["/v1/systemone"])
        body["model"] = PUBLIC_MODEL
        status, response = self.public("/v1/systemone", body)
        self.assertEqual(status, HTTP_OK, response)
        body["options"] = {"return_meta": True}
        status, response = self.public("/v1/systemone", body)
        self.assertEqual(status, HTTP_OK, response)
        self.assertEqual(response["meta"]["numerics"], "approximate", response)
        self.assertNotIn("profile", response["meta"], response)
        # The Router refuses options the public System One API does not define.
        for options, message in (
            ({"profile": "exact"}, "options.profile was removed: "),
            ({"verbose": True}, "options.verbose is not a System One option; "),
        ):
            body["options"] = options
            status, failure = self.public("/v1/systemone", body)
            self.assertEqual(status, 400, failure)
            self.assertEqual(failure["error"]["code"], "invalid_request", failure)
            self.assertTrue(failure["error"]["message"].startswith(message), failure)


class TestEngineModeDeadlineLimit(EngineStack):
    """The listener's systemone.max_deadline_ms bounds a request's deadline_ms."""

    def serve(self, *, engine):
        config = engine_config()
        config["listeners"][0]["systemone"]["max_deadline_ms"] = 1
        self.config.write_text(yaml.safe_dump(config))
        super().serve(engine=engine)

    def test_the_server_limit_bounds_a_longer_request_deadline(self):
        status, models = self.public("/v1/systemone/models")
        self.assertEqual(status, HTTP_OK, models)
        body = copy.deepcopy(page_requests(QUICKSTART)["/v1/systemone"])
        body["model"] = PUBLIC_MODEL
        body["options"] = {"deadline_ms": 600000}
        status, failure = self.public("/v1/systemone", body)
        self.assertEqual(status, 504, failure)
        self.assertEqual(
            failure["error"]["code"], "systemone_deadline_exceeded", failure
        )

    def test_the_server_limit_bounds_a_long_context_request(self):
        body = copy.deepcopy(page_requests(QUICKSTART)["/v1/systemone"])
        body["model"] = PUBLIC_MODEL
        body["options"] = {"long_context": True}
        status, failure = self.public("/v1/systemone", body)
        self.assertEqual(status, 504, failure)
        self.assertEqual(
            failure["error"]["code"], "systemone_deadline_exceeded", failure
        )
        self.assertIn("systemone.max_deadline_ms", failure["error"]["message"], failure)


class TestEngineModeImages(EngineStack):
    fixture = ("decision3", "vision")

    def ask(self, path, images=None, videos=None):
        body = copy.deepcopy(page_requests(QUICKSTART)["/v1/systemone"])
        body["model"] = PUBLIC_MODEL
        if images is not None:
            body["images"] = images
        if videos is not None:
            body["videos"] = videos
        return self.public(path, body)

    def test_unset_limits_resolve_for_the_device_and_lanes_merge_requests(self):
        status, inventory = call(
            f"http://127.0.0.1:{8080 + self.offset}",
            "/api/v1/inventory/model-runtime",
            headers={"Authorization": "Bearer " + MANAGEMENT_KEY},
        )
        self.assertEqual(status, HTTP_OK, inventory)
        (primary,) = [
            deployment
            for deployment in inventory["deployments"]
            if deployment["name"] == "primary"
        ]
        # Batch limits the deployment leaves unset come back as the ones the
        # worker chose for its device: the CPU's.
        engine_args = primary["engine_args"]
        self.assertEqual(
            (engine_args["max_num_seqs"], engine_args["max_num_batched_tokens"]),
            (8192, 65536),
        )
        # Image and text requests sent together merge in lanes and all answer.
        _, alone = self.ask("/v1/systemone")
        with ThreadPoolExecutor(8) as pool:
            futures = [
                pool.submit(self.ask, "/v1/systemone", [png_data_url(64, 48, seed=s)])
                for s in (11, 12, 13)
            ]
            futures += [pool.submit(self.ask, "/v1/systemone") for _ in range(6)]
            replies = [future.result() for future in futures]
        for status, reply in replies:
            self.assertEqual(status, 200, reply)
            self.assertEqual(set(reply["answers"]), set(alone["answers"]))

    def test_images_and_videos_reach_the_model_through_the_public_api(self):
        status, text = self.ask("/v1/systemone")
        self.assertEqual(status, 200, text)
        small = png_data_url(64, 48, seed=1)
        status, seen = self.ask("/v1/systemone", [small])
        self.assertEqual(status, 200, seen)
        self.assertNotEqual(seen["answers"], text["answers"])
        self.assertGreater(seen["usage"]["input_tokens"], text["usage"]["input_tokens"])
        status, alias = self.ask("/v1/decisions", [small])
        self.assertEqual(status, 200, alias)
        self.assertEqual(alias["answers"], seen["answers"])
        large = png_data_url(1024, 1024, seed=2)
        self.assertGreater(len(large), FORMER_REQUEST_LIMIT)
        status, both = self.ask("/v1/systemone", [small, large])
        self.assertEqual(status, 200, both)
        self.assertNotEqual(both["answers"], seen["answers"])
        status, failure = self.ask("/v1/systemone", ["data:image/png;base64,AAAA"])
        self.assertEqual(status, 400, failure)
        self.assertEqual(failure["error"]["code"], "invalid_request", failure)
        self.assertEqual(failure["error"]["message"], "malformed image", failure)
        status, clip = self.ask("/v1/systemone", videos=[TINY_MP4])
        self.assertEqual(status, 200, clip)
        self.assertNotEqual(clip["answers"], text["answers"])
        self.assertGreater(clip["usage"]["input_tokens"], text["usage"]["input_tokens"])
        status, failure = self.ask(
            "/v1/systemone", videos=["data:video/mp4;base64,AAAA"]
        )
        self.assertEqual(status, 400, failure)
        self.assertEqual(failure["error"]["code"], "invalid_request", failure)
        self.assertEqual(failure["error"]["message"], "malformed video", failure)
        body = copy.deepcopy(page_requests(QUICKSTART)["/v1/systemone"])
        body["model"] = PUBLIC_MODEL
        body["state"] = "x" * (REQUEST_LIMIT + 1024)
        status, failure = self.public("/v1/systemone", body)
        self.assertEqual(status, 413, failure)
        self.assertEqual(failure["error"]["code"], "request_too_large", failure)


class TestEngineModeStateOnce(EngineStack):
    """A Decision 3.0 package whose attention_mode is prefix_bidir reads each state once. In FP32 a question's
    answer does not depend on the other questions: a lone question runs as a row, several branch from the state,
    and the two agree exactly; the published BF16 differs between them by rounding."""

    fixture = ("decision3", "prefix_bidir")

    def serve(self, *, engine):
        self.cli(
            "serve",
            "--config",
            str(self.config),
            "--engine" if engine else "--minimal",
            "--gateway",
            "standalone",
            "--image-pull-policy",
            "ifnotpresent",
            "--startup-timeout",
            "300",
            "--dtype",
            "float32",
        )

    def ask(self, count, images=None):
        body = copy.deepcopy(page_requests(QUICKSTART)["/v1/systemone"])
        body["model"] = PUBLIC_MODEL
        first = next(iter(body["questions"].values()))
        body["questions"] = {f"q{index}": first for index in range(count)}
        if images is not None:
            body["images"] = images
        status, response = self.public("/v1/systemone", body)
        self.assertEqual(status, HTTP_OK, response)
        self.assertEqual(len(response["answers"]), count)
        self.assertTrue(all("error" not in a for a in response["answers"].values()))
        return response

    def test_questions_branch_from_one_reading_of_the_state(self):
        status, inventory = call(
            f"http://127.0.0.1:{8080 + self.offset}",
            "/api/v1/inventory/model-runtime",
            headers={"Authorization": "Bearer " + MANAGEMENT_KEY},
        )
        self.assertEqual(status, HTTP_OK, inventory)
        (primary,) = [d for d in inventory["deployments"] if d["name"] == "primary"]
        self.assertTrue(primary["engine_args"]["enable_shared_context"], primary)
        one, two, four = (self.ask(count) for count in (1, 2, 4))
        tokens = [r["usage"]["input_tokens"] for r in (one, two, four)]
        # The state counts once: each further question adds only its own block.
        block = tokens[1] - tokens[0]
        self.assertGreater(block, 0)
        self.assertLess(block, tokens[0])
        self.assertEqual(tokens[2], tokens[0] + 3 * block)
        # The same question over the same state answers alike, however many are asked.
        alone = one["answers"]["q0"]
        for response in (two, four):
            for answer in response["answers"].values():
                for key, value in answer["probabilities"].items():
                    self.assertAlmostEqual(value, alone["probabilities"][key], places=6)
        image = [png_data_url(64, 48, seed=3)]
        seen_one, seen_two = self.ask(1, image), self.ask(2, image)
        # The image is part of the state: its tokens count once per request.
        self.assertEqual(
            seen_two["usage"]["input_tokens"] - seen_one["usage"]["input_tokens"], block
        )
        self.assertGreater(seen_one["usage"]["input_tokens"], tokens[0])
        self.assertNotEqual(seen_one["answers"]["q0"], alone)


def raw_post(base, path, data, headers):
    """POST raw bytes with headers; the status and the decoded answer."""
    request = urllib_request.Request(
        base + path, data=data, method="POST", headers=headers
    )
    try:
        with urllib_request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib_error.HTTPError as failure:
        return failure.code, json.loads(failure.read())


def multipart(request, files):
    """A System One multipart body: the request part, then one part per image."""
    boundary = uuid.uuid4().hex
    chunks = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\n'
        "Content-Type: application/json\r\n\r\n".encode(),
        json.dumps(request).encode(),
        b"\r\n",
    ]
    for name, data in files.items():
        chunks += [
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{name}"\r\nContent-Type: image/png\r\n\r\n'.encode(),
            data,
            b"\r\n",
        ]
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


class TestEngineModeSystemOneSDKForms(EngineStack):
    """The public System One API reads requests as the official System One SDKs send them."""

    fixture = ("decision3", "validated")

    def headers(self, **extra):
        return {"Authorization": "Bearer " + API_KEY, **extra}

    def test_compressed_and_multipart_bodies_reach_the_model_as_assets(self):
        image = base64.b64decode(png_data_url(64, 48, seed=5).split(",", 1)[1])
        questions = {"red": {"type": "noul", "instructions": "Is the photo red?"}}
        request = {
            "model": PUBLIC_MODEL,
            "state": {"photo": "<image:1>", "note": "A product photo."},
            "questions": questions,
        }
        status, as_json = self.public(
            "/v1/systemone",
            {
                **request,
                "assets": {
                    "<image:1>": {
                        "type": "image",
                        "data_base64": base64.b64encode(image).decode(),
                    }
                },
            },
        )
        self.assertEqual(status, 200, as_json)
        body, content_type = multipart(
            {**request, "assets": {"<image:1>": {"type": "image", "part": "asset-0"}}},
            {"asset-0": image},
        )
        status, as_parts = raw_post(
            self.base,
            "/v1/systemone",
            gzip.compress(body),
            self.headers(**{"Content-Type": content_type, "Content-Encoding": "gzip"}),
        )
        self.assertEqual(status, 200, as_parts)
        self.assertEqual(as_parts["answers"], as_json["answers"])
        status, hoisted = self.public(
            "/v1/systemone",
            {
                **request,
                "state": {"photo": "[image 1]", "note": "A product photo."},
                "images": ["data:image/png;base64," + base64.b64encode(image).decode()],
            },
        )
        self.assertEqual((status, hoisted["answers"]), (200, as_json["answers"]))
        status, refused = raw_post(
            self.base,
            "/v1/systemone",
            json.dumps(request).encode(),
            self.headers(
                **{"Content-Type": "application/json", "Content-Encoding": "br"}
            ),
        )
        self.assertEqual(status, 415, refused)
        self.assertEqual(refused["error"]["reason"], "encoding_unsupported", refused)
        status, unused = self.public(
            "/v1/systemone",
            {
                **request,
                "assets": {
                    "<image:1>": {
                        "type": "image",
                        "data_base64": base64.b64encode(image).decode(),
                    },
                    "<image:2>": {
                        "type": "image",
                        "data_base64": base64.b64encode(image).decode(),
                    },
                },
            },
        )
        self.assertEqual(status, 400, unused)
        self.assertEqual(unused["error"]["reason"], "asset_unreferenced", unused)

    def test_a_failed_question_fails_a_system_one_request_as_a_whole(self):
        image = png_data_url(64, 48, seed=6).split(",", 1)[1]
        body = {
            "model": PUBLIC_MODEL,
            "state": "<photo>",
            "assets": {"<photo>": {"type": "image", "data_base64": image}},
            "questions": {
                "red": {"type": "noul", "instructions": "Is it red?"},
                "mug": {"type": "box", "instructions": "The mug."},
                "same": {
                    "type": "noul",
                    "instructions": "Is <photo> the same as the state?",
                },
            },
        }
        status, decisions = self.public("/v1/decisions", body)
        self.assertEqual(status, 200, decisions)
        self.assertEqual(
            decisions["answers"]["mug"]["reason"], "box_unsupported", decisions
        )
        self.assertEqual(
            decisions["answers"]["same"]["reason"],
            "question_images_unsupported",
            decisions,
        )
        self.assertIn("noul", decisions["answers"]["red"], decisions)
        status, systemone = self.public("/v1/systemone", body)
        self.assertEqual(status, 422, systemone)
        self.assertEqual(
            {
                key: systemone["error"][key]
                for key in ("code", "question_error", "reason", "question")
            },
            {
                "code": "question_failed",
                "question_error": "invalid_question",
                "reason": "box_unsupported",
                "question": "mug",
            },
            systemone,
        )

    def test_questions_past_the_validated_length_need_long_context(self):
        status, listing = call(self.base, "/v1/models", headers=self.headers())
        self.assertEqual(status, 200, listing)
        self.assertIn(
            PUBLIC_MODEL, [model["name"] for model in listing["models"]], listing
        )
        body = {
            "model": PUBLIC_MODEL,
            "state": "word " * 600,
            "questions": {"long": {"type": "noul", "instructions": "Is this long?"}},
        }
        status, refused = self.public("/v1/systemone", body)
        self.assertEqual(status, 400, refused)
        self.assertEqual(refused["error"]["reason"], "question_tokens", refused)
        status, answered = self.public(
            "/v1/systemone", {**body, "options": {"long_context": True}}
        )
        self.assertEqual(status, 200, answered)
        status, unknown = self.public(
            "/v1/systemone", {**body, "model": "unserved-model"}
        )
        self.assertEqual(status, 403, unknown)
        self.assertIn(PUBLIC_MODEL, unknown["error"]["message"], unknown)


class TestEngineModeD3Turbo(EngineStack):
    """A package with d3-turbo's card (contract 3.2.0) through the public API: state once, Box and track, images in
    questions and options, the validated length, the per-request cap and the video forms' reasons.
    """

    fixture = ("decision3", "turbo")
    # The turbo fixture's validated_input_tokens and max_request_tokens.
    TURBO_VALIDATED_TOKENS, TURBO_MAX_TOKENS = 4096, 32768

    def headers(self, **extra):
        return {"Authorization": "Bearer " + API_KEY, **extra}

    def listed(self):
        status, listing = call(self.base, "/v1/models", headers=self.headers())
        self.assertEqual(status, 200, listing)
        self.assertIn(PUBLIC_MODEL, [m["name"] for m in listing["models"]], listing)

    def test_the_answers_follow_d3_turbos_contract(self):
        self.listed()
        photo = base64.b64decode(png_data_url(96, 64, seed=7).split(",", 1)[1])
        swatch = base64.b64decode(png_data_url(32, 32, seed=8).split(",", 1)[1])
        request = {
            "model": PUBLIC_MODEL,
            "state": {"note": "A product photo."},
            "assets": {
                "<photo>": {"type": "image", "part": "photo"},
                "<swatch>": {"type": "image", "part": "swatch"},
            },
            "questions": {
                "where": {"type": "box", "instructions": "Find the mug in <photo>."},
                "colour": {
                    "type": "choice",
                    "instructions": "Which colour is closest to <swatch>?",
                    "criteria": {"warm": "a warm colour", "cool": "a cool colour"},
                },
                "pick": {
                    "type": "choice",
                    "instructions": "Which is the swatch?",
                    "criteria": {"a": "<swatch>", "b": "plain grey"},
                },
            },
        }
        body, content_type = multipart(request, {"photo": photo, "swatch": swatch})
        status, answered = raw_post(
            self.base,
            "/v1/systemone",
            gzip.compress(body),
            self.headers(**{"Content-Type": content_type, "Content-Encoding": "gzip"}),
        )
        self.assertEqual(status, 200, answered)
        box = answered["answers"]["where"]
        self.assertEqual(set(box), {"type", "box", "confidence"}, box)
        edges = box["box"]
        self.assertTrue(0 <= edges["top"] < edges["bottom"] <= 1, box)
        self.assertTrue(0 <= edges["left"] < edges["right"] <= 1, box)
        self.assertIn(
            answered["answers"]["colour"]["choice"], {"warm", "cool"}, answered
        )
        self.assertIn(answered["answers"]["pick"]["choice"], {"a", "b"}, answered)
        status, clip = self.public(
            "/v1/decisions",
            {
                "model": PUBLIC_MODEL,
                "state": "A short clip.",
                "videos": [TINY_MP4],
                "questions": {
                    "at": {
                        "type": "box",
                        "instructions": "The blue square.",
                        "at": 0.5,
                    },
                    "track": {
                        "type": "track",
                        "instructions": "The blue square.",
                        "rate": 2,
                    },
                },
            },
        )
        self.assertEqual(status, 200, clip)
        self.assertIn("time", clip["answers"]["at"], clip)
        track = clip["answers"]["track"]["track"]
        times = [entry["time"] for entry in track]
        self.assertTrue(track and times == sorted(times), clip)
        self.assertTrue(all(0 <= entry["visible"] <= 1 for entry in track), clip)
        state = "w " * 150
        question = {"type": "noul", "instructions": "Is it long?"}
        counts = []
        for count in (1, 2):
            status, response = self.public(
                "/v1/systemone",
                {
                    "model": PUBLIC_MODEL,
                    "state": state,
                    "questions": {f"q{i}": question for i in range(count)},
                },
            )
            self.assertEqual(status, 200, response)
            counts.append(response["usage"]["input_tokens"])
        # The state counts once, each further question only its own block.
        self.assertLess(0, counts[1] - counts[0])
        self.assertLess(counts[1] - counts[0], counts[0] / 3)

    def test_the_errors_and_limits_follow_d3_turbos_contract(self):
        self.listed()
        still = {
            "model": PUBLIC_MODEL,
            "images": [png_data_url(64, 48, seed=9)],
            "state": "A still.",
            "questions": {
                "dog": {"type": "track", "instructions": "The dog."},
                "still": {"type": "noul", "instructions": "Is it a still?"},
            },
        }
        status, decisions = self.public("/v1/decisions", still)
        self.assertEqual(status, 200, decisions)
        self.assertEqual(
            decisions["answers"]["dog"]["reason"], "video_needed", decisions
        )
        self.assertIn("noul", decisions["answers"]["still"], decisions)
        status, systemone = self.public("/v1/systemone", still)
        self.assertEqual(status, 422, systemone)
        self.assertEqual(
            {
                k: systemone["error"][k]
                for k in ("code", "question_error", "reason", "question")
            },
            {
                "code": "question_failed",
                "question_error": "invalid_question",
                "reason": "video_needed",
                "question": "dog",
            },
            systemone,
        )
        # The fixture's tokenizer makes about five tokens of "word ", so this state is just over the validated
        # length; one far longer does not finish within the 30-s System One deadline on CPU.
        long = {
            "model": PUBLIC_MODEL,
            "state": "word " * (self.TURBO_VALIDATED_TOKENS // 4),
            "questions": {"long": {"type": "noul", "instructions": "Is this long?"}},
        }
        status, refused = self.public("/v1/systemone", long)
        self.assertEqual(
            (status, refused["error"]["reason"]), (400, "question_tokens"), refused
        )
        long_context = {"long_context": True}
        status, answered = self.public(
            "/v1/systemone", {**long, "options": long_context}
        )
        self.assertEqual(status, 200, answered)
        cap = self.TURBO_MAX_TOKENS
        many = {
            "model": PUBLIC_MODEL,
            "state": "short",
            "questions": {
                f"q{i}": {"type": "noul", "instructions": "word " * 2000}
                for i in range(cap // 2000 + 1)
            },
            "options": long_context,
        }
        status, over = self.public("/v1/systemone", many)
        self.assertEqual(status, 400, over)
        self.assertEqual(
            (over["error"]["code"], over["error"]["reason"], over["error"]["limit"]),
            ("invalid_request", "request_tokens", cap),
            over,
        )


class TestEngineModeD3Ultra(EngineStack):
    """A package with d3-ultra's card through the public API: a text-only request lists its questions in front of its
    state, so each question it adds also adds its line to that state; a request with images keeps d3-turbo's layout.
    """

    fixture = ("decision3", "ultra")

    def answered(self, body):
        status, response = self.public("/v1/systemone", {"model": PUBLIC_MODEL, **body})
        self.assertEqual(status, 200, response)
        return response

    def test_a_text_request_lists_its_questions_before_the_state(self):
        first = {"type": "noul", "instructions": "Is the request urgent?"}
        second = {
            "type": "choice",
            "instructions": "Which team handles it?",
            "criteria": {"billing": "Billing", "tech": "Technical support"},
        }
        added, answers = {}, {}
        for kind, media in (
            ("text", {}),
            ("image", {"images": [png_data_url(64, 48, seed=11)]}),
        ):
            one = self.answered(
                {"state": "A refund request.", "questions": {"first": first}, **media}
            )
            two = self.answered(
                {
                    "state": "A refund request.",
                    "questions": {"first": first, "second": second},
                    **media,
                }
            )
            added[kind] = two["usage"]["input_tokens"] - one["usage"]["input_tokens"]
            answers[kind] = (one["answers"]["first"], two["answers"]["first"])
        # The second question's block costs the same either way; in a text request its line in the list counts too.
        self.assertGreater(added["text"], added["image"], added)
        self.assertGreater(added["image"], 0, added)
        # The list is part of the text request's state, so the first question reads a different state.
        self.assertNotEqual(
            answers["text"][0]["noul"], answers["text"][1]["noul"], answers
        )


if __name__ == "__main__":
    unittest.main()
