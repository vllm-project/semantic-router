from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_MK_PATH = REPO_ROOT / "tools" / "make" / "docker.mk"

# `vllm-sr-router` names the router container role, not an image. A tag built
# from DOCKER_REGISTRY here produced an image no consumer referenced.
ORPHAN_ROUTER_TAG = "$(DOCKER_REGISTRY)/vllm-sr-router:$(DOCKER_TAG)"
CANONICAL_PREFIX = "ghcr.io/vllm-project/semantic-router"


def _docker_mk_target(name: str) -> str:
    """Return one Make target's declaration lines plus its recipe body."""
    lines = DOCKER_MK_PATH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}:"))
    end = start + 1
    while end < len(lines) and (
        not lines[end].strip()
        or lines[end].startswith("\t")
        or lines[end].startswith(f"{name}:")
    ):
        end += 1
    return "\n".join(lines[start:end])


def _mk_default(name: str) -> str:
    """Return the `?=` default assigned to a Make variable."""
    content = DOCKER_MK_PATH.read_text(encoding="utf-8")
    match = re.search(rf"^{name} \?= (.+)$", content, re.MULTILINE)
    if match is None:
        raise AssertionError(f"{name} has no ?= default in docker.mk")
    return match.group(1)


class DockerMakeSurfaceTest(unittest.TestCase):
    """Contract checks for the developer-facing surface of tools/make/docker.mk."""

    def test_router_image_targets_use_the_canonical_image_name(self):
        content = DOCKER_MK_PATH.read_text(encoding="utf-8")
        self.assertNotIn(ORPHAN_ROUTER_TAG, content)

        build = _docker_mk_target("docker-build-vllm-sr-router")
        self.assertIn("-t $(VLLM_SR_ROUTER_IMAGE)", build)
        self.assertIn("$(VLLM_SR_BUILD_ARGS)", build)

        push = _docker_mk_target("docker-push-vllm-sr-router")
        self.assertIn("push $(VLLM_SR_ROUTER_IMAGE)", push)

    def test_runtime_images_derive_from_the_configured_registry(self):
        registry = _mk_default("DOCKER_REGISTRY")
        tag = _mk_default("DOCKER_TAG")
        self.assertEqual(registry, CANONICAL_PREFIX)

        for name, repo in (
            ("VLLM_SR_IMAGE", "vllm-sr"),
            ("VLLM_SR_IMAGE_ROCM", "vllm-sr-rocm"),
            ("VLLM_SR_IMAGE_CUDA", "vllm-sr-cuda"),
        ):
            with self.subTest(image=name):
                rhs = _mk_default(name)
                # The documented registry/tag overrides must reach every
                # runtime image variant, including the platform ones.
                self.assertIn("$(DOCKER_REGISTRY)", rhs)
                self.assertIn("$(DOCKER_TAG)", rhs)
                # Expanding the defaults must reproduce the previously
                # hard-coded canonical path exactly.
                expanded = rhs.replace("$(DOCKER_REGISTRY)", registry).replace(
                    "$(DOCKER_TAG)", tag
                )
                self.assertEqual(expanded, f"{CANONICAL_PREFIX}/{repo}:{tag}")

    def test_router_image_inherits_the_registry_derived_variants(self):
        self.assertEqual(
            _mk_default("VLLM_SR_ROUTER_IMAGE_DEFAULT"), "$(VLLM_SR_IMAGE)"
        )
        self.assertEqual(
            _mk_default("VLLM_SR_ROUTER_IMAGE_ROCM"), "$(VLLM_SR_IMAGE_ROCM)"
        )
        self.assertEqual(
            _mk_default("VLLM_SR_ROUTER_IMAGE_CUDA"), "$(VLLM_SR_IMAGE_CUDA)"
        )
        self.assertEqual(
            _mk_default("VLLM_SR_ROUTER_IMAGE"), "$(VLLM_SR_ROUTER_IMAGE_DEFAULT)"
        )

    def test_docker_test_llm_katan_owns_its_container_lifecycle(self):
        content = DOCKER_MK_PATH.read_text(encoding="utf-8")
        target = _docker_mk_target("docker-test-llm-katan")

        self.assertIn("LLM_KATAN_HOST_PORT ?= 8000", content)
        self.assertIn("LLM_KATAN_TEST_CONTAINER ?= llm-katan-docker-test", content)
        self.assertIn("docker-test-llm-katan: docker-build-llm-katan", target)

        # Detached start under a dedicated name, so the check cannot collide
        # with the `llm-katan` container the memory integration stack owns.
        self.assertIn("run -d", target)
        self.assertIn("--name $(LLM_KATAN_TEST_CONTAINER)", target)
        # The image CMD downloads a real model; the test must not.
        self.assertIn("--model dummy", target)
        self.assertIn("--backend echo", target)
        # Readiness on /health must gate the /v1/models assertion.
        self.assertIn("/health", target)
        self.assertIn("/v1/models", target)
        self.assertIn("logs $(LLM_KATAN_TEST_CONTAINER)", target)

        # Bounded retry with a per-request ceiling, not an unbounded wait.
        self.assertRegex(target, r"-lt\s+\d+")
        self.assertIn("--max-time", target)

        handlers = re.findall(r"trap '([^']*)'", target)
        self.assertTrue(handlers, "no trap-based cleanup")
        self.assertTrue(
            any("stop $(LLM_KATAN_TEST_CONTAINER)" in h for h in handlers),
            "cleanup does not stop the container",
        )
        self.assertTrue(
            any("rm $(LLM_KATAN_TEST_CONTAINER)" in h for h in handlers),
            "cleanup does not remove the container",
        )
        # A signal handler must terminate the shell; otherwise an interrupted
        # run falls through and re-enters the readiness loop.
        self.assertTrue(any("exit" in h for h in handlers))

        self.assertNotIn("docker run", target)

    def test_container_liveness_match_is_not_a_regex(self):
        target = _docker_mk_target("docker-test-llm-katan")

        # The container name is user-overridable, so it must never be read as
        # a pattern.
        self.assertIn('grep -qxF "$(LLM_KATAN_TEST_CONTAINER)"', target)
        self.assertNotIn('"^$(LLM_KATAN_TEST_CONTAINER)', target)

    def test_docker_help_documents_the_katan_knobs(self):
        help_target = _docker_mk_target("docker-help")

        self.assertIn("LLM_KATAN_HOST_PORT", help_target)
        self.assertIn("LLM_KATAN_TEST_CONTAINER", help_target)

    def test_docker_run_llm_katan_stays_foreground(self):
        target = _docker_mk_target("docker-run-llm-katan")

        self.assertIn("run --rm -p 8000:8000", target)
        self.assertNotIn(" run -d ", target)


if __name__ == "__main__":
    unittest.main()
